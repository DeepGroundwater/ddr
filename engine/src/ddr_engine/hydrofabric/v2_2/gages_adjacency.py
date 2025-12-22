"""
@author Tadd Bindas

@date June 20 2025
@version 0.1

A script to build subset COO matrices from the conus_adjacency.zarr
"""

import numpy as np
import polars as pl
import zarr
from scipy import sparse

from ddr.dataset import Gauge


def find_origin(gauge: Gauge, fp: pl.LazyFrame, network: pl.LazyFrame) -> np.ndarray:
    """A function to query the network Lazyframe for a gauge ID

    Parameters
    ----------
    gauge: Gauge
        A pydantic object containing gauge information
    fp: pl.LazyFrame
        The hydrofabric flowpaths table
    network: pl.LazyFrame
        The hydrofabric network table

    Returns
    -------
    np.ndarray
        The flowpaths associated with the gauge ID
    """
    try:
        flowpaths = (
            network.filter(
                pl.col("hl_uri") == f"gages-{gauge.STAID}"  # Finding the matching gauge
            )
            .select(
                pl.col("id")  # Select the `wb` values
            )
            .collect()
            .to_numpy()
            .squeeze()
        )
        if flowpaths.size > 1:
            return (
                fp.filter(
                    pl.col("id").is_in(flowpaths)  # finds the rows with matching IDs
                )
                .with_columns(
                    (pl.col("tot_drainage_areasqkm") - gauge.DRAIN_SQKM)
                    .abs()
                    .alias("diff")  # creates a new column with the DA diference from the USGS Gauge
                )
                .sort("diff")
                .head(1)
                .select("id")
                .collect()
                .item()
            )  # Selects the flowpath with the smallest difference
        else:
            return flowpaths.item()
    except ValueError as e:
        raise ValueError from e


def subset(origin: str, wb_network_dict: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Subsets the hydrofabric to find all upstream watershed boundaries upstream of the origin fp

    Parameters
    ----------
    origin: str
        The starting point from which to find upstream connections from
    wb_network_dict: dict[str, list[str]]
        a dictionary which maps toid -> list[id] for upstream subsets

    Returns
    -------
    list[tuple[str, str]]
        The watershed boundary connections that make up the subset. Note [0] is the toid and [1] is the from_id
    """
    upstream_segments = set()
    connections = []

    def trace_upstream_recursive(current_id: str) -> None:
        """Recursively trace upstream from current_id."""
        if current_id in upstream_segments:
            return
        upstream_segments.add(current_id)

        # Find all segments that flow into current_id
        if current_id in wb_network_dict:
            for upstream_id in wb_network_dict[current_id]:
                connections.append(
                    (current_id, upstream_id)
                )  # Row is where the flow is going, col is where the flow is coming from
                if upstream_id not in upstream_segments:
                    trace_upstream_recursive(upstream_id)

    trace_upstream_recursive(origin)
    return connections


def create_coo(
    connections: list[tuple[str, str]], conus_mapping: dict[str, int]
) -> tuple[sparse.coo_matrix, list[str]]:
    """A function to create a coo matrix out of the ts_ordering from the conus_adjacency matrix indices

    Parameters
    ----------
    connections: list[tuple[str, str]]
        The connections of the watershed boundaries from the gauge subset
    conus_mapping: dict[str, int]
        The mapping of watershed boundaries to their conus index (topo sorted already)

    Returns
    -------
    sparse.coo
        The sparse coo matrix from subset indexed from the CONUS adjacency matrix
    list[str]
        The topological sorted ordering from the subset
    """
    row_idx = []
    col_idx = []
    for flowpaths in connections:
        try:
            row_idx.append(conus_mapping[flowpaths[0]])
        except KeyError:
            flowpath_id = f"wb-{int(float(flowpaths[0].split('-')[1]))}"
            row_idx.append(conus_mapping[flowpath_id])
        try:
            col_idx.append(conus_mapping[flowpaths[1]])
        except KeyError:
            flowpath_id = f"wb-{int(float(flowpaths[1].split('-')[1]))}"
            col_idx.append(conus_mapping[flowpath_id])
    coo = sparse.coo_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(len(conus_mapping), len(conus_mapping)),
        dtype=np.int8,
    )
    all_flowpaths = {item for connection in connections for item in connection}
    assert np.all(coo.row >= coo.col), "Matrix is not lower triangular"
    return coo, all_flowpaths


def preprocess_river_network(network: pl.LazyFrame) -> dict[str, list[str]]:
    """Preprocesses the network dictionary to find all connections

    Connections are ordered by the key being the toid, and the values being ids (upstream segments)

    Parameters
    ----------
    network: pl.LazyFrame

    Returns
    -------
    dict[str, list[str]]
        A dictionary which maps downstream segments to their upstream values in a one -> many relationship
    """
    network_dict = (
        network.filter(pl.col("toid").is_not_null())
        .group_by("toid")
        .agg(pl.col("id").alias("upstream_ids"))
        .collect()
    )

    # Create a lookup for nexus -> downstream wb connections
    nexus_downstream = (
        network.filter(pl.col("id").str.starts_with("nex-"))
        .filter(pl.col("toid").str.starts_with("wb-"))
        .select(["id", "toid"])
        .rename({"id": "nexus_id", "toid": "downstream_wb"})
    ).collect()

    # Explode the upstream_ids to get one row per connection
    connections = network_dict.with_row_index().explode("upstream_ids")

    # Separate wb-to-wb connections (keep as-is)
    wb_to_wb = (
        connections.filter(pl.col("upstream_ids").str.starts_with("wb-"))
        .filter(pl.col("toid").str.starts_with("wb-"))
        .select(["toid", "upstream_ids"])
    )

    # Handle nexus connections: wb -> nex -> wb becomes wb -> wb
    wb_to_nexus = (
        connections.filter(pl.col("upstream_ids").str.starts_with("wb-"))
        .filter(pl.col("toid").str.starts_with("nex-"))
        .join(nexus_downstream, left_on="toid", right_on="nexus_id", how="inner")
        .select(["downstream_wb", "upstream_ids"])
        .rename({"downstream_wb": "toid"})
    )

    # Combine both types of connections
    wb_connections = pl.concat([wb_to_wb, wb_to_nexus]).unique()

    # Group back to dictionary format
    wb_network_result = wb_connections.group_by("toid").agg(pl.col("upstream_ids")).unique()
    wb_network_dict = {row["toid"]: row["upstream_ids"] for row in wb_network_result.iter_rows(named=True)}
    return wb_network_dict


def coo_to_zarr_group(
    coo: sparse.coo_matrix,
    ts_order: list[str],
    origin: str,
    gauge_root: zarr.Group,
    conus_mapping: dict[str, int],
) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[str]
        Topological sort order of flowpaths.
    origin: str
        The origin edge of the flow network
    gauge_root: zarr.Group
        The zarr group for the subset COO matrix
    conus_mapping: dict[str, int]
        The index mapping from watershed boundary to it's position in the array. Ordering is determined through toposort.

    Returns
    -------
    None
    """
    # Converting to a sparse COO matrix, and saving the output in many arrays within a zarr v3 group
    zarr_order = np.array([int(float(_id.split("-")[1])) for _id in ts_order], dtype=np.int32)

    indices_0 = gauge_root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = gauge_root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = gauge_root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order = gauge_root.create_array(name="order", shape=zarr_order.shape, dtype=zarr_order.dtype)
    indices_0[:] = coo.row
    indices_1[:] = coo.col
    values[:] = coo.data
    order[:] = zarr_order

    gauge_root.attrs["format"] = "COO"
    gauge_root.attrs["shape"] = list(coo.shape)
    gauge_root.attrs["gage_wb"] = origin
    gauge_root.attrs["gage_idx"] = conus_mapping[origin]
    gauge_root.attrs["data_types"] = {
        "indices_0": coo.row.dtype.__str__(),
        "indices_1": coo.col.dtype.__str__(),
        "values": coo.data.dtype.__str__(),
    }
