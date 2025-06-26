#!/usr/bin/env python

"""
@author Tadd Bindas

@date June 20 2025
@version 0.1

A script to build subset COO matrices from the conus_adjacency.zarr
"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from scipy import sparse
from tqdm import tqdm

from ddr import Gauge, GaugeSet, validate_gages


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


def subset(origin: str, wb_network_dict: dict[str, list[str]]) -> list[tuple[str]]:
    """Subsets the hydrofabric to find all upstream watershed boundaries upstream of the origin fp

    Parameters
    ----------
    origin: str
        The starting point from which to find upstream connections from
    wb_network_dict: dict[str, list[str]]
        a dictionary which maps toid -> list[id] for upstream subsets

    Returns
    -------
    list[tuple[str]]
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
    connections: list[tuple[str]], conus_mapping: dict[str, int]
) -> tuple[sparse.coo_matrix, list[str]]:
    """A function to create a coo matrix out of the ts_ordering from the conus_adjacency matrix indices

    Parameters
    ----------
    connections: list[tuple[str]]
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

    root.attrs["format"] = "COO"
    root.attrs["shape"] = list(coo.shape)
    root.attrs["gage_wb"] = origin
    root.attrs["gage_idx"] = conus_mapping[origin]
    root.attrs["data_types"] = {
        "indices_0": coo.row.dtype.__str__(),
        "indices_1": coo.col.dtype.__str__(),
        "values": coo.data.dtype.__str__(),
    }


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a lower triangular adjacency matrix from hydrofabric data."
    )
    parser.add_argument(
        "pkg",
        type=Path,
        help="Path to the hydrofabric geopackage.",
    )
    parser.add_argument(
        "gages",
        type=Path,
        help="The gauges CSV file containing the training locations",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to save the gages group. Defaults to current working directory",
    )
    parser.add_argument(
        "--conus-adj",
        type=Path,
        required=True,
        default=None,
        help="Path where the conus adjacency matrix is stored. If non existent, please run `adjacency.py`",
    )
    args = parser.parse_args()

    if args.path is None:
        out_path = Path.cwd() / "observation_adjacency.zarr"
    else:
        out_path = Path(args.path)

    if args.conus_adj is None:
        conus_path = Path.cwd() / "conus_adjacency.zarr"
    else:
        conus_path = Path(args.conus_adj)
    if conus_path.exists() is False:
        raise FileNotFoundError(f"Cannot find {conus_path}")

    gage_path = Path(args.gages)
    if gage_path.exists():
        gauge_set: GaugeSet = validate_gages(gage_path)
    else:
        raise FileNotFoundError("Can't find the Gauge Information file")

    # Read hydrofabric geopackage using sqlite
    uri = "sqlite://" + str(args.pkg)
    query = "SELECT id,toid,tot_drainage_areasqkm FROM flowpaths"
    # fp = pl.read_database_uri(query=query, uri=uri, engine="adbc")
    # Using adbc is about 2 seconds faster than using the sqlite3 connection
    conn = sqlite3.connect(args.pkg)
    flowpaths_schema = {
        "id": pl.String,  # String type for IDs
        "toid": pl.String,  # String type for downstream IDs (can be null)
        "tot_drainage_areasqkm": pl.Float64,  # the total drainage area for a flowpath
    }
    fp = pl.read_database(query=query, connection=conn, schema_overrides=flowpaths_schema).lazy()

    # build the network table
    query = "SELECT id,toid,hl_uri FROM network"
    network_schema = {
        "id": pl.String,  # String type for IDs
        "toid": pl.String,  # String type for downstream IDs
        "hl_uri": pl.String,  # String type for URIs (handles mixed content)
    }
    # network = pl.read_database_uri(query=query, uri=uri, engine="adbc").lazy()
    network = pl.read_database(query=query, connection=conn, schema_overrides=network_schema).lazy()

    print("Preprocessing network Table")
    wb_network_dict = preprocess_river_network(network)

    # Read in conus_adjacency.zarr
    conus_root = zarr.open_group(store=conus_path)
    ts_order = conus_root["order"][:]
    ts_order = np.array([f"wb-{_id}" for _id in ts_order])
    ts_order_dict = {wb_id: idx for idx, wb_id in enumerate(ts_order)}

    # Create local zarr store
    store = zarr.storage.LocalStore(root=out_path)
    if out_path.exists():
        root = zarr.open_group(store=store)
    else:
        root = zarr.create_group(store=store)

    for gauge in tqdm(gauge_set.gauges, desc="Creating Gauge COO matrices"):
        try:
            gauge_root = root.create_group(gauge.STAID)
        except zarr.errors.ContainsGroupError:
            print(f"Zarr Group exists for: {gauge.STAID}. Skipping write")
            continue
        try:
            origin = find_origin(gauge, fp, network)
        except ValueError:
            print(f"Cannot find gauge: {gauge.STAID}. Skipping")
            root.__delitem__(gauge.STAID)
            continue
        connections = subset(origin, wb_network_dict)
        coo, subset_flowpaths = create_coo(connections, ts_order_dict)
        coo_to_zarr_group(
            coo=coo,
            ts_order=subset_flowpaths,
            origin=origin,
            gauge_root=gauge_root,
            conus_mapping=ts_order_dict,
        )
    conn.close()
