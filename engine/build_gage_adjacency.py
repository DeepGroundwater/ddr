#!/usr/bin/env python

"""
@author Tadd Bindas

@date June 20 2025
@version 0.1

A script to build subset COO matrices from the conus_adjacency.zarr
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from Gauges import Gauge, GaugeSet, validate_gages
from pyiceberg.catalog import load_catalog
from scipy import sparse


def find_origin(gauge: Gauge, fp: pl.LazyFrame, network: pl.LazyFrame) -> np.ndarray:
    """A function to query the network Lazyframe for a gauge ID

    Parameters
    ----------
    gauge: Gauge
        A pydantic object containing gauge information
    network: pl.LazyFrame
        The hydrofabric network table

    Returns
    -------
    np.ndarray
        The flowpaths associated with the gauge ID
    """
    flowpaths = (
        network.filter(
            pl.col("hl_uri") == f"gages-{gauge.STAID.zfill(8)}"  # Finding the matching gauge
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


def subset(origin: str, wb_network_dict: dict[str, list[str]]) -> list[str]:
    """Subsets the hydrofabric to find all upstream watershed boundaries upstream of the origin fp

    Parameters
    ----------
    origin: str
        The starting point from which to find upstream connections from
    wb_network_dict: dict[str, list[str]]
        a dictionary which maps toid -> list[id] for upstream subsets

    Returns
    -------
    list[str]
        The watershed boundaries that make up the subset
    """
    upstream_segments = set()

    def trace_upstream_recursive(current_id: str) -> None:
        """Recursively trace upstream from current_id."""
        if current_id in upstream_segments:
            return
        upstream_segments.add(current_id)

        # Find all segments that flow into current_id
        if current_id in wb_network_dict:
            for upstream_id in wb_network_dict[current_id]:
                if upstream_id not in upstream_segments:
                    trace_upstream_recursive(upstream_id)

    trace_upstream_recursive(origin)
    upstream_segments.add(origin)
    return list(upstream_segments)


def create_coo(ts_subset_order: list[str], conus_root: zarr.Group) -> tuple[sparse.coo, list[str]]:
    """A function to create a coo matrix out of the ts_ordering from the conus_adjacency matrix indices

    Parameters
    ----------
    ts_subset_order: list[str]
        The ordering of the watershed boundaries from the gauge subset
    conus_root: zarr.Group
        The zarr store representing the CONUS adjacency matrix

    Returns
    -------
    sparse.coo
        The sparse coo matrix from subset indexed from the CONUS adjacency matrix
    list[str]
        The topological sorted ordering from the subset
    """
    raise NotImplementedError


def coo_to_zarr_group(coo: sparse.coo_matrix, ts_order: list[str], out_path: Path, gauge_id: str) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[str]
        Topological sort order of flowpaths.
    out_path : Path | str | None, optional
        Path to save the zarr group. If None, defaults to current working directory with name appended.

    Returns
    -------
    None
    """
    # Converting to a sparse COO matrix, and saving the output in many arrays within a zarr v3 group
    store = zarr.storage.LocalStore(root=out_path)
    if out_path.exists():
        root = zarr.open_group(store=store)
    else:
        root = zarr.create_group(store=store)

    zarr_order = np.array([int(float(_id.split("-")[1])) for _id in ts_order], dtype=np.int32)

    gauge_root = root.create_group(name=gauge_id)
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

    # Read in hydrofabric
    namespace = "hydrofabric"
    catalog = load_catalog(namespace)
    fp = catalog.load_table("hydrofabric.flowpaths").to_polars()
    network = catalog.load_table("hydrofabric.network").to_polars()
    print("Creating network dictionary")
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

    # Read in conus_adjacency.zarr
    conus_root = zarr.open_group(store=conus_path)

    for gauge in gauge_set.gauges:
        try:
            origin = find_origin(gauge, fp, network)
            subset_fp = subset(origin, wb_network_dict)
            coo, ts_subset_order = create_coo(subset_fp, conus_root)
            coo_to_zarr_group(coo, ts_subset_order, out_path)
        except KeyError:
            print(f"Cannot find gauge: {gauge}. Skipping")
            continue
