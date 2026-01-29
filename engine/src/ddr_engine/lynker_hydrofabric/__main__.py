"""CLI entrypoint for building Lynker Hydrofabric adjacency matrices.

Usage:
    python -m ddr_engine.lynker_hydrofabric <pkg> [--path PATH] [--gages GAGES]
"""

import argparse
import sqlite3
from pathlib import Path

import polars as pl

from .build import build_lynker_hydrofabric_adjacency, build_lynker_hydrofabric_gages_adjacency


def main() -> None:
    """The main function for the Lynker Hydrofabric engine."""
    parser = argparse.ArgumentParser(
        description="Create lower triangular adjacency matrices from Lynker Hydrofabric data."
    )
    parser.add_argument(
        "pkg",
        type=Path,
        help="Path to the hydrofabric geopackage.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/"),
        help="Path to save the zarr group. Defaults to 'data/'.",
    )
    parser.add_argument(
        "--gages",
        type=Path,
        default=None,
        help="The gauges CSV file containing the training locations.",
    )
    args = parser.parse_args()

    print(f"Reading Lynker Hydrofabric data from {args.pkg}")

    # Read hydrofabric geopackage using sqlite
    query = "SELECT id,toid FROM flowpaths"
    conn = sqlite3.connect(args.pkg)
    fp = pl.read_database(query=query, connection=conn)

    # Make sure wb-0 exists as a flowpath -- this is effectively
    # the terminal node of all hydrofabric terminals -- use this if not using ghosts
    # If you want to have each independent network have its own terminal ghost-N
    # identifier, then you would need to actually drop all wb-0 instances in
    # the network table toid column and replace them with null values...
    fp = fp.extend(pl.DataFrame({"id": ["wb-0"], "toid": [None]})).lazy()

    # Build the network table
    query = "SELECT id,toid FROM network"
    network = pl.read_database(query=query, connection=conn).lazy()
    network = network.filter(pl.col("id").str.starts_with("wb-").not_())

    out_path = args.path / "hydrofabric_v2.2_conus_adjacency.zarr"
    build_lynker_hydrofabric_adjacency(fp, network, out_path)
    conn.close()

    if args.gages is not None:
        from ddr.geodatazoo.dataclasses import Gauge, GaugeSet, validate_gages

        gauge_set: GaugeSet = validate_gages(args.gages, type=Gauge)
        gages_out_path = args.path / "hydrofabric_v2.2_gages_conus_adjacency.zarr"
        build_lynker_hydrofabric_gages_adjacency(args.pkg, out_path, gauge_set, gages_out_path)


if __name__ == "__main__":
    main()
