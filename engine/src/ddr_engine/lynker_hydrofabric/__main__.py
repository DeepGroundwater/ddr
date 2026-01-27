"""CLI entrypoint for building Lynker Hydrofabric adjacency matrices.

Usage:
    python -m ddr_engine.lynker_hydrofabric <pkg> [--path PATH] [--gages GAGES]

@author Nels Frazier
@author Tadd Bindas
@date Jan 26 2026
"""

import argparse
import sqlite3
from pathlib import Path

import polars as pl

from .build import build_lynker_hydrofabric_adjacency, build_lynker_hydrofabric_gages_adjacency


def main() -> None:
    """Main function for the module"""
    parser = argparse.ArgumentParser(
        description="Create a lower triangular adjacency matrix from hydrofabric data."
    )
    parser.add_argument(
        "pkg",
        type=Path,
        help="Path to the hydrofabric geopackage.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default="data/",
        help="Path to save the zarr group. Defaults to current working directory with name appended.",
    )

    parser.add_argument(
        "--gages",
        type=Path,
        default=Path("streamflow_datasets/gage_info/dhbv2_gages.csv"),
        help="The gauges CSV file containing the training locations. Only needed if gage adjacency matrices are being made.",
    )
    args = parser.parse_args()

    if args.path is None:
        raise FileNotFoundError("Path not provided for zarr group outputs")
    else:
        out_path = Path(args.path) / "hydrofabric_v2.2_conus_adjacency.zarr"
        out_path.parent.mkdir(exist_ok=True)

        if args.gages is not None:
            gages_out_path = Path(args.path) / "hydrofabric_v2.2_gages_conus_adjacency.zarr"
            if gages_out_path.exists():
                print(f"Cannot create zarr store {gages_out_path}. One already exists")
                exit(1)

    if out_path.exists():
        print(f"Cannot create zarr store {args.path}. One already exists")
        exit(1)

    # Read hydrofabric geopackage using sqlite
    # uri = "sqlite://" + str(args.pkg)
    query = "SELECT id,toid FROM flowpaths"
    conn = sqlite3.connect(args.pkg)
    fp = pl.read_database(query=query, connection=conn)

    # Make sure wb-0 exists as a flowpath -- this is effectively
    # the terminal node of all hydrofabric terminals -- use this if not using ghosts
    # If you want to have each independent network have its own terminal ghost-N
    # identifier, then you would need to actually drop all wb-0 instances in
    # the network table toid column and replace them with null values...
    fp = fp.extend(pl.DataFrame({"id": ["wb-0"], "toid": [None]})).lazy()
    # build the network table
    query = "SELECT id,toid FROM network"
    # network = pl.read_database_uri(query=query, uri=uri, engine="adbc").lazy()
    network = pl.read_database(query=query, connection=conn).lazy()
    network = network.filter(pl.col("id").str.starts_with("wb-").not_())
    build_lynker_hydrofabric_adjacency(fp, network, out_path)
    conn.close()

    if args.gages is not None:
        from ddr.geodatazoo.dataclasses import Gauge, GaugeSet, validate_gages

        print("Creating Gages Adjacency Matrix")
        gauge_set: GaugeSet = validate_gages(args.gages, type=Gauge)
        build_lynker_hydrofabric_gages_adjacency(args.pkg, out_path, gauge_set, gages_out_path)
    print(f"Gage Adjacency matrices for v2.2 were created at: {out_path}")


if __name__ == "__main__":
    main()
