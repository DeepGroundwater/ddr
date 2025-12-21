"""A script for building adjacency matrices for DDR on the CONUS v2.2 Hydrofabric"""

import argparse
import sqlite3
from pathlib import Path

import polars as pl
from ddr_engine.hydrofabric import create_v2_2_conus_coo_to_zarr, create_v2_2_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a lower triangular adjacency matrix from hydrofabric data."
    )
    parser.add_argument(
        "pkg",
        type=Path,
        help="Path to the hydrofabric geopackage.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to save the zarr group. Defaults to current working directory with name appended.",
    )
    args = parser.parse_args()

    if args.path is None:
        raise FileNotFoundError("Path not provided for zarr group outputs")
    else:
        out_path = Path(args.path) / "hydrofabric_v2.2_adjacency.zarr"
        out_path.parent.mkdir(exist_ok=True)
    if out_path.exists():
        print(f"Cannot create zarr store {args.path}. One already exists")
        exit(1)

    # Read hydrofabric geopackage using sqlite
    uri = "sqlite://" + str(args.pkg)
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
    matrix, ts_order = create_v2_2_matrix(fp, network)
    create_v2_2_conus_coo_to_zarr(matrix, ts_order, out_path)
    conn.close()
