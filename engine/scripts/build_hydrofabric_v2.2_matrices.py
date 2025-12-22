"""A script for building adjacency matrices for DDR on the CONUS v2.2 Hydrofabric"""

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from ddr_engine.hydrofabric import (
    coo_to_zarr,
    coo_to_zarr_group,
    create_coo,
    create_matrix,
    find_origin,
    preprocess_river_network,
    subset,
)
from tqdm import tqdm

from ddr.dataset import GaugeSet, validate_gages

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

    parser.add_argument(
        "--gages",
        type=Path,
        required=False,
        default=None,
        help="The gauges CSV file containing the training locations. Only needed if gage adjacency matrices are being made.",
    )
    args = parser.parse_args()

    if args.path is None:
        raise FileNotFoundError("Path not provided for zarr group outputs")
    else:
        out_path = Path(args.path) / "hydrofabric_v2.2_conus_adjacency.zarr"
        out_path.parent.mkdir(exist_ok=True)

        if args.gages is not None:
            gage_path = Path(args.gages)
            gauge_set: GaugeSet = validate_gages(gage_path)
            gages_out_path = Path(args.path) / "hydrofabric_v2.2_gages_conu_adjacency.zarr"
            if gages_out_path.exists():
                print(f"Cannot create zarr store {gages_out_path}. One already exists")
                exit(1)
        else:
            gages_out_path = None

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
    matrix, ts_order = create_matrix(fp, network)
    coo_to_zarr(matrix, ts_order, out_path)
    conn.close()

    if gages_out_path is not None:
        print("Creating Gages Adjacency Matrix")

        query = "SELECT id,toid,tot_drainage_areasqkm FROM flowpaths"
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
        print("Read CONUS zarr store")
        conus_root = zarr.open_group(store=out_path)
        ts_order = conus_root["order"][:]
        ts_order = np.array([f"wb-{_id}" for _id in ts_order])
        ts_order_dict = {wb_id: idx for idx, wb_id in enumerate(ts_order)}

        # Create local zarr store
        store = zarr.storage.LocalStore(root=gages_out_path)
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
                print(f"Cannot find gauge: {gauge.STAID}. Skipping write")
                root.__delitem__(gauge.STAID)
                continue
            connections = subset(origin, wb_network_dict)
            if len(connections) == 0:
                print(
                    f"Gauge: {gauge.STAID} is a headwater catchment with no upstream catchments. Skipping write"
                )
                root.__delitem__(gauge.STAID)
                continue
            coo, subset_flowpaths = create_coo(connections, ts_order_dict)
            coo_to_zarr_group(
                coo=coo,
                ts_order=subset_flowpaths,
                origin=origin,
                gauge_root=gauge_root,
                conus_mapping=ts_order_dict,
            )
        conn.close()

        print(f"Gage Adjacency matrices for v2.2 were created at: {out_path}")
