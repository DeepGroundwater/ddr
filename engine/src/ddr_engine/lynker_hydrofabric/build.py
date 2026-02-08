"""Build functions for adjacency matrices from Lynker Hydrofabric flowpaths."""

import sqlite3
from pathlib import Path

import numpy as np
import polars as pl
import zarr
from scipy import sparse
from tqdm import tqdm

from ddr.geodatazoo.dataclasses import GaugeSet

from .graph import find_origin, preprocess_river_network, subset
from .io import coo_to_zarr, coo_to_zarr_group, create_coo, create_matrix


def build_lynker_hydrofabric_adjacency(
    fp: pl.LazyFrame,
    network: pl.LazyFrame,
    out_path: Path,
) -> None:
    """
    Build the large-scale CONUS adjacency matrix for the Lynker Hydrofabric.

    Parameters
    ----------
    fp : pl.LazyFrame
        Flowpaths dataframe with 'id' and 'toid' columns.
    network : pl.LazyFrame
        Network dataframe with 'id' and 'toid' columns.
    out_path : Path
        Path to save the zarr group.

    Returns
    -------
    None
    """
    matrix, ts_order = create_matrix(fp, network)
    coo_to_zarr(matrix, ts_order, out_path)


def build_lynker_hydrofabric_gages_adjacency(
    gpkg_path: Path,
    out_path: Path,
    gauge_set: GaugeSet,
    gages_out_path: Path,
) -> None:
    """
    Build per-gauge adjacency matrices for the Lynker Hydrofabric.

    Parameters
    ----------
    gpkg_path : Path
        Path to the hydrofabric geopackage.
    out_path : Path
        Path to the CONUS adjacency zarr store.
    gauge_set : GaugeSet
        Validated gauge set containing gauge information.
    gages_out_path : Path
        Path to save the gauge adjacency zarr store.

    Returns
    -------
    None

    Notes
    -----
    Creates a zarr group with one subgroup per gauge, each containing:
    - indices_0, indices_1: COO matrix indices
    - values: COO matrix values
    - order: Topological ordering of watershed boundaries
    """
    query = "SELECT id,toid,tot_drainage_areasqkm FROM flowpaths"
    conn = sqlite3.connect(gpkg_path)
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
            # print(f"Gauge: {gauge.STAID} is a headwater catchment (single reach)")
            coo = sparse.coo_matrix((len(ts_order_dict), len(ts_order_dict)), dtype=np.int8)
            subset_flowpaths = [origin]
        else:
            coo, subset_flowpaths = create_coo(connections, ts_order_dict)
        coo_to_zarr_group(
            coo=coo,
            ts_order=subset_flowpaths,
            origin=origin,
            gauge_root=gauge_root,
            conus_mapping=ts_order_dict,
        )
    conn.close()
