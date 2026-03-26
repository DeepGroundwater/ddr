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


def write_flowpath_attributes(gpkg_path: Path, out_path: Path) -> None:
    """Extract flowpath physical attributes from a GeoPackage and write them to an existing zarr store.

    Reads the ``flowpath-attributes-ml`` and ``flowpaths`` layers from the
    GeoPackage and writes six new arrays to the zarr store, all aligned to the
    existing ``order`` array:

    - ``length_m`` (float32) — from column ``Length_m``
    - ``slope`` (float32) — from column ``So``
    - ``top_width`` (float32) — from column ``TopWdth``
    - ``side_slope`` (float32) — from column ``ChSlp``
    - ``muskingum_x`` (float32) — from column ``MusX``
    - ``toid`` (str) — from the ``flowpaths`` layer column ``toid``

    Parameters
    ----------
    gpkg_path : Path
        Path to the Lynker Hydrofabric GeoPackage.
    out_path : Path
        Path to the existing adjacency zarr store (must already contain ``order``).
    """
    root = zarr.open_group(store=out_path, mode="r+")
    order = root["order"][:]  # int32 array of numeric IDs

    # --- flowpath-attributes-ml layer ---
    conn = sqlite3.connect(gpkg_path)
    attr_query = "SELECT id, Length_m, So, TopWdth, ChSlp, MusX FROM 'flowpath-attributes-ml'"
    attr_df = pl.read_database(query=attr_query, connection=conn)

    # Build lookup: wb-{id} -> row index in attr_df
    attr_df = attr_df.with_columns(pl.col("id").str.replace("^wb-", "").cast(pl.Int32).alias("numeric_id"))
    attr_lookup = {row["numeric_id"]: idx for idx, row in enumerate(attr_df.iter_rows(named=True))}

    n = len(order)
    length_m = np.full(n, np.nan, dtype=np.float32)
    slope = np.full(n, np.nan, dtype=np.float32)
    top_width = np.full(n, np.nan, dtype=np.float32)
    side_slope = np.full(n, np.nan, dtype=np.float32)
    muskingum_x = np.full(n, np.nan, dtype=np.float32)

    attr_length_m = attr_df["Length_m"].to_numpy()
    attr_so = attr_df["So"].to_numpy()
    attr_topwdth = attr_df["TopWdth"].to_numpy()
    attr_chslp = attr_df["ChSlp"].to_numpy()
    attr_musx = attr_df["MusX"].to_numpy()

    for i, seg_id in enumerate(order):
        row_idx = attr_lookup.get(int(seg_id))
        if row_idx is not None:
            length_m[i] = attr_length_m[row_idx]
            slope[i] = attr_so[row_idx]
            top_width[i] = attr_topwdth[row_idx]
            side_slope[i] = attr_chslp[row_idx]
            muskingum_x[i] = attr_musx[row_idx]

    # --- flowpaths layer (toid) ---
    fp_query = "SELECT id, toid FROM flowpaths"
    fp_df = pl.read_database(query=fp_query, connection=conn)
    conn.close()

    fp_df = fp_df.with_columns(pl.col("id").str.replace("^wb-", "").cast(pl.Int32).alias("numeric_id"))
    fp_lookup = {row["numeric_id"]: row["toid"] for row in fp_df.iter_rows(named=True)}

    toid_arr = np.array([fp_lookup.get(int(seg_id), "") or "" for seg_id in order], dtype=object)

    # --- Write arrays to zarr ---
    for name, data in [
        ("length_m", length_m),
        ("slope", slope),
        ("top_width", top_width),
        ("side_slope", side_slope),
        ("muskingum_x", muskingum_x),
    ]:
        arr = root.create_array(name=name, shape=data.shape, dtype=data.dtype)
        arr[:] = data

    toid_z = root.create_array(name="toid", shape=toid_arr.shape, dtype="str")
    toid_z[:] = toid_arr

    print(f"Flowpath attributes written to zarr at {out_path}")


def build_lynker_hydrofabric_adjacency(
    fp: pl.LazyFrame,
    network: pl.LazyFrame,
    out_path: Path,
    gpkg_path: Path | None = None,
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
    gpkg_path : Path, optional
        Path to the hydrofabric GeoPackage. When provided, flowpath physical
        attributes are extracted and written into the zarr store alongside
        the adjacency topology.

    Returns
    -------
    None
    """
    matrix, ts_order = create_matrix(fp, network)
    coo_to_zarr(matrix, ts_order, out_path)

    if gpkg_path is not None:
        write_flowpath_attributes(gpkg_path, out_path)


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

    for gauge in tqdm(gauge_set.gauges, desc="Creating Gauge COO matrices", ncols=140, ascii=True):
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
