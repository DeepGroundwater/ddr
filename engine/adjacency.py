#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date June 19 2025
@version 1.1

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric and writing a sparse zarr group
"""

import graphlib as gl
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import zarr
from polars import LazyFrame
from pyiceberg.catalog import load_catalog
from scipy import sparse
from tqdm import tqdm


def index_matrix(matrix: np.ndarray, fp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a 2D dataframe with rows and columns indexed by flowpath IDs
    and values from the lower triangular adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Lower triangular adjacency matrix.
    fp : gpd.GeoDataFrame
        Flowpaths dataframe with 'toid' column indicating downstream nexus IDs.

    Returns
    -------
    gpd.GeoDataFrame
        matrix dataframe with flowpath IDs as index and columns
    """
    # Create a new GeoDataFrame with the same index as the flowpaths
    matrix_df = gpd.GeoDataFrame(
        index=fp.index, columns=fp.index, data=np.zeros((len(fp), len(fp)), dtype=int)
    )
    matrix_df.rename_axis("to", inplace=True)
    matrix_df.rename_axis("from", axis=1, inplace=True)
    # Fill the dataframe with the values from the matrix
    for i in range(len(fp)):
        for j in range(len(fp)):
            matrix_df.iloc[i, j] = matrix[i, j]

    return matrix_df


def create_matrix(fp: LazyFrame, network: LazyFrame, ghost=False) -> tuple[sparse.coo_matrix, list[str]]:
    """
    Create a lower triangular adjacency matrix from flowpaths and network dataframes.

    Parameters
    ----------
    fp : LazyFrame
        Flowpaths dataframe with 'toid' column indicating downstream nexus IDs.
    network : LazyFrame
        Network dataframe with 'toid' column indicating downstream flowpath IDs.

    Returns
    -------
    np.ndarray
        Lower triangular adjacency matrix.
    """
    _tnx_counter = 0

    # Toposort for the win
    sorter = gl.TopologicalSorter()
    fp_df = fp.select([pl.col("id"), pl.col("toid")]).collect()
    network_df = network.collect()

    # Pre-collect network data to avoid repeated filtering
    network_dict = dict(zip(network_df["id"].to_list(), network_df["toid"].to_list(), strict=True))

    ghost_nodes_to_add = []
    network_updates = {}

    for row in tqdm(fp_df.iter_rows(named=True), desc="finding indices", total=len(fp_df)):
        id_val = row["id"]
        nex = row["toid"]

        # Fast lookup instead of filtering each time
        ds_wb = network_dict.get(nex)

        if ds_wb is None:
            print("Terminal nex???", nex)
            ds_wb = np.nan

        if pd.isna(ds_wb):
            if ghost:
                ds_wb = f"ghost-{_tnx_counter}"
                # Track changes to apply later
                network_updates[nex] = ds_wb  # Point nexus to ghost
                ghost_nodes_to_add.append(
                    {
                        "id": ds_wb,
                        "toid": None,  # Ghost points to nothing
                    }
                )
                network_dict[nex] = ds_wb
                network_dict[ds_wb] = None
                _tnx_counter += 1

        # Add a node to the sorter, ds_wb is the node, id_val is its predecessor
        sorter.add(ds_wb, id_val)

    # Apply network updates efficiently in Polars
    if network_updates or ghost_nodes_to_add:
        # Update existing network entries
        if network_updates:
            network_df = network_df.with_columns(
                [
                    pl.when(pl.col("id").is_in(list(network_updates.keys())))
                    .then(pl.col("id").map_elements(lambda x: network_updates.get(x), return_dtype=pl.String))
                    .otherwise(pl.col("toid"))
                    .alias("toid")
                ]
            )

        # Add ghost nodes to network
        if ghost_nodes_to_add:
            ghost_network_df = pl.DataFrame(ghost_nodes_to_add, schema={"id": pl.String, "toid": pl.String})
            network_df = pl.concat([network_df, ghost_network_df])

        # Add ghost nodes to flowpaths
        if ghost_nodes_to_add:
            ghost_fp_df = pl.DataFrame(ghost_nodes_to_add, schema={"id": pl.String, "toid": pl.String})
            fp_df = pl.concat([fp_df, ghost_fp_df])

    # Get topological sort order
    if ghost:
        ts_order = list(sorter.static_order())
    else:
        ts_order = list(filter(lambda s: not pd.isna(s), sorter.static_order()))

    # Create dictionaries for matrix building
    fp_dict = dict(zip(fp_df["id"].to_list(), fp_df["toid"].to_list(), strict=True))
    network_dict = dict(zip(network_df["id"].to_list(), network_df["toid"].to_list(), strict=True))
    id_to_pos = {id_val: pos for pos, id_val in enumerate(ts_order)}

    # Build sparse matrix
    row_idx = []
    col_idx = []

    for wb in tqdm(ts_order, desc="ordering matrix"):
        nex = fp_dict.get(wb)
        if nex is None or pd.isna(nex):
            continue
        ds_wb = network_dict.get(nex)
        if ds_wb is None or pd.isna(ds_wb):
            continue
        if ds_wb == "wb-0":  # Skip this special case
            continue

        idx = id_to_pos.get(wb)
        idxx = id_to_pos.get(ds_wb)

        if idx is None or idxx is None:
            continue

        col_idx.append(idx)
        row_idx.append(idxx)

    coo = sparse.coo_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)), shape=(len(ts_order), len(ts_order)), dtype=np.int8
    )

    # Ensure matrix is lower triangular
    assert np.all(coo.row >= coo.col), "Matrix is not lower triangular"
    _tnx_counter = 0
    return coo, ts_order


def coo_to_zarr(coo: sparse.coo_matrix, ts_order: list[str], out_path: Path) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[str]
        Topological sort order of flowpaths.
    name : str
        Name of the zarr group to create.
    out_path : Path | str | None, optional
        Path to save the zarr group. If None, defaults to current working directory with name appended.

    Returns
    -------
    None
    """
    # Converting to a sparse COO matrix, and saving the output in many arrays within a zarr v3 group
    store = zarr.storage.LocalStore(root=out_path)
    root = zarr.create_group(store=store)

    zarr_order = np.array([int(_id.split("-")[1]) for _id in ts_order], dtype=np.int32)

    indices_0 = root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order = root.create_array(name="order", shape=zarr_order.shape, dtype=zarr_order.dtype)
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
    print(f"CONUS Hydrofabric adjacency written to zarr at {out_path}")


if __name__ == "__main__":
    import argparse

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
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to save the zarr group. Defaults to current working directory with name appended.",
    )
    args = parser.parse_args()

    if args.path is None:
        out_path = Path.cwd() / "conus_adjacency.zarr"
    else:
        out_path = Path(args.path)
    if out_path.exists():
        raise FileExistsError("Cannot create zarr store. One already exists")

    namespace = "hydrofabric"
    catalog = load_catalog(namespace)
    fp = catalog.load_table("hydrofabric.flowpaths").to_polars()
    network = catalog.load_table("hydrofabric.network").to_polars()
    coo, ts_order = create_matrix(fp, network)
    coo_to_zarr(coo, ts_order, out_path)

    # Visual verification
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(fp)
    # print(matrix)
