#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date June 12 2025
@version 1.0

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric and writing a sparse zarr group
"""

import graphlib as gl
from pathlib import Path

import geopandas as gpd
import numpy as np
import zarr
import polars as pl
from polars import LazyFrame
from pyiceberg.catalog import load_catalog
from pyiceberg.table import Table
from pyiceberg.expressions import And, EqualTo, In
from scipy import sparse
from tqdm import tqdm


def index_matrix(matrix: np.ndarray, fp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a 2D dataframe with ros and columns indexed by flowpath IDs
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


_tnx_counter = 0


def create_matrix(
    fp: LazyFrame, network: LazyFrame, ghost=False
) -> tuple[np.ndarray, list[str]]:
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
    global _tnx_counter

    # Toposort for the win
    sorter = gl.TopologicalSorter()
    fp_rows = fp.select([pl.col("id"), pl.col("toid")]).collect()

    # Pre-collect network data to avoid repeated filtering
    network_df = network.collect()
    network_lookup = dict(zip(network_df["id"].to_list(), network_df["toid"].to_list()))

    network_changes = []
    fp_changes = []

    for row in tqdm(fp_rows.iter_rows(), desc="finding indices"):
        id_val = row[0]
        nex = row[1]
        
        # Fast lookup instead of filtering each time
        ds_wb = network_lookup.get(nex)
        
        if ds_wb is None:
            print("Terminal nex???", nex)
            ds_wb = np.nan
        
        if isinstance(ds_wb, float) and np.isnan(ds_wb):
            if ghost:
                ds_wb = f"ghost-{_tnx_counter}"
                network_changes.extend([
                    (nex, ds_wb),      # network.loc[nex, "toid"] = ds_wb
                    (ds_wb, None),     # network.loc[ds_wb, "toid"] = np.nan
                ])
                fp_changes.append((ds_wb, None))  # fp.loc[ds_wb, "toid"] = np.nan
                _tnx_counter += 1
        
        # Add a node to the sorter, ds_wb is the node, id_val is its predecessor
        sorter.add(ds_wb, id_val)

    # Apply all changes efficiently after the loop and overwrite original variables
    if network_changes:
        # Create lookup for changes
        changes_dict = {}
        for id_to_change, new_toid in network_changes:
            changes_dict[id_to_change] = new_toid
        
        # Apply all network changes in one operation and overwrite
        network = network_df.with_columns([
            pl.col("id").map_elements(
                lambda x: changes_dict.get(x, network_lookup.get(x)),
                return_dtype=pl.String
            ).alias("toid")
        ])
    else:
        network = network_df

    if fp_changes:
        # Apply fp changes and overwrite
        fp_changes_dict = {id_to_change: new_toid for id_to_change, new_toid in fp_changes}
        fp = fp.with_columns([
            pl.col("id").map_elements(
                lambda x: fp_changes_dict.get(x, x),  # You'll need to adjust this based on your fp structure
                return_dtype=pl.String
            ).alias("toid")
        ]).collect()
    else:
        fp = fp.collect()

    # There are possibly more than one correct topological sort orders
    # Just grab one and go...
    if ghost:
        ts_order = list(sorter.static_order())
    else:
        ts_order = list(filter(lambda s: not (isinstance(s, float) and np.isnan(s)), sorter.static_order()))

    # Reindex the flowpaths based on the topo order
    fp = fp.reindex(ts_order)

    # Create matrix, "indexed" the same as the re-ordered fp dataframe
    matrix = np.zeros((len(fp), len(fp)))

    for wb in tqdm(ts_order, desc="Creating toposort ordering"):
        nex = fp.loc[wb]["toid"]
        if isinstance(nex, float) and np.isnan(nex):
            continue
        # Use the network to find wb -> wb topology
        try:
            ds_wb = network.loc[nex]["toid"]
            if isinstance(ds_wb, gpd.pd.Series):
                ds_wb = ds_wb.iloc[0]
        except KeyError:
            print("Terminal nex???", nex)
            continue
        if isinstance(ds_wb, float) and np.isnan(ds_wb):
            continue
        idx = fp.index.get_loc(wb)
        idxx = fp.index.get_loc(ds_wb)
        fp["matrix_idxx"] = idxx
        fp["matrix_idx"] = idx
        # print(wb, " -> ", nex, " -> ", ds_wb)
        # Set the matrix value
        matrix[idxx][idx] = 1

    # Ensure, within tolerance, that this is a lower triangular matrix
    assert np.allclose(matrix, np.tril(matrix))
    _tnx_counter = 0
    return matrix, ts_order


def matrix_to_zarr(
    matrix: np.ndarray, ts_order: list[str], name: str, out_path: Path | str | None = None
) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    matrix : np.ndarray
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
    out_path = Path(out_path) if out_path is not None else Path.cwd() / f"{name}_adjacency.zarr"
    store = zarr.storage.LocalStore(root=out_path)
    if out_path.exists():
        root = zarr.open_group(store=store)
    else:
        root = zarr.create_group(store=store)

    coo = sparse.coo_matrix(matrix)
    zarr_order = np.array([int(_id.split("-")[1]) for _id in ts_order], dtype=np.int32)

    gauge_root = root.create_group(name=name)
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
    gauge_root.attrs["data_types"] = {
        "indices_0": coo.row.dtype.__str__(),
        "indices_1": coo.col.dtype.__str__(),
        "values": coo.data.dtype.__str__(),
    }
    print(f"{name} written to zarr at {out_path}")


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
        "name",
        type=str,
        help="Name of the matrix for saving in zarr group.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to save the zarr group. Defaults to current working directory with name appended.",
    )
    args = parser.parse_args()

    # Useful for some debugging, not needed for algorithm
    # nexi = gpd.read_file(pkg, layer='nexus').set_index('id')
    # fp = gpd.read_file(args.pkg, layer="flowpaths").set_index("id")
    # network = gpd.read_file(args.pkg, layer="network").set_index("id")
    namespace = "hydrofabric"
    catalog = load_catalog(namespace)
    fp = catalog.load_table("hydrofabric.flowpaths").to_polars()
    network = catalog.load_table("hydrofabric.network").to_polars()
    matrix, ts_order = create_matrix(fp, network)
    matrix_to_zarr(matrix, ts_order, args.name, args.path)

    # Visual verification
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(fp)
    # print(matrix)
