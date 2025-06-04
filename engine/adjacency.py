#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date Febuary 8, 2025
@version 0.2

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric and writing a sparse zarr group
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import graphlib as gl

from scipy import sparse
from typing import Tuple
import zarr

def create_matrix(fp: gpd.GeoDataFrame, network: gpd.GeoDataFrame) -> Tuple[np.ndarray, list[str]]:
    """
    Create a lower triangular adjacency matrix from flowpaths and network dataframes.
    
    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths dataframe with 'toid' column indicating downstream nexus IDs.
    network : gpd.GeoDataFrame
        Network dataframe with 'toid' column indicating downstream flowpath IDs.
    
    Returns
    -------
    np.ndarray
        Lower triangular adjacency matrix.
    """
    
    # Toposort for the win
    sorter = gl.TopologicalSorter()

    for id in fp.index:
        nex = fp.loc[id]['toid']
        try:
            ds_wb = network.loc[nex]['toid']
        except:
            print("Terminal nex???", nex)
            continue
        if isinstance(ds_wb, gpd.pd.Series):
            ds_wb = ds_wb.iloc[0]
        # Add a node to the sorter, ds_wb is the node, id is its predesessor
        sorter.add(ds_wb, id)

    # There are possibly more than one correct topological sort orders
    # Just grab one and go...
    ts_order = list(sorter.static_order())

    # Reindex the flowpaths based on the topo order
    fp = fp.reindex(ts_order)

    # Create matrix, "indexed" the same as the re-ordered fp dataframe
    matrix = np.zeros( (len(fp), len(fp) ) )

    for wb in ts_order:
        nex = fp.loc[wb]['toid']
        if isinstance(nex, float) and np.isnan(nex):
            continue
        # Use the network to find wb -> wb topology
        try:
            ds_wb = network.loc[nex]['toid']
            if isinstance(ds_wb, gpd.pd.Series):
                ds_wb = ds_wb.iloc[0]
        except KeyError:
            print("Terminal nex???", nex)
            continue
        # Find the inicies of the adajcent flowpaths
        idx = fp.index.get_loc(wb)
        idxx = fp.index.get_loc(ds_wb)
        fp['matrix_idxx'] = idxx
        fp['matrix_idx'] = idx
        # print(wb, " -> ", nex, " -> ", ds_wb)
        # Set the matrix value
        matrix[idxx][idx] = 1

    # Ensure, within tolerance, that this is a lower triangular matrix
    assert np.allclose(matrix, np.tril(matrix))
    return matrix, ts_order

def matrix_to_zarr(matrix: np.ndarray, ts_order: list[str], name: str, out_path: Path | str | None = None) -> None:
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
    indices_0 = gauge_root.create_array(
        name='indices_0', shape=coo.row.shape, dtype=coo.row.dtype
    )
    indices_1 = gauge_root.create_array(
        name='indices_1', shape=coo.col.shape, dtype=coo.row.dtype
    )
    values = gauge_root.create_array(
        name='values', shape=coo.data.shape, dtype=coo.data.dtype
    )
    order = gauge_root.create_array(
        name='order', shape=zarr_order.shape, dtype=zarr_order.dtype
    )
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
    fp = gpd.read_file(args.pkg, layer='flowpaths').set_index('id')
    network = gpd.read_file(args.pkg, layer='network').set_index('id')
    matrix, ts_order = create_matrix(fp, network)
    matrix_to_zarr(matrix, ts_order, args.name, args.path)

    # Visual verification
    # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    # print(fp)
    # print(matrix)
