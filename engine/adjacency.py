#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date June 19 2025
@version 1.1

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric and writing a sparse zarr group
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import rustworkx as rx
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

    fp = fp.with_row_index(name="idx").collect()
    network = network.collect().unique(subset=["id"])
    # Create tuples of the index location and the downstream nexus ID
    _values = zip(fp["idx"], fp["toid"])
    # define flowpaths as a dictionary of ids to tuples of (index, downstream nexus id)
    fp = dict(zip(fp["id"], _values, strict=True))
    # define network as a dictionary of nexus ids to downstream flowpath ids
    network = dict(zip(network["id"], network["toid"], strict=True))

    # pre-allocate the graph with the number of flowpaths
    graph = rx.PyDiGraph(check_cycle=False, node_count_hint=len(fp), edge_count_hint=len(fp))
    # in this graph form -- each waterbody/flowpath is a node and each nexus is a directed edge
    # All flowpaths are nodes, add them upfront...
    gidx = graph.add_nodes_from(fp.keys())
    for idx in tqdm(gidx, desc="Building network graph"):
        id = graph.get_node_data(idx)
        nex = fp[id][1] # the downstream nexus id
        terminal = False
        ds_wb = network.get(nex)
        if ds_wb is None:
            # This allows a ghost node to be used by multiple upstream
            # flowpaths which accumulate at the same location
            # If we find a terminal we haven't seen before, we create a new ghost node
            if ghost and not id.startswith("ghost-"):
                ghost_node = graph.add_node(f"ghost-{_tnx_counter}")
                ds_wb = f"ghost-{_tnx_counter}"
                network[nex] = ds_wb
                network[ds_wb] = None
                fp[ds_wb] = (ghost_node, None)
                _tnx_counter += 1
            else:
                terminal = True
        if not terminal:
            graph.add_edge(idx, fp[ds_wb][0], nex)

    ts_order = rx.topological_sort(graph)

    # Reindex the flowpaths based on the topo order
    id_order = [ graph.get_node_data(gidx) for gidx in ts_order ]
    idx_map = {id: idx for idx, id in enumerate(id_order)}

    col = []
    row = []

    for node in tqdm(ts_order, "Creating sparse matrix indicies"):
        if graph.out_degree(node) == 0: # terminal node
            continue
        id = graph.get_node_data(node)
        # if successors is not size 1, then not dendritic and should be an error...
        assert len(graph.successors(node)) == 1, f"Node {id} has multiple successors, not dendritic"
        id_ds = graph.successors(node)[0]
        col.append( idx_map[id] )
        row.append( idx_map[id_ds] )

    matrix = sparse.coo_matrix( (np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(ts_order), len(ts_order)), dtype=np.uint8)
    
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
