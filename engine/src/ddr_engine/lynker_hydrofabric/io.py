"""IO utilities for reading and writing Lynker Hydrofabric adjacency matrices to zarr.

This module provides Lynker-specific wrappers around the core COO zarr I/O
functions. Lynker uses wb-* string IDs as identifiers.

For most use cases, prefer the auto-detecting functions from ddr_engine:

    >>> from ddr_engine import coo_to_zarr, coo_from_zarr
    >>> coo_to_zarr(coo, ts_order, path, "lynker")
    >>> coo, ts_order = coo_from_zarr(path)  # Auto-detects
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import rustworkx as rx
import zarr
from scipy import sparse
from tqdm import tqdm

from ddr_engine.core import (
    coo_from_zarr_generic,
    coo_to_zarr_generic,
    coo_to_zarr_group_generic,
    lynker_converter,
)

GEODATASET = "lynker"


def index_matrix(matrix: np.ndarray, fp: pd.DataFrame) -> pd.DataFrame:
    """Create a 2D dataframe with rows and columns indexed by flowpath IDs and values from the lower triangular adjacency matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Lower triangular adjacency matrix.
    fp : pd.DataFrame
        Flowpaths dataframe with 'toid' column indicating downstream nexus IDs.

    Returns
    -------
    pd.DataFrame
        matrix dataframe with flowpath IDs as index and columns
    """
    # Create a new DataFrame with the same index as the flowpaths
    matrix_df = pd.DataFrame(index=fp.index, columns=fp.index, data=np.zeros((len(fp), len(fp)), dtype=int))
    matrix_df.rename_axis("to", inplace=True)
    matrix_df.rename_axis("from", axis=1, inplace=True)
    # Fill the dataframe with the values from the matrix
    for i in range(len(fp)):
        for j in range(len(fp)):
            matrix_df.iloc[i, j] = matrix[i, j]

    return matrix_df


def create_matrix(
    fp: pl.LazyFrame, network: pl.LazyFrame, ghost: bool = False
) -> tuple[sparse.coo_matrix, list[str]]:
    """
    Create a lower triangular adjacency matrix from flowpaths and network dataframes.

    @author Nels Frazier

    Parameters
    ----------
    fp : pl.LazyFrame
        Flowpaths dataframe with 'toid' column indicating downstream nexus IDs.
    network : pl.LazyFrame
        Network dataframe with 'toid' column indicating downstream flowpath IDs.

    Returns
    -------
    tuple[sparse.coo_matrix, list[str]]
        tuple[0]: A scipy sparse matrix in COO format
        tuple[1]: Topological ordering of Flowpaths
    """
    _tnx_counter = 0

    fp = fp.with_row_index(name="idx").collect()
    network = network.collect().unique(subset=["id"])
    # Create tuples of the index location and the downstream nexus ID
    _values = zip(fp["idx"], fp["toid"], strict=False)
    # define flowpaths as a dictionary of ids to tuples of (index, downstream nexus id)
    fp = dict(zip(fp["id"], _values, strict=True))
    # define network as a dictionary of nexus ids to downstream flowpath ids
    network = dict(zip(network["id"], network["toid"], strict=True))

    # pre-allocate the graph with the number of flowpaths
    graph = rx.PyDiGraph(check_cycle=False, node_count_hint=len(fp), edge_count_hint=len(fp))
    # in this graph form -- each waterbody/flowpath is a node and each nexus is a directed edge
    # All flowpaths are nodes, add them upfront...
    gidx = graph.add_nodes_from(fp.keys())
    for idx in tqdm(gidx, desc="Building network graph", ncols=140, ascii=True):
        id = graph.get_node_data(idx)
        nex = fp[id][1]  # the downstream nexus id
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
    id_order = [graph.get_node_data(gidx) for gidx in ts_order]
    idx_map = {id: idx for idx, id in enumerate(id_order)}

    col = []
    row = []

    for node in tqdm(ts_order, "Creating sparse matrix indicies", ncols=140, ascii=True):
        if graph.out_degree(node) == 0:  # terminal node
            continue
        id = graph.get_node_data(node)
        # if successors is not size 1, then not dendritic and should be an error...
        assert len(graph.successors(node)) == 1, f"Node {id} has multiple successors, not dendritic"
        id_ds = graph.successors(node)[0]
        col.append(idx_map[id])
        row.append(idx_map[id_ds])

    matrix = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(ts_order), len(ts_order)), dtype=np.uint8
    )

    # Ensure matrix is lower triangular
    assert np.all(matrix.row >= matrix.col), "Matrix is not lower triangular"
    # assert sparse.linalg.is_sptriangular(matrix)[0] == True

    # If we want to get updated flowpath and network dataframes,
    # we can create them from the topological sort order
    # fp = pl.DataFrame({"id": [graph.get_node_data(gidx) for gidx in ts_order], "toid": [fp.get(id) for id in ts_order]})
    # technically, the network dataframe doesn't need to worry about sort order...just recreate
    # from the possibly updated network dictionary
    # network = pl.DataFrame( {"id": network.keys(), "toid": network.values()})
    # With polars frames, these would need to be returned as new objects
    # from this function.

    return matrix, id_order


def coo_to_zarr(coo: sparse.coo_matrix, ts_order: list[str], out_path: Path) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[str]
        Topological sort order of flowpaths (as wb-* strings).
    out_path : Path
        Path to save the zarr group.
    """
    coo_to_zarr_generic(coo, ts_order, out_path, lynker_converter, geodataset=GEODATASET)


def coo_to_zarr_group(
    coo: sparse.coo_matrix,
    ts_order: list[str],
    origin: str,
    gauge_root: zarr.Group,
    conus_mapping: dict[str, int],
) -> None:
    """
    Save a COO matrix to a zarr group for a gauge subset.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[str]
        Watershed boundary IDs in deterministic order (sorted by CONUS index).
    origin : str
        The origin watershed boundary ID of the gauge.
    gauge_root : zarr.Group
        The zarr group for the subset COO matrix.
    conus_mapping : dict[str, int]
        Mapping of watershed boundary ID to its position in the CONUS array.
    """
    coo_to_zarr_group_generic(
        coo, ts_order, origin, gauge_root, conus_mapping, lynker_converter, geodataset=GEODATASET
    )


def coo_from_zarr(zarr_path: Path) -> tuple[sparse.coo_matrix, list[str]]:
    """
    Load a COO adjacency matrix from a zarr group.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr group.

    Returns
    -------
    tuple[sparse.coo_matrix, list[str]]
        The COO matrix and topological order (as wb-* strings).
    """
    return coo_from_zarr_generic(zarr_path, lynker_converter)


def create_coo(
    connections: list[tuple[str, str]],
    conus_mapping: dict[str, int],
) -> tuple[sparse.coo_matrix, list[str]]:
    """
    Create a COO matrix from connections indexed by CONUS adjacency matrix indices.

    Parameters
    ----------
    connections : list[tuple[str, str]]
        List of (downstream_wb, upstream_wb) connection tuples.
    conus_mapping : dict[str, int]
        Mapping of watershed boundary IDs to their CONUS index (topologically sorted).

    Returns
    -------
    tuple[sparse.coo_matrix, list[str]]
        The sparse COO matrix and deterministically ordered list of flowpath IDs.

    Notes
    -----
    The flowpath list is sorted by CONUS mapping index for deterministic ordering.
    """
    row_idx = []
    col_idx = []
    for flowpaths in connections:
        try:
            row_idx.append(conus_mapping[flowpaths[0]])
        except KeyError:
            flowpath_id = f"wb-{int(float(flowpaths[0].split('-')[1]))}"
            row_idx.append(conus_mapping[flowpath_id])
        try:
            col_idx.append(conus_mapping[flowpaths[1]])
        except KeyError:
            flowpath_id = f"wb-{int(float(flowpaths[1].split('-')[1]))}"
            col_idx.append(conus_mapping[flowpath_id])
    coo = sparse.coo_matrix(
        (np.ones(len(row_idx)), (row_idx, col_idx)),
        shape=(len(conus_mapping), len(conus_mapping)),
        dtype=np.int8,
    )

    # Collect unique flowpaths and sort by CONUS index for determinism
    all_flowpaths_set = {item for connection in connections for item in connection}
    all_flowpaths = sorted(all_flowpaths_set, key=lambda x: conus_mapping.get(x, float("inf")))

    assert np.all(coo.row >= coo.col), "Matrix is not lower triangular"
    return coo, all_flowpaths
