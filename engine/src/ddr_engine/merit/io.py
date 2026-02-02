"""IO utilities for reading and writing MERIT adjacency matrices to zarr.

This module provides MERIT-specific wrappers around the core COO zarr I/O
functions. MERIT uses integer COMIDs as identifiers.

For most use cases, prefer the auto-detecting functions from ddr_engine:

    >>> from ddr_engine import coo_to_zarr, coo_from_zarr
    >>> coo_to_zarr(coo, ts_order, path, "merit")
    >>> coo, ts_order = coo_from_zarr(path)  # Auto-detects
"""

from pathlib import Path

import numpy as np
import rustworkx as rx
import zarr
from scipy import sparse

from ddr_engine.core import (
    coo_from_zarr_generic,
    coo_to_zarr_generic,
    coo_to_zarr_group_generic,
    merit_converter,
)

GEODATASET = "merit"


def coo_to_zarr(
    coo: sparse.coo_matrix,
    ts_order: list[int],
    out_path: Path,
) -> None:
    """
    Save a COO adjacency matrix to a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[int]
        Topological sort order of flowpaths (as COMID integers).
    out_path : Path
        Path to save the zarr group.
    """
    coo_to_zarr_generic(coo, ts_order, out_path, merit_converter, geodataset=GEODATASET)


def coo_to_zarr_group(
    coo: sparse.coo_matrix,
    ts_order: list[int],
    origin_comid: int,
    gauge_root: zarr.Group,
    merit_mapping: dict[int, int],
) -> None:
    """
    Save a COO matrix to a zarr group for a gauge subset.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix
    ts_order : list[int]
        COMIDs in topological sort order
    origin_comid : int
        The origin COMID of the gauge
    gauge_root : zarr.Group
        The zarr group for the subset COO matrix
    merit_mapping : dict[int, int]
        Mapping of COMID to its position in the array
    """
    coo_to_zarr_group_generic(
        coo, ts_order, origin_comid, gauge_root, merit_mapping, merit_converter, geodataset=GEODATASET
    )


def coo_from_zarr(zarr_path: Path) -> tuple[sparse.coo_matrix, list[int]]:
    """
    Load a COO adjacency matrix from a zarr group.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr group.

    Returns
    -------
    tuple[sparse.coo_matrix, list[int]]
        The COO matrix and topological order (as COMID integers).
    """
    return coo_from_zarr_generic(zarr_path, merit_converter)


def create_subset_coo(
    subset_comids: list[int],
    merit_mapping: dict[int, int],
    graph: rx.PyDiGraph,
    node_indices: dict[int, int],
) -> tuple[sparse.coo_matrix, list[int]]:
    """
    Create a COO matrix for a subset indexed from the MERIT adjacency matrix.

    Parameters
    ----------
    subset_comids : list[int]
        List of COMIDs in the subset
    merit_mapping : dict[int, int]
        Mapping of COMID to its index in the MERIT adjacency matrix
    graph : rx.PyDiGraph
        The river network graph
    node_indices : dict[int, int]
        Mapping of COMID to graph node index

    Returns
    -------
    tuple[sparse.coo_matrix, list[int]]
        The sparse COO matrix and list of COMIDs in the subset
    """
    subset_set = set(subset_comids)
    row_idx = []
    col_idx = []

    for comid in subset_comids:
        if comid not in node_indices:
            continue

        node_idx = node_indices[comid]
        successors = graph.successors(node_idx)

        for ds_comid in successors:
            if ds_comid in subset_set:
                row_idx.append(merit_mapping[ds_comid])
                col_idx.append(merit_mapping[comid])

    if len(row_idx) == 0:
        return sparse.coo_matrix((len(merit_mapping), len(merit_mapping)), dtype=np.uint8), subset_comids

    coo = sparse.coo_matrix(
        (np.ones(len(row_idx), dtype=np.uint8), (row_idx, col_idx)),
        shape=(len(merit_mapping), len(merit_mapping)),
        dtype=np.uint8,
    )

    assert np.all(coo.row >= coo.col), "Matrix is not lower triangular"
    return coo, subset_comids
