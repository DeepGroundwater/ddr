"""IO utilities for reading and writing adjacency matrices to zarr"""

from pathlib import Path

import numpy as np
import rustworkx as rx
import zarr
from scipy import sparse


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
    store = zarr.storage.LocalStore(root=out_path)
    root = zarr.create_group(store=store)

    _order = np.array(ts_order, dtype=np.int32)

    indices_0 = root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order = root.create_array(name="order", shape=_order.shape, dtype=_order.dtype)

    indices_0[:] = coo.row
    indices_1[:] = coo.col
    values[:] = coo.data
    order[:] = _order

    root.attrs["format"] = "COO"
    root.attrs["shape"] = list(coo.shape)
    root.attrs["data_types"] = {
        "indices_0": str(coo.row.dtype),
        "indices_1": str(coo.col.dtype),
        "values": str(coo.data.dtype),
    }

    print(f"Adjacency matrix written to zarr at {out_path}")


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
    zarr_order = np.array(ts_order, dtype=np.int32)

    indices_0 = gauge_root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = gauge_root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = gauge_root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order_array = gauge_root.create_array(name="order", shape=zarr_order.shape, dtype=zarr_order.dtype)

    indices_0[:] = coo.row
    indices_1[:] = coo.col
    values[:] = coo.data
    order_array[:] = zarr_order

    gauge_root.attrs["format"] = "COO"
    gauge_root.attrs["shape"] = list(coo.shape)
    gauge_root.attrs["gage_catchment"] = int(origin_comid)
    gauge_root.attrs["gage_idx"] = int(merit_mapping[origin_comid])
    gauge_root.attrs["data_types"] = {
        "indices_0": str(coo.row.dtype),
        "indices_1": str(coo.col.dtype),
        "values": str(coo.data.dtype),
    }


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
        The COO matrix and topological order.
    """
    root = zarr.open_group(store=zarr_path, mode="r")

    row = root["indices_0"][:]
    col = root["indices_1"][:]
    data = root["values"][:]
    shape = tuple(root.attrs["shape"])
    ts_order = root["order"][:].tolist()

    coo = sparse.coo_matrix((data, (row, col)), shape=shape)

    return coo, ts_order


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
