from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
import zarr
from scipy import sparse
from tqdm import tqdm

from .utils import _build_rustworkx_object, _build_upstream_dict_from_merit


def create_matrix(fp: gpd.GeoDataFrame) -> tuple[sparse.coo_matrix, list[int]]:
    """
    Create a lower triangular adjacency matrix from MERIT flowpaths.

    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths dataframe with 'COMID', 'NextDownID', and 'up1'-'up4' columns.

    Returns
    -------
    tuple[sparse.coo_matrix, list[int]]
        tuple[0]: A scipy sparse matrix in COO format
        tuple[1]: Topological ordering of Flowpaths (as COMID integers)
    """
    # Build upstream connectivity dictionary
    print("Building upstream connectivity dictionary...")
    upstream_dict = _build_upstream_dict_from_merit(fp)

    if not upstream_dict:
        raise ValueError("No upstream connections found in the data")

    print(f"Found {len(upstream_dict)} downstream nodes with upstream connections")

    # Build RustWorkX graph
    print("Building RustWorkX graph...")
    graph, node_indices = _build_rustworkx_object(upstream_dict)

    print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    # Get topological sort
    try:
        ts_order = rx.topological_sort(graph)
    except rx.DAGHasCycle:
        print("\nDAG has cycle detected! Removing all flowpaths in cycles...")

        # Find cycles
        cycles_iter = rx.simple_cycles(graph)
        cycles = list(cycles_iter)
        print(f"Found {len(cycles)} cycle(s)")

        # Collect all COMIDs involved in cycles
        cycle_comids = set()
        for cycle in cycles:
            for node_idx in cycle:
                comid = graph.get_node_data(node_idx)
                cycle_comids.add(comid)

        print(f"Removing {len(cycle_comids)} flowpaths involved in cycles")

        # Remove ALL flowpaths in cycles using polars
        fp_pl = pl.DataFrame(fp.drop(columns="geometry"))
        fp_filtered_pl = fp_pl.filter(~pl.col("COMID").is_in(list(cycle_comids)))

        # Convert back to GeoDataFrame
        fp_filtered = fp[fp["COMID"].isin(fp_filtered_pl["COMID"].to_list())].copy()
        print(f"Dataset reduced from {len(fp)} to {len(fp_filtered)} flowpaths")

        # Recursively call create_matrix with filtered data
        return create_matrix(fp_filtered)

    # Reindex the flowpaths based on the topo order
    id_order = [graph.get_node_data(gidx) for gidx in ts_order]
    idx_map = {id: idx for idx, id in enumerate(id_order)}

    col = []
    row = []

    for node in tqdm(ts_order, desc="Creating sparse matrix indices"):
        if graph.out_degree(node) == 0:  # terminal node
            continue
        id = graph.get_node_data(node)
        # if successors is not size 1, then not dendritic and should be an error...
        assert len(graph.successors(node)) == 1, f"Node {id} has multiple successors, not dendritic"
        id_ds = graph.successors(node)[0]  # This is the successor's node index
        col.append(idx_map[id])
        row.append(idx_map[id_ds])  # Get COMID from node index

    matrix = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(ts_order), len(ts_order)), dtype=np.uint8
    )

    # Ensure matrix is lower triangular
    assert np.all(matrix.row >= matrix.col), "Matrix is not lower triangular"

    return matrix, id_order


def coo_to_zarr(coo: sparse.coo_matrix, ts_order: list[int], out_path: Path) -> None:
    """
    Convert a lower triangular adjacency matrix to a sparse COO matrix and save it in a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list[int]
        Topological sort order of flowpaths (as COMID integers).
    out_path : Path
        Path to save the zarr group.

    Returns
    -------
    None
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
        "indices_0": coo.row.dtype.__str__(),
        "indices_1": coo.col.dtype.__str__(),
        "values": coo.data.dtype.__str__(),
    }

    print(f"MERIT Hydrofabric adjacency written to zarr at {out_path}")
