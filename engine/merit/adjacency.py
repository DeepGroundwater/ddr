from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
import zarr
from scipy import sparse
from tqdm import tqdm


def _build_upstream_dict_from_merit(
    fp: gpd.GeoDataFrame,
) -> dict[int, list[int]]:
    """
    Build upstream connectivity dictionary from MERIT flowpaths.

    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths with COMID, NextDownID, and up1-up4 columns.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping downstream COMID to list of upstream COMIDs.
    """
    # Convert to polars (drop geometry)
    df = pl.DataFrame(fp.drop(columns="geometry"))

    # Build connections from up1, up2, up3, up4
    # Create a long-form dataframe with all upstream connections
    connections = []
    for up_col in ["up1", "up2", "up3", "up4"]:
        conn = df.select(
            [
                pl.col("COMID").cast(pl.Int32).alias("dn_comid"),
                pl.col(up_col).cast(pl.Int32).alias("up_comid"),
            ]
        ).filter(pl.col("up_comid") > 0)
        connections.append(conn)

    if not connections:
        return {}

    # Concatenate all connections
    all_connections = pl.concat(connections)

    # Group by downstream COMID and aggregate upstream COMIDs
    upstream_dict_df = all_connections.group_by("dn_comid").agg(
        pl.col("up_comid").sort().alias("upstream_list")
    )

    # Convert to dictionary
    return dict(
        zip(
            upstream_dict_df["dn_comid"].to_list(),
            upstream_dict_df["upstream_list"].to_list(),
            strict=False,
        )
    )


def _build_rustworkx_object(
    upstream_network: dict[int, list[int]],
) -> tuple[rx.PyDiGraph, dict[int, int]]:
    """
    Build a RustWorkX directed graph from upstream network dictionary.

    Parameters
    ----------
    upstream_network : dict[int, list[int]]
        Dictionary mapping downstream COMID to list of upstream COMIDs.

    Returns
    -------
    tuple[rx.PyDiGraph, dict[int, int]]
        Graph and mapping of COMID to graph node index.
    """
    graph = rx.PyDiGraph(check_cycle=False)
    node_indices: dict[int, int] = {}

    # Add all nodes first
    for to_comid in tqdm(sorted(upstream_network.keys()), desc="Adding nodes"):
        from_comids = upstream_network[to_comid]
        if to_comid not in node_indices:
            node_indices[to_comid] = graph.add_node(to_comid)
        for from_comid in from_comids:
            if from_comid not in node_indices:
                node_indices[from_comid] = graph.add_node(from_comid)

    # Add edges
    for to_comid, from_comids in tqdm(upstream_network.items(), desc="Adding edges"):
        for from_comid in from_comids:
            graph.add_edge(node_indices[from_comid], node_indices[to_comid], None)

    return graph, node_indices


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


if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a lower triangular adjacency matrix from MERIT hydrofabric data."
    )
    parser.add_argument(
        "--pkg",
        type=Path,
        default=Path(
            "/projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
        ),
        help="Path to the MERIT shapefile.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to save the zarr group. Defaults to current working directory with name appended.",
    )
    args = parser.parse_args()

    if args.path is None:
        out_path = Path.cwd() / "data/merit_adjacency.zarr"
    else:
        out_path = args.path

    if out_path.exists():
        print(f"Cannot create zarr store {out_path}. One already exists")
        exit(1)

    print(f"Reading MERIT data from {args.pkg}")
    fp = gpd.read_file(args.pkg)

    print(f"Creating adjacency matrix for {len(fp)} flowpaths")
    matrix, ts_order = create_matrix(fp)

    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    coo_to_zarr(matrix, ts_order, out_path)
