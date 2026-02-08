"""Build functions for adjacency matrices from MERIT flowpaths"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
import zarr
from scipy import sparse
from tqdm import tqdm

from ddr.geodatazoo.dataclasses import GaugeSet

from .graph import build_graph, build_upstream_dict, subset_upstream
from .io import coo_to_zarr, coo_to_zarr_group, create_subset_coo


def create_adjacency_matrix(
    fp: gpd.GeoDataFrame,
) -> tuple[sparse.coo_matrix, list[int]]:
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
    print("Building upstream connectivity dictionary...")
    upstream_dict = build_upstream_dict(fp)

    if not upstream_dict:
        raise ValueError("No upstream connections found in the data")

    print(f"Found {len(upstream_dict)} downstream nodes with upstream connections")

    print("Building RustWorkX graph...")
    graph, node_indices = build_graph(upstream_dict)

    print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    try:
        ts_order = rx.topological_sort(graph)
    except rx.DAGHasCycle:
        print("\nDAG has cycle detected! Removing all flowpaths in cycles...")

        cycles_iter = rx.simple_cycles(graph)
        cycles = list(cycles_iter)
        print(f"Found {len(cycles)} cycle(s)")

        cycle_comids = set()
        for cycle in cycles:
            for node_idx in cycle:
                comid = graph.get_node_data(node_idx)
                cycle_comids.add(comid)

        print(f"Removing {len(cycle_comids)} flowpaths involved in cycles")

        fp_pl = pl.DataFrame(fp.drop(columns="geometry"))
        fp_filtered_pl = fp_pl.filter(~pl.col("COMID").is_in(list(cycle_comids)))

        fp_filtered = fp[fp["COMID"].isin(fp_filtered_pl["COMID"].to_list())].copy()
        print(f"Dataset reduced from {len(fp)} to {len(fp_filtered)} flowpaths")

        return create_adjacency_matrix(fp_filtered)

    id_order = [graph.get_node_data(gidx) for gidx in ts_order]

    # Include isolated COMIDs (single-reach basins with no connections in the data)
    all_comids = {int(c) for c in fp["COMID"].values}
    connected_comids = set(id_order)
    isolated_comids = sorted(all_comids - connected_comids)
    if isolated_comids:
        print(f"Adding {len(isolated_comids)} isolated COMIDs (no upstream/downstream connections)")
    id_order = id_order + isolated_comids

    idx_map = {id: idx for idx, id in enumerate(id_order)}

    col = []
    row = []

    for node in tqdm(ts_order, desc="Creating sparse matrix indices"):
        if graph.out_degree(node) == 0:
            continue
        id = graph.get_node_data(node)
        assert len(graph.successors(node)) == 1, f"Node {id} has multiple successors, not dendritic"
        id_ds = graph.successors(node)[0]
        col.append(idx_map[id])
        row.append(idx_map[id_ds])

    matrix = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.uint8), (row, col)),
        shape=(len(id_order), len(id_order)),
        dtype=np.uint8,
    )

    assert np.all(matrix.row >= matrix.col), "Matrix is not lower triangular"

    return matrix, id_order


def build_merit_adjacency(
    fp: gpd.GeoDataFrame,
    out_path: Path,
) -> Path:
    """
    Build adjacency matrix from MERIT-compatible flowpaths and save to zarr.

    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths with COMID, NextDownID, up1-up4 columns.
    out_path : Path
        Path to save the zarr group.

    Returns
    -------
    Path
        Path to the created zarr store.

    Raises
    ------
    FileExistsError
        If zarr store already exists at out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        raise FileExistsError(f"Cannot create zarr store {out_path}. One already exists")

    print(f"Creating adjacency matrix for {len(fp)} flowpaths")
    matrix, ts_order = create_adjacency_matrix(fp)

    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    coo_to_zarr(matrix, ts_order, out_path)

    return out_path


def build_gauge_adjacencies(
    fp: gpd.GeoDataFrame,
    merit_zarr_path: Path,
    gauge_set: GaugeSet,
    out_path: Path,
) -> Path:
    """
    Build per-gauge adjacency matrices from MERIT flowpaths.

    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths with COMID, NextDownID, up1-up4 columns.
    merit_zarr_path : Path
        Path to the MERIT adjacency zarr store.
    gauge_set : GaugeSet
        Validated gauge set with STAID and COMID attributes.
    out_path : Path
        Path to save the gauge zarr group.

    Returns
    -------
    Path
        Path to the created zarr store.

    Raises
    ------
    FileExistsError
        If zarr store already exists at out_path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        raise FileExistsError(f"Cannot create zarr store {out_path}. One already exists")

    print("Building upstream connectivity dictionary...")
    upstream_dict = build_upstream_dict(fp)

    print("Building RustWorkX graph...")
    graph, node_indices = build_graph(upstream_dict)
    print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    print("Reading MERIT zarr store...")
    merit_root = zarr.open_group(store=merit_zarr_path)
    ts_order = merit_root["order"][:]
    merit_mapping = {comid: idx for idx, comid in enumerate(ts_order)}

    store = zarr.storage.LocalStore(root=out_path)
    root = zarr.create_group(store=store)

    for gauge in tqdm(gauge_set.gauges, desc="Creating Gauge COO matrices"):
        staid = gauge.STAID
        origin_comid = gauge.COMID

        try:
            gauge_root = root.create_group(staid)
        except zarr.errors.ContainsGroupError:
            print(f"Zarr Group exists for: {staid}. Skipping write")
            continue

        if origin_comid not in merit_mapping:
            print(f"COMID {origin_comid} for gauge {staid} not found in MERIT adjacency matrix. Skipping.")
            root.__delitem__(staid)
            continue

        subset_comids = subset_upstream(origin_comid, graph, node_indices)

        # if len(subset_comids) == 1:
        #     print(
        #         f"Gauge {str(staid).zfill(8)} (COMID {origin_comid}) is a headwater catchment (single reach)"
        #     )

        coo, subset_list = create_subset_coo(subset_comids, merit_mapping, graph, node_indices)

        coo_to_zarr_group(
            coo=coo,
            ts_order=subset_list,
            origin_comid=origin_comid,
            gauge_root=gauge_root,
            merit_mapping=merit_mapping,
        )

    print(f"MERIT Gauge adjacency matrices written to {out_path}")
    return out_path
