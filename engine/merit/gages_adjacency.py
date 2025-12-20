#!/usr/bin/env python

"""
@author Tadd Bindas

@date December 20 2024
@version 0.1

A script to build subset COO matrices from the merit_adjacency.zarr
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import polars as pl
import rustworkx as rx
import zarr
from scipy import sparse
from tqdm import tqdm
from utils import _build_rustworkx_object, _build_upstream_dict_from_merit


def subset(origin_comid: int, graph: rx.PyDiGraph, node_indices: dict[int, int]) -> list[int]:
    """
    Find all upstream COMIDs from the origin using graph ancestors.

    Parameters
    ----------
    origin_comid : int
        The COMID to start from
    graph : rx.PyDiGraph
        The river network graph
    node_indices : dict[int, int]
        Mapping of COMID to graph node index

    Returns
    -------
    list[int]
        List of all COMIDs in the subset (including origin)
    """
    if origin_comid not in node_indices:
        return [origin_comid]  # Just the origin if no upstream

    origin_node_idx = node_indices[origin_comid]

    # Get all ancestors (upstream nodes)
    ancestor_indices = rx.ancestors(graph, origin_node_idx)

    # Convert node indices back to COMIDs
    ancestor_comids = [graph.get_node_data(idx) for idx in ancestor_indices]

    # Include the origin
    all_comids = ancestor_comids + [origin_comid]

    return all_comids


def create_coo(
    subset_comids: list[int], merit_mapping: dict[int, int], graph: rx.PyDiGraph, node_indices: dict[int, int]
) -> tuple[sparse.coo_matrix, list[int]]:
    """
    Create a COO matrix for the subset indexed from the MERIT adjacency matrix.

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

    # Build edges within the subset
    for comid in subset_comids:
        if comid not in node_indices:
            continue

        node_idx = node_indices[comid]

        # Get successors (downstream connections)
        successors = graph.successors(node_idx)

        for ds_comid in successors:
            # Only add edge if downstream COMID is also in subset
            if ds_comid in subset_set:
                row_idx.append(merit_mapping[ds_comid])
                col_idx.append(merit_mapping[comid])

    if len(row_idx) == 0:
        # Headwater - no upstream connections
        return sparse.coo_matrix((len(merit_mapping), len(merit_mapping)), dtype=np.uint8), subset_comids

    coo = sparse.coo_matrix(
        (np.ones(len(row_idx), dtype=np.uint8), (row_idx, col_idx)),
        shape=(len(merit_mapping), len(merit_mapping)),
        dtype=np.uint8,
    )

    assert np.all(coo.row >= coo.col), "Matrix is not lower triangular"
    return coo, subset_comids


def coo_to_zarr_group(
    coo: sparse.coo_matrix,
    ts_order: list[int],
    origin_comid: int,
    gauge_root: zarr.Group,
    merit_mapping: dict[int, int],
) -> None:
    """
    Save a COO matrix to a zarr group.

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
    gauge_root.attrs["gage_comid"] = int(origin_comid)
    gauge_root.attrs["gage_idx"] = int(merit_mapping[origin_comid])
    gauge_root.attrs["data_types"] = {
        "indices_0": str(coo.row.dtype),
        "indices_1": str(coo.col.dtype),
        "values": str(coo.data.dtype),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create subset COO matrices for MERIT gauges from the MERIT adjacency matrix."
    )
    parser.add_argument(
        "--pkg",
        type=Path,
        default="/Users/taddbindas/projects/forks/taddyb/ddr/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp",
        help="Path to the MERIT shapefile.",
    )
    parser.add_argument(
        "--gages",
        default="/Users/taddbindas/projects/forks/taddyb/datasets/mhpi/dHBV2.0UH/training_gauges.csv",
        type=Path,
        help="The gauges CSV file containing the training locations with COMID column.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to save the gages group. Defaults to current working directory",
    )
    parser.add_argument(
        "--merit-adj",
        type=Path,
        required=False,
        default="/Users/taddbindas/projects/forks/taddyb/ddr/data/merit_adjacency.zarr",
        help="Path where the MERIT adjacency matrix is stored.",
    )
    args = parser.parse_args()

    if args.path is None:
        out_path = Path.cwd() / "data/merit_gages_adjacency.zarr"
        out_path.parent.mkdir(exist_ok=True)
    else:
        out_path = Path(args.path)

    merit_path = Path(args.merit_adj)
    if not merit_path.exists():
        raise FileNotFoundError(f"Cannot find {merit_path}")

    gauge_path = Path(args.gages)
    if not gauge_path.exists():
        raise FileNotFoundError(f"Cannot find gauge file: {gauge_path}")

    # Read gauge CSV
    print("Reading gauge data...")
    gauges_df = pl.read_csv(gauge_path, columns=["STAID", "COMID"])

    # Read MERIT shapefile
    print(f"Reading MERIT data from {args.pkg}")
    fp = gpd.read_file(args.pkg)

    # Build network graph
    print("Building upstream connectivity dictionary...")
    upstream_dict = _build_upstream_dict_from_merit(fp)

    print("Building RustWorkX graph...")
    graph, node_indices = _build_rustworkx_object(upstream_dict)
    print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

    # Read MERIT adjacency matrix
    print("Reading MERIT zarr store...")
    merit_root = zarr.open_group(store=merit_path)
    ts_order = merit_root["order"][:]
    merit_mapping = {comid: idx for idx, comid in enumerate(ts_order)}

    # Create local zarr store
    store = zarr.storage.LocalStore(root=out_path)
    if out_path.exists():
        root = zarr.open_group(store=store)
    else:
        root = zarr.create_group(store=store)

    # Process each gauge
    for row in tqdm(
        gauges_df.iter_rows(named=True), total=len(gauges_df), desc="Creating Gauge COO matrices"
    ):
        staid = str(row["STAID"])
        origin_comid = int(row["COMID"])

        try:
            gauge_root = root.create_group(staid)
        except zarr.errors.ContainsGroupError:
            print(f"Zarr Group exists for: {staid}. Skipping write")
            continue

        # Check if COMID exists in the MERIT mapping
        if origin_comid not in merit_mapping:
            print(f"COMID {origin_comid} for gauge {staid} not found in MERIT adjacency matrix. Skipping.")
            root.__delitem__(staid)
            continue

        # Get subset of upstream COMIDs
        subset_comids = subset(origin_comid, graph, node_indices)

        if len(subset_comids) == 1:
            print(
                f"Gauge {str(staid).zfill(8)} (COMID {origin_comid}) is a headwater catchment. Skipping write"
            )
            root.__delitem__(staid)
            continue

        # Create COO matrix for subset
        coo, subset_list = create_coo(subset_comids, merit_mapping, graph, node_indices)

        # Save to zarr
        coo_to_zarr_group(
            coo=coo,
            ts_order=subset_list,
            origin_comid=origin_comid,
            gauge_root=gauge_root,
            merit_mapping=merit_mapping,
        )

    print(f"MERIT Gauge adjacency matrices written to {out_path}")
