"""A script to build adjaency matrices for MERIT flowpaths"""

from pathlib import Path

import geopandas as gpd
import zarr
from ddr_engine.merit import (
    _build_rustworkx_object,
    _build_upstream_dict_from_merit,
    coo_to_zarr,
    coo_to_zarr_group,
    create_coo,
    create_matrix,
    subset,
)
from tqdm import tqdm

from ddr.dataset import GaugeSet, MERITGauge, validate_gages

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create a lower triangular adjacency matrix from MERIT hydrofabric data."
    )
    parser.add_argument(
        "--pkg",
        type=Path,
        required=False,
        default="/projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp",
        help="Path to the MERIT shapefile.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default="/projects/mhpi/tbindas/ddr/data/tmp/",
        required=False,
        help="Path to save the zarr group. Defaults to current working directory",
    )
    parser.add_argument(
        "--gages",
        type=Path,
        required=False,
        default="/projects/mhpi/tbindas/datasets/mhpi/dHBV2.0UH/training_gauges.csv",
        help="The gauges CSV file containing the training locations with COMID column.",
    )
    args = parser.parse_args()

    if args.path is None:
        raise FileNotFoundError("Path not provided for zarr group outputs")
    else:
        out_path = Path(args.path) / "merit_conus_adjacency.zarr"
        out_path.parent.mkdir(exist_ok=True)

        if args.gages is not None:
            gage_path = Path(args.gages)
            gauge_set: GaugeSet = validate_gages(gage_path, type=MERITGauge)
            gages_out_path = Path(args.path) / "merit_gages_conus_adjacency.zarr"
            if gages_out_path.exists():
                print(f"Cannot create zarr store {gages_out_path}. One already exists")
                exit(1)
        else:
            gages_out_path = None

    if out_path.exists():
        print(f"Cannot create zarr store {out_path}. One already exists")
        exit(1)

    print(f"Reading MERIT data from {args.pkg}")
    fp = gpd.read_file(args.pkg)

    print(f"Creating adjacency matrix for {len(fp)} flowpaths")
    matrix, ts_order = create_matrix(fp)

    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    coo_to_zarr(matrix, ts_order, out_path)

    if gages_out_path is not None:
        # Build network graph
        print("Building upstream connectivity dictionary...")
        upstream_dict = _build_upstream_dict_from_merit(fp)

        print("Building RustWorkX graph...")
        graph, node_indices = _build_rustworkx_object(upstream_dict)
        print(f"Graph has {graph.num_nodes()} nodes and {graph.num_edges()} edges")

        # Read MERIT adjacency matrix
        print("Reading MERIT zarr store...")
        merit_root = zarr.open_group(store=out_path)
        ts_order = merit_root["order"][:]
        merit_mapping = {comid: idx for idx, comid in enumerate(ts_order)}

        # Create local zarr store
        store = zarr.storage.LocalStore(root=out_path)
        if out_path.exists():
            root = zarr.open_group(store=store)
        else:
            root = zarr.create_group(store=store)

        # Process each gauge
        for gauge in tqdm(gauge_set.gauges, desc="Creating Gauge COO matrices"):
            staid = gauge.STAID
            origin_comid = gauge.COMID

            try:
                gauge_root = root.create_group(staid)
            except zarr.errors.ContainsGroupError:
                print(f"Zarr Group exists for: {staid}. Skipping write")
                continue

            # Check if COMID exists in the MERIT mapping
            if origin_comid not in merit_mapping:
                print(
                    f"COMID {origin_comid} for gauge {staid} not found in MERIT adjacency matrix. Skipping."
                )
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
