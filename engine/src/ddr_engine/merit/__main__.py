"""CLI entrypoint for building MERIT adjacency matrices.

Usage:
    python -m ddr_engine.merit <pkg> [--path PATH] [--gages GAGES]
"""

import argparse
from pathlib import Path

import geopandas as gpd

from . import build_gauge_adjacencies, build_merit_adjacency


def main() -> None:
    """The main function for the MERIT engine"""
    parser = argparse.ArgumentParser(
        description="Create lower triangular adjacency matrices from MERIT hydrofabric data."
    )
    parser.add_argument(
        "pkg",
        type=Path,
        help="Path to the hydrofabric geopackage.",
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("data/"),
        help="Path to save the zarr group. Defaults to 'data/'.",
    )
    parser.add_argument(
        "--gages",
        type=Path,
        default=None,
        help="The gauges CSV file containing the training locations.",
    )
    args = parser.parse_args()

    print(f"Reading MERIT data from {args.pkg}")
    fp = gpd.read_file(args.pkg)

    out_path = args.path / "merit_conus_adjacency.zarr"
    build_merit_adjacency(fp, out_path)

    if args.gages is not None:
        from ddr.geodatazoo.dataclasses import GaugeSet, MERITGauge, validate_gages

        gauge_set: GaugeSet = validate_gages(args.gages, type=MERITGauge)
        gages_out_path = args.path / "merit_gages_conus_adjacency.zarr"
        build_gauge_adjacencies(fp, out_path, gauge_set, gages_out_path)


if __name__ == "__main__":
    main()
