"""Patch dhbv2_gages.csv with standard columns for consistency with GAGES-II reference files.

Adds:
  - STANAME         (= STAID, no station names available)
  - COMID_DRAIN_SQKM (= zone_edge_uparea)
  - PCT_DIFF         (= drainage_area_percent_error * 100)
  - REL_ERR          (= -drainage_area_percent_error)
  - ABS_DIFF         (= abs(zone_edge_vs_gage_area_difference))
  - COMID_UNITAREA_SQKM (looked up from MERIT catchments shapefile by COMID)

Usage:
    python references/geo_io/patch_dhbv2_gages.py \
        --csv references/gage_info/dhbv2_gages.csv \
        --merit-catchments data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Patch dhbv2_gages.csv with standard columns.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to dhbv2_gages.csv")
    parser.add_argument(
        "--merit-catchments",
        type=Path,
        required=True,
        help="Path to MERIT catchment shapefile (for unitarea lookup).",
    )
    return parser.parse_args()


def patch(csv_path: Path, merit_catchments_path: Path) -> None:
    """Patch the dhbv2 gage CSV with standard columns."""
    df = pd.read_csv(csv_path)

    # Derive standard columns from existing dhbv2-specific data
    df["STANAME"] = df["STAID"].astype(str)
    df["COMID_DRAIN_SQKM"] = df["zone_edge_uparea"]
    df["PCT_DIFF"] = df["drainage_area_percent_error"] * 100
    df["REL_ERR"] = -df["drainage_area_percent_error"]
    df["ABS_DIFF"] = df["zone_edge_vs_gage_area_difference"].abs()

    # Look up unitarea from MERIT catchments shapefile
    print(f"Loading MERIT catchments for unitarea lookup: {merit_catchments_path}")
    catchments = gpd.read_file(merit_catchments_path, columns=["COMID", "unitarea"], ignore_geometry=True)
    unitarea_map = catchments.set_index("COMID")["unitarea"].to_dict()
    df["COMID_UNITAREA_SQKM"] = df["COMID"].map(unitarea_map)

    matched = df["COMID_UNITAREA_SQKM"].notna().sum()
    print(f"  {matched}/{len(df)} gages matched to unitarea")

    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    patch(args.csv, args.merit_catchments)
