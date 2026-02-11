"""Patch dhbv2_gages.csv with standard columns for consistency with GAGES-II reference files.

Adds:
  - STANAME         (= STAID, no station names available)
  - COMID_DRAIN_SQKM (= zone_edge_uparea)
  - ABS_DIFF         (= abs(zone_edge_vs_gage_area_difference))
  - COMID_UNITAREA_SQKM (looked up from MERIT catchments shapefile by COMID)

WARNING — Drainage area source differs from GAGES-II / camels / gages_3000:
  This script sets COMID_DRAIN_SQKM = zone_edge_uparea (the discretized upstream
  area from the dHBV2.0 zone-edge network), NOT the MERIT rivers native ``uparea``.
  The other gage reference files (GAGES-II.csv, camels_670.csv, gages_3000.csv) use
  MERIT rivers ``uparea`` for COMID_DRAIN_SQKM. This means ABS_DIFF is NOT directly
  comparable across datasets.

  To add comparable MERIT-based metrics, pass ``--merit-rivers`` to generate
  MERIT_DRAIN_SQKM and MERIT_ABS_DIFF columns.

Usage:
    python references/geo_io/patch_dhbv2_gages.py \
        --csv references/gage_info/dhbv2_gages.csv \
        --merit-catchments data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp

    # With MERIT rivers for comparable error metrics:
    python references/geo_io/patch_dhbv2_gages.py \
        --csv references/gage_info/dhbv2_gages.csv \
        --merit-catchments data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp \
        --merit-rivers /path/to/riv_pfaf_7_MERIT_Hydro_v07.shp
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
    parser.add_argument(
        "--merit-rivers",
        type=Path,
        default=None,
        help="Path to MERIT river shapefile (for native uparea lookup). "
        "When provided, adds MERIT_DRAIN_SQKM, MERIT_ABS_DIFF, and DA_VALID columns.",
    )
    return parser.parse_args()


def merge_merit_uparea(df: pd.DataFrame, merit_rivers_path: Path) -> pd.DataFrame:
    """Add MERIT native uparea columns alongside existing zone_edge-based ones.

    Loads the MERIT rivers shapefile, maps COMID → uparea as MERIT_DRAIN_SQKM,
    and computes MERIT_ABS_DIFF. Also adds DA_VALID column using the simpler
    ABS_DIFF <= COMID_UNITAREA_SQKM check.

    Parameters
    ----------
    df : pd.DataFrame
        dhbv2 gage DataFrame with COMID, DRAIN_SQKM, and COMID_UNITAREA_SQKM columns.
    merit_rivers_path : Path
        Path to MERIT river shapefile (for uparea lookup).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with MERIT_DRAIN_SQKM, MERIT_ABS_DIFF, and DA_VALID columns added.
    """
    print(f"Loading MERIT rivers for native uparea lookup: {merit_rivers_path}")
    rivers = gpd.read_file(
        merit_rivers_path,
        columns=["COMID", "uparea"],
        ignore_geometry=True,
    )
    uparea_map = rivers.set_index("COMID")["uparea"].to_dict()

    df["MERIT_DRAIN_SQKM"] = df["COMID"].map(uparea_map)
    matched = df["MERIT_DRAIN_SQKM"].notna().sum()
    print(f"  {matched}/{len(df)} gages matched to MERIT uparea")

    df["MERIT_ABS_DIFF"] = (df["DRAIN_SQKM"] - df["MERIT_DRAIN_SQKM"]).abs()

    # DA_VALID: mismatch within one local catchment (floored at 100 km²)
    da_threshold = df["COMID_UNITAREA_SQKM"].clip(lower=100.0)
    df["DA_VALID"] = df["MERIT_ABS_DIFF"] <= da_threshold
    n_valid = df["DA_VALID"].sum()
    print(f"  DA_VALID: {n_valid}/{len(df)} gages pass (MERIT_ABS_DIFF <= max(COMID_UNITAREA_SQKM, 100))")

    # Print comparison summary
    zone_edge_median = df["ABS_DIFF"].median()
    merit_median = df["MERIT_ABS_DIFF"].median()
    print("\n  Median ABS_DIFF comparison:")
    print(f"    zone_edge (COMID_DRAIN_SQKM): {zone_edge_median:.2f} km²")
    print(f"    MERIT native (MERIT_DRAIN_SQKM): {merit_median:.2f} km²")

    return df


def patch(csv_path: Path, merit_catchments_path: Path, merit_rivers_path: Path | None = None) -> None:
    """Patch the dhbv2 gage CSV with standard columns."""
    df = pd.read_csv(csv_path, dtype={"STAID": str})

    # Derive standard columns from existing dhbv2-specific data
    df["STAID"] = df["STAID"].str.zfill(8)
    df["STANAME"] = df["STAID"]
    df["COMID_DRAIN_SQKM"] = df["zone_edge_uparea"]
    df["ABS_DIFF"] = df["zone_edge_vs_gage_area_difference"].abs()

    # Look up unitarea from MERIT catchments shapefile
    print(f"Loading MERIT catchments for unitarea lookup: {merit_catchments_path}")
    catchments = gpd.read_file(merit_catchments_path, columns=["COMID", "unitarea"], ignore_geometry=True)
    unitarea_map = catchments.set_index("COMID")["unitarea"].to_dict()
    df["COMID_UNITAREA_SQKM"] = df["COMID"].map(unitarea_map)

    matched = df["COMID_UNITAREA_SQKM"].notna().sum()
    print(f"  {matched}/{len(df)} gages matched to unitarea")

    # Optionally add MERIT native uparea columns for cross-dataset comparison
    if merit_rivers_path is not None:
        df = merge_merit_uparea(df, merit_rivers_path)

    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")


if __name__ == "__main__":
    args = parse_args()
    patch(args.csv, args.merit_catchments, args.merit_rivers)
