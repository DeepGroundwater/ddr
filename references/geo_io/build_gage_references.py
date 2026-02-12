"""Build gage reference CSVs by spatially joining GAGES-II points to MERIT catchments.

Pipeline:
  1. Load MERIT catchments (assume 4326) → reproject to 5070
  2. Load GAGES-II gage points (already 5070)
  3. Spatial join: assign each gage a MERIT COMID
  4. Merge upstream area from MERIT rivers
  5. Compute ABS_DIFF, DA_VALID, FLOW_SCALE
  6. Write three filtered CSVs: GAGES-II.csv, camels_670.csv, gages_3000.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

OUTPUT_COLUMNS = [
    "STAID",
    "STANAME",
    "DRAIN_SQKM",
    "LAT_GAGE",
    "LNG_GAGE",
    "COMID",
    "COMID_DRAIN_SQKM",
    "COMID_UNITAREA_SQKM",
    "ABS_DIFF",
    "DA_VALID",
    "FLOW_SCALE",
]


def parse_args() -> argparse.Namespace:
    """A helper function to hold all arguments"""
    parser = argparse.ArgumentParser(description="Build gage reference CSVs with MERIT COMID assignments.")
    parser.add_argument(
        "--gages-gpkg",
        type=Path,
        required=True,
        help="Path to GAGES-II geopackage (EPSG:5070 points).",
    )
    parser.add_argument(
        "--merit-catchments",
        type=Path,
        required=True,
        help="Path to MERIT catchment shapefile (polygons, assumed EPSG:4326).",
    )
    parser.add_argument(
        "--merit-rivers",
        type=Path,
        required=True,
        help="Path to MERIT river shapefile (for uparea lookup, no geometry needed).",
    )
    parser.add_argument(
        "--camels-ids",
        type=Path,
        required=True,
        help="Path to camels_name.txt (semicolon-delimited, has gauge_id column).",
    )
    parser.add_argument(
        "--gages3000-ids",
        type=Path,
        required=True,
        help="Path to gages3000Info.csv (has id column).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write output CSVs.",
    )
    return parser.parse_args()


def load_merit_catchments(path: Path) -> gpd.GeoDataFrame:
    """Load MERIT catchment polygons, set CRS to 4326, reproject to 5070."""
    print(f"Loading MERIT catchments: {path}")
    catchments = gpd.read_file(path)
    catchments = catchments.set_crs(epsg=4326)
    catchments = catchments.to_crs(epsg=5070)
    print(f"  {len(catchments)} catchments loaded")
    return catchments


def load_gages(path: Path) -> gpd.GeoDataFrame:
    """Load GAGES-II gage points (already EPSG:5070)."""
    print(f"Loading GAGES-II gages: {path}")
    gages = gpd.read_file(path)
    print(f"  {len(gages)} gages loaded (CRS: {gages.crs})")
    return gages


def spatial_join(gages: gpd.GeoDataFrame, catchments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign each gage a MERIT COMID via point-in-polygon join."""
    print("Performing spatial join (within)...")
    joined = gpd.sjoin(gages, catchments[["COMID", "unitarea", "geometry"]], how="left", predicate="within")
    joined = joined.rename(columns={"unitarea": "COMID_UNITAREA_SQKM"})
    # sjoin adds index_right; drop it
    joined = joined.drop(columns=["index_right"], errors="ignore")
    matched = joined["COMID"].notna().sum()
    print(f"  {matched}/{len(joined)} gages matched to a catchment")
    return joined


def merge_uparea(gages: gpd.GeoDataFrame, merit_rivers_path: Path) -> gpd.GeoDataFrame:
    """Merge upstream area from MERIT rivers by COMID."""
    print(f"Loading MERIT rivers (attributes only): {merit_rivers_path}")
    rivers = gpd.read_file(
        merit_rivers_path,
        columns=["COMID", "uparea"],
        ignore_geometry=True,
    )
    rivers = rivers.rename(columns={"uparea": "COMID_DRAIN_SQKM"})
    rivers["COMID"] = rivers["COMID"].astype("Int64")
    print(f"  {len(rivers)} reaches loaded")

    gages["COMID"] = gages["COMID"].astype("Int64")
    merged = gages.merge(rivers[["COMID", "COMID_DRAIN_SQKM"]], on="COMID", how="left")
    return merged


def compute_flow_scale(df: pd.DataFrame) -> pd.Series:
    """Compute per-gage flow scale factor in [0, 1].

    When a gage sits partway through a catchment, the modeled lateral inflow is
    too large. The scale factor reduces Q' proportionally to the area mismatch:
        scale = (unit_area - |diff|) / unit_area   when diff < 0 and |diff| < unit_area
        scale = 1.0                                 otherwise

    Parameters
    ----------
    df : pd.DataFrame
        Must have DRAIN_SQKM, COMID_DRAIN_SQKM, COMID_UNITAREA_SQKM columns.

    Returns
    -------
    pd.Series[float]
        Flow scale factor per gage, in [0, 1].
    """
    diff = df["DRAIN_SQKM"] - df["COMID_DRAIN_SQKM"]
    abs_diff = diff.abs()
    unit_area = df["COMID_UNITAREA_SQKM"]
    scale = (unit_area - abs_diff) / unit_area
    # Only scale when gage is upstream of COMID outlet (diff < 0) and mismatch < unit area
    scale = scale.where((diff < 0) & (abs_diff < unit_area), 1.0)
    return scale


def load_camels_ids(path: Path) -> set[str]:
    """Load CAMELS gauge IDs from semicolon-delimited file, zero-pad to 8 chars."""
    print(f"Loading CAMELS IDs: {path}")
    df = pd.read_csv(path, sep=";", dtype=str)
    # Strip whitespace from column names and values
    df.columns = df.columns.str.strip()
    ids = {gid.strip().zfill(8) for gid in df["gauge_id"]}
    print(f"  {len(ids)} CAMELS gauge IDs")
    return ids


def load_gages3000_ids(path: Path) -> set[str]:
    """Load Gages-3000 IDs, zero-pad to 8 chars."""
    print(f"Loading Gages-3000 IDs: {path}")
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    ids = {gid.strip().zfill(8) for gid in df["id"]}
    print(f"  {len(ids)} Gages-3000 IDs")
    return ids


def write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write a filtered DataFrame to CSV and print summary."""
    df.to_csv(path, index=False)
    matched = df["COMID_DRAIN_SQKM"].notna().sum()
    median_abs = df["ABS_DIFF"].median()
    print(f"  {label}: {len(df)} rows, {matched} with uparea, median ABS_DIFF={median_abs:.2f} km²")


def main() -> None:
    """The main function for building reference gage information"""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Load data
    catchments = load_merit_catchments(args.merit_catchments)
    gages = load_gages(args.gages_gpkg)

    # 3. Spatial join
    joined = spatial_join(gages, catchments)

    # 4. Merge uparea
    joined = merge_uparea(joined, args.merit_rivers)

    # 5. Compute ABS_DIFF, DA_VALID, FLOW_SCALE
    joined["ABS_DIFF"] = (joined["DRAIN_SQKM"] - joined["COMID_DRAIN_SQKM"]).abs()
    da_threshold = joined["COMID_UNITAREA_SQKM"].clip(lower=100.0)
    joined["DA_VALID"] = joined["ABS_DIFF"] <= da_threshold
    n_valid = joined["DA_VALID"].sum()
    print(f"  DA_VALID: {n_valid}/{len(joined)} gages pass (ABS_DIFF <= max(COMID_UNITAREA_SQKM, 100))")

    joined["FLOW_SCALE"] = compute_flow_scale(joined)
    n_scaled = (joined["FLOW_SCALE"] < 1.0).sum()
    print(f"  FLOW_SCALE: {n_scaled}/{len(joined)} gages have scale < 1.0")

    # Ensure STAID is zero-padded string for filtering
    joined["STAID"] = joined["STAID"].astype(str).str.zfill(8)

    # Build base DataFrame with output columns only
    base = pd.DataFrame(joined[OUTPUT_COLUMNS])

    # 6a. GAGES-II.csv: drop NaN COMIDs + non-standard IDs (>8 digits)
    gagesii = base.dropna(subset=["COMID"]).copy()
    gagesii = gagesii[gagesii["STAID"].str.len() <= 8]
    gagesii["COMID"] = gagesii["COMID"].astype(int)
    write_csv(gagesii, args.output_dir / "GAGES-II.csv", "GAGES-II")

    # 6b. camels_670.csv
    camels_ids = load_camels_ids(args.camels_ids)
    camels = base[base["STAID"].isin(camels_ids)].dropna(subset=["COMID"]).copy()
    camels["COMID"] = camels["COMID"].astype(int)
    write_csv(camels, args.output_dir / "camels_670.csv", "camels_670")

    # 6c. gages_3000.csv
    g3k_ids = load_gages3000_ids(args.gages3000_ids)
    g3k = base[base["STAID"].isin(g3k_ids)].dropna(subset=["COMID"]).copy()
    g3k["COMID"] = g3k["COMID"].astype(int)
    write_csv(g3k, args.output_dir / "gages_3000.csv", "gages_3000")

    print("\nDone.")


if __name__ == "__main__":
    main()
