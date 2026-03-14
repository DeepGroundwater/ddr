"""Build SWOT/SWORD geometry NetCDF and GeoPackage for MERIT COMIDs.

Uses the MERIT-SWORD translate shapefiles (Wade et al., 2025; Zenodo 14675925)
which contain SWORD reaches with type, diag_flag, width, and MB COMID translations
all in one file per region.

Pipeline:
  1. Load ms_region_overlap CSVs to determine which SWORD region files are needed.
  2. Load SWORD translate shapefiles (ms_translate_shp/sword/).
  3. Filter to type=1 (river) and diag_flag=0 (valid translation).
  4. Expand each SWORD reach width to its MB COMIDs (mb_1..mb_40).
     If a COMID maps to multiple SWORD reaches, weighted average by part_len.
  5. Optionally join EIV side_slope (independent, from SWOT observations).
  6. Save NetCDF (COMID-indexed, for routing engine).
  7. Save GeoPackage with two layers:
     - sword_reaches: filtered SWORD reaches with geometry
     - merit_network: complete MERIT river network with top_width/side_slope

References
----------
  Wade, J., David, C. H., Altenau, E. H., et al. (2025). Bidirectional
  Translations Between Observational and Topography-Based Hydrographic
  Data Sets: MERIT-Basins and the SWOT River Database (SWORD). Water
  Resources Research, 61, e2024WR038633. https://doi.org/10.1029/2024WR038633

Usage:
  python references/geo_io/build_swot_geometry.py \
    --merit-sword-dir /mnt/ssd1/data/swot/merit_sword/ \
    --eiv-fits-dir /mnt/ssd1/data/swot/swot-river-volume/output_revision/output_revision/EIV_fits/ \
    --output data/swot_merit_geometry.nc \
    --gpkg data/swot_merit_geometry.gpkg
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

# MERIT-Basins Pfafstetter level 2 regions for CONUS
MB_PFAF_REGIONS = [71, 72, 73, 74, 75, 76, 77, 78]

# Column name mapping for part_len (shapefile truncates names > 10 chars)
_PART_LEN_COLS = {j: f"part_len_{j}" if j <= 9 else f"part_len{j}" for j in range(1, 41)}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build SWOT/SWORD geometry NetCDF + GeoPackage for MERIT COMIDs "
        "using MERIT-SWORD translations (Wade et al., 2025).",
    )
    parser.add_argument(
        "--merit-sword-dir",
        type=Path,
        required=True,
        help="Root of MERIT-SWORD dataset (contains ms_translate_shp/, ms_region_overlap/, etc.).",
    )
    parser.add_argument(
        "--eiv-fits-dir",
        type=Path,
        default=None,
        help="Optional: directory containing swot_vol_fits_*.csv for SWOT-derived side_slope.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NetCDF path (e.g. data/swot_merit_geometry.nc).",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=None,
        help="Optional: output GeoPackage path (e.g. data/swot_merit_geometry.gpkg).",
    )
    parser.add_argument(
        "--min-nobs",
        type=int,
        default=10,
        help="Minimum SWOT observations to use EIV side_slope (default: 10).",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Region overlap
# ──────────────────────────────────────────────────────────────────────────────
def load_sword_regions_for_mb(
    merit_sword_dir: Path,
    mb_regions: list[int],
) -> list[int]:
    """Determine which SWORD regions are needed for a set of MB PFAF regions.

    Uses ms_region_overlap/mb_to_sword_reg_overlap.csv (Wade et al., 2025).
    """
    csv_path = merit_sword_dir / "ms_region_overlap" / "mb_to_sword_reg_overlap.csv"
    overlap = pd.read_csv(csv_path)
    sword_regs: set[int] = set()
    for pfaf in mb_regions:
        row = overlap[overlap["mb"] == pfaf]
        if row.empty:
            print(f"  WARNING: MB region {pfaf} not in overlap table")
            continue
        for col in ["sword0", "sword1", "sword2", "sword3", "sword4"]:
            val = row.iloc[0][col]
            if pd.notna(val):
                sword_regs.add(int(val))
    result = sorted(sword_regs)
    print(f"  MB regions {mb_regions} -> SWORD regions needed: {result}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# SWORD translate shapefiles
# ──────────────────────────────────────────────────────────────────────────────
def load_sword_translates(
    merit_sword_dir: Path,
    sword_regions: list[int],
) -> gpd.GeoDataFrame:
    """Load SWORD translate shapefiles for the given SWORD regions.

    These are from ms_translate_shp/sword/ and contain: reach_id, type, width,
    diag_flag, mb_1..mb_40, part_len_1..part_len40, and SWORD reach geometry.
    """
    shp_dir = merit_sword_dir / "ms_translate_shp" / "ms_translate_shp" / "sword"
    frames = []
    for reg in sword_regions:
        shp_path = shp_dir / f"na_sword_reaches_hb{reg}_v16_translate.shp"
        if not shp_path.exists():
            print(f"  WARNING: not found: {shp_path}")
            continue
        gdf = gpd.read_file(shp_path)
        frames.append(gdf)
        has_wid = gdf["width"].notna() & (gdf["width"] > 0)
        n_valid = ((gdf["diag_flag"] == 0) & has_wid).sum()
        print(f"  hb{reg}: {len(gdf):,} reaches, {n_valid:,} flag=0 with width")

    if not frames:
        raise FileNotFoundError(f"No SWORD translate shapefiles found in {shp_dir}")

    result = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=frames[0].crs)


def filter_sword_reaches(sword_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter to diag_flag=0 (valid translation) and width > 0.

    All SWORD types are included (river, lake, dam, unreliable, ghost) to
    maximize coverage. Major rivers passing through reservoirs are often
    classified as type=3 (lake) in SWORD; excluding them loses high-order
    mainstem reaches. The diag_flag=0 filter ensures translation quality.
    """
    total = len(sword_gdf)
    has_width = sword_gdf["width"].notna() & (sword_gdf["width"] > 0)
    mask = (sword_gdf["diag_flag"] == 0) & has_width
    filtered = sword_gdf[mask].copy()

    # Print type/flag breakdown
    type_names = {1: "river", 3: "lake", 4: "dam/waterfall", 5: "unreliable", 6: "ghost"}
    type_counts = sword_gdf["type"].value_counts().sort_index()
    for t, c in type_counts.items():
        print(f"    type={t} ({type_names.get(t, '?')}): {c:,} ({100 * c / total:.1f}%)")

    flag_names = {
        0: "valid",
        1: "topo discontinuity",
        2: "no translation",
        21: "facc mismatch",
        22: "coastal",
    }
    flag_counts = sword_gdf["diag_flag"].value_counts().sort_index()
    for f, c in flag_counts.items():
        print(f"    flag={f} ({flag_names.get(f, '?')}): {c:,} ({100 * c / total:.1f}%)")

    print(f"  After filter: {len(filtered):,} / {total:,} reaches")
    return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Expand SWORD width → MB COMIDs
# ──────────────────────────────────────────────────────────────────────────────
def expand_sword_to_comids(sword_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Expand each SWORD reach width to all its MB COMIDs.

    For each filtered SWORD reach, assign its width to every non-zero mb_j.
    If a COMID appears from multiple SWORD reaches, compute weighted average
    by part_len (partial intersecting length in meters).

    Returns DataFrame indexed by COMID with top_width column.
    """
    # Collect (COMID, width, part_len) triples
    rows: list[tuple[int, float, float]] = []
    for _, reach in sword_gdf.iterrows():
        width = float(reach["width"])
        if np.isnan(width) or width <= 0:
            continue
        for j in range(1, 41):
            mb_col = f"mb_{j}"
            pl_col = _PART_LEN_COLS[j]
            if mb_col not in reach.index or pl_col not in reach.index:
                break
            comid = int(reach[mb_col])
            if comid == 0:
                break  # mb columns are ranked; 0 means no more matches
            part_len = float(reach[pl_col])
            if part_len <= 0:
                part_len = 1.0  # fallback weight
            rows.append((comid, width, part_len))

    df = pd.DataFrame(rows, columns=["COMID", "width", "part_len"])
    print(f"  Expanded to {len(df):,} (COMID, width, part_len) triples")
    print(f"  Unique COMIDs: {df['COMID'].nunique():,}")

    # Weighted average: top_width = sum(width_i * part_len_i) / sum(part_len_i)
    df["w_x_pl"] = df["width"] * df["part_len"]
    agg = df.groupby("COMID").agg(w_x_pl_sum=("w_x_pl", "sum"), pl_sum=("part_len", "sum"))
    agg["top_width"] = agg["w_x_pl_sum"] / agg["pl_sum"]

    result = pd.DataFrame({"top_width": agg["top_width"]})
    result.index.name = "COMID"
    print(f"  COMIDs with top_width: {len(result):,}")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# EIV side_slope (same as before)
# ──────────────────────────────────────────────────────────────────────────────
def load_eiv_fits(eiv_dir: Path, pfaf_regions: list[int], min_nobs: int) -> pd.DataFrame:
    """Load EIV_fits CSVs for SWOT-derived side_slope (m_1).

    Returns DataFrame indexed by reach_id with m_1 and nobs.
    """
    frames = []
    for pfaf in pfaf_regions:
        pattern = f"swot_vol_fits_{pfaf}_*.csv"
        matches = list(eiv_dir.glob(pattern))
        if not matches:
            print(f"  WARNING: no EIV file found for PFAF {pfaf}")
            continue
        for csv_path in matches:
            df = pd.read_csv(csv_path)
            frames.append(df)
            print(f"  Loaded {csv_path.name}: {len(df)} reaches")

    if not frames:
        print("  WARNING: No EIV_fits CSVs found — side_slope will be NaN everywhere")
        return pd.DataFrame(columns=["m_1", "nobs", "med_flow_area"])

    eiv = pd.concat(frames, ignore_index=True)

    n_before = len(eiv)
    eiv = eiv[eiv["nobs"] >= min_nobs].copy()
    print(f"  Filtered {n_before - len(eiv)} reaches with nobs < {min_nobs} ({len(eiv)} remaining)")

    eiv = eiv.drop_duplicates(subset="reach_id", keep="first")
    eiv = eiv.set_index("reach_id")
    print(f"  Total unique EIV reaches: {len(eiv)}")
    return eiv[["m_1", "nobs", "med_flow_area"]]


def join_eiv_to_comids(
    eiv: pd.DataFrame,
    sword_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Join EIV side_slope to MERIT COMIDs via the SWORD translate shapefile.

    Uses the mb_1..mb_40 columns from the (already filtered) SWORD reaches.
    For each COMID, if multiple SWORD reaches have EIV data, keep highest nobs.
    """
    rows: list[dict] = []
    for _, reach in sword_gdf.iterrows():
        reach_id = int(reach["reach_id"])
        if reach_id not in eiv.index:
            continue
        m_1 = float(eiv.loc[reach_id, "m_1"])
        nobs = int(eiv.loc[reach_id, "nobs"])
        if np.isnan(m_1):
            continue
        side_slope = float(np.clip(m_1 / 2.0, 0.5, 50.0))
        for j in range(1, 41):
            mb_col = f"mb_{j}"
            if mb_col not in reach.index:
                break
            comid = int(reach[mb_col])
            if comid == 0:
                break
            rows.append({"COMID": comid, "side_slope": side_slope, "nobs": nobs})

    if not rows:
        return pd.DataFrame(columns=["side_slope", "nobs"]).rename_axis("COMID")

    df = pd.DataFrame(rows)
    df = df.sort_values("nobs", ascending=False).drop_duplicates(subset="COMID", keep="first")
    return df.set_index("COMID")


# ──────────────────────────────────────────────────────────────────────────────
# MB translate shapefiles (complete MERIT network)
# ──────────────────────────────────────────────────────────────────────────────
def load_merit_network(merit_sword_dir: Path, mb_regions: list[int]) -> gpd.GeoDataFrame:
    """Load complete MERIT river network from MB translate shapefiles.

    Returns GeoDataFrame indexed by COMID with geometry and selected attributes.
    """
    shp_dir = merit_sword_dir / "ms_translate_shp" / "ms_translate_shp" / "mb"
    frames = []
    for pfaf in mb_regions:
        shp_path = shp_dir / f"riv_pfaf_{pfaf}_MERIT_Hydro_v07_Basins_v01_translate.shp"
        if not shp_path.exists():
            print(f"  WARNING: not found: {shp_path}")
            continue
        gdf = gpd.read_file(shp_path, columns=["COMID", "uparea", "slope", "order", "lengthkm"])
        frames.append(gdf)
        print(f"  PFAF {pfaf}: {len(gdf):,} reaches")

    result = pd.concat(frames, ignore_index=True)
    gdf_out = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326").set_index("COMID")
    print(f"  Total MERIT reaches: {len(gdf_out):,}")
    return gdf_out


# ──────────────────────────────────────────────────────────────────────────────
# Output
# ──────────────────────────────────────────────────────────────────────────────
def save_netcdf(geometry: pd.DataFrame, output_path: Path) -> None:
    """Save geometry DataFrame as NetCDF with COMID dimension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "top_width": ("COMID", geometry["top_width"].values.astype(np.float32)),
            "side_slope": ("COMID", geometry["side_slope"].values.astype(np.float32)),
        },
        coords={"COMID": geometry.index.values},
        attrs={
            "description": (
                "SWORD v16 river geometry mapped to MERIT COMIDs via "
                "MERIT-SWORD translations (Wade et al., 2025)"
            ),
            "top_width_units": "meters",
            "top_width_derivation": (
                "SWORD v16 width transferred to MB COMIDs via sword_to_mb translations. "
                "Weighted average by part_len when multiple SWORD reaches overlap a COMID. "
                "Filtered to diag_flag=0 (valid) with width > 0 (all SWORD types included)."
            ),
            "side_slope_units": "H:V ratio (dimensionless)",
            "side_slope_derivation": (
                "m_1/2 from SWOT EIV fits where available, NaN otherwise, "
                "clamped [0.5, 50.0]. Independent of top_width coverage."
            ),
            "source_sword": "SWORD v16 (Altenau et al., 2021)",
            "source_eiv": "SWOT River Volume EIV fits",
            "source_crosswalk": "MERIT-SWORD v0.4 (Wade et al., 2025; Zenodo 14675925)",
        },
    )
    ds.to_netcdf(output_path)
    print(f"\nSaved {output_path} ({len(geometry)} COMIDs)")


def save_gpkg(
    geometry: pd.DataFrame,
    sword_filtered: gpd.GeoDataFrame,
    merit_network: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Save GeoPackage with two layers: sword_reaches and merit_network."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to avoid appending to stale data
    if output_path.exists():
        output_path.unlink()

    # Layer 1: SWORD reaches (flag=0, width>0) with their geometry
    sword_out = sword_filtered[["reach_id", "type", "width", "diag_flag", "geometry"]].copy()
    sword_out.to_file(output_path, driver="GPKG", layer="sword_reaches")
    print(f"  Layer sword_reaches: {len(sword_out):,} features")

    # Layer 2: Complete MERIT network with joined attributes
    merit_out = merit_network.copy()
    merit_out["top_width"] = np.nan
    merit_out["side_slope"] = np.nan
    merit_out["data_source"] = "no_data"

    # Join computed attributes for COMIDs that have data
    common = merit_out.index.intersection(geometry.index)
    merit_out.loc[common, "top_width"] = geometry.loc[common, "top_width"].values
    merit_out.loc[common, "side_slope"] = geometry.loc[common, "side_slope"].values

    has_tw = merit_out["top_width"].notna()
    has_ss = merit_out["side_slope"].notna()
    merit_out.loc[has_tw & ~has_ss, "data_source"] = "width_only"
    merit_out.loc[has_tw & has_ss, "data_source"] = "width+slope"
    merit_out.loc[~has_tw & has_ss, "data_source"] = "slope_only"

    merit_out.to_file(output_path, driver="GPKG", layer="merit_network", mode="a")
    print(
        f"  Layer merit_network: {len(merit_out):,} features "
        f"({has_tw.sum():,} width, {has_ss.sum():,} slope, "
        f"{(~has_tw & ~has_ss).sum():,} no data)"
    )

    print(f"\nSaved {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    """Build SWOT/SWORD geometry NetCDF + GeoPackage using MERIT-SWORD translations."""
    args = parse_args()

    # Step 1: Determine SWORD regions via region overlap
    print("Step 1: Loading region overlap...")
    sword_regions = load_sword_regions_for_mb(args.merit_sword_dir, MB_PFAF_REGIONS)

    # Step 2: Load SWORD translate shapefiles
    print("\nStep 2: Loading SWORD translate shapefiles...")
    sword_gdf = load_sword_translates(args.merit_sword_dir, sword_regions)

    # Step 3: Filter to type=1 (river) + diag_flag=0 (valid)
    print("\nStep 3: Filtering SWORD reaches...")
    sword_filtered = filter_sword_reaches(sword_gdf)

    # Step 4: Expand SWORD width → MB COMIDs
    print("\nStep 4: Expanding SWORD width to MB COMIDs...")
    comid_widths = expand_sword_to_comids(sword_filtered)

    # Step 5: Load EIV side_slope (independent, optional)
    eiv_comids = None
    if args.eiv_fits_dir is not None:
        print("\nStep 5: Loading EIV fits (for side_slope)...")
        eiv = load_eiv_fits(args.eiv_fits_dir, sword_regions, args.min_nobs)
        if len(eiv) > 0:
            eiv_comids = join_eiv_to_comids(eiv, sword_filtered)
            print(f"  EIV side_slope mapped to {len(eiv_comids):,} COMIDs")

    # Step 6: Build output geometry (union of width + side_slope COMIDs)
    print("\nStep 6: Building geometry...")
    geometry = comid_widths.copy()
    geometry["side_slope"] = np.nan

    if eiv_comids is not None and len(eiv_comids) > 0:
        # Update side_slope for COMIDs that already have width
        common = geometry.index.intersection(eiv_comids.index)
        geometry.loc[common, "side_slope"] = eiv_comids.loc[common, "side_slope"]

        # Add COMIDs that have EIV but no width
        eiv_only = eiv_comids.index.difference(geometry.index)
        if len(eiv_only) > 0:
            eiv_only_df = pd.DataFrame(index=eiv_only)
            eiv_only_df["top_width"] = np.nan
            eiv_only_df["side_slope"] = eiv_comids.loc[eiv_only, "side_slope"]
            geometry = pd.concat([geometry, eiv_only_df])

    # Summary
    tw = geometry["top_width"]
    ss = geometry["side_slope"]
    tw_valid = tw.notna()
    ss_valid = ss.notna()
    print("\nSummary:")
    print(f"  Total COMIDs in NetCDF: {len(geometry):,}")
    print(f"  top_width:  {tw_valid.sum():,} valid ({100 * tw_valid.mean():.1f}%)")
    if tw_valid.any():
        print(
            f"    median={tw[tw_valid].median():.1f}m, range=[{tw[tw_valid].min():.1f}, {tw[tw_valid].max():.1f}]"
        )
    print(f"  side_slope: {ss_valid.sum():,} valid ({100 * ss_valid.mean():.1f}%)")
    if ss_valid.any():
        print(
            f"    median={ss[ss_valid].median():.2f}, range=[{ss[ss_valid].min():.2f}, {ss[ss_valid].max():.2f}]"
        )

    # Step 7: Save NetCDF
    print("\nStep 7: Saving NetCDF...")
    save_netcdf(geometry, args.output)

    # Step 8: Save GeoPackage (optional)
    if args.gpkg is not None:
        print("\nStep 8: Loading complete MERIT network for GeoPackage...")
        merit_network = load_merit_network(args.merit_sword_dir, MB_PFAF_REGIONS)
        print("\nSaving GeoPackage...")
        save_gpkg(geometry, sword_filtered, merit_network, args.gpkg)

    print("Done.")


if __name__ == "__main__":
    main()
