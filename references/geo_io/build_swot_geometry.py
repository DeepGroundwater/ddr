"""Build SWOT satellite geometry NetCDF by joining EIV fits to MERIT COMIDs.

Pipeline:
  1. Read EIV_fits CSVs from PFAF regions 71-78 (North America)
  2. Read MERIT-SWORD crosswalk NetCDFs (mb_to_sword, COMID → SWORD reach_id)
  3. Join: for each MERIT COMID with a SWORD match that has EIV data, derive:
       top_width = clamp(med_width, 1.0, 5000.0)
       side_slope = clamp(m_1 / 2, 0.5, 50.0)
  4. Save as NetCDF with COMID dimension and top_width, side_slope, nobs, med_flow_area

Usage:
  python references/geo_io/build_swot_geometry.py \
    --eiv-fits-dir /mnt/ssd1/data/swot/swot-river-volume/output_revision/output_revision/EIV_fits/ \
    --crosswalk-dir /mnt/ssd1/data/swot/merit_sword/ms_translate/mb_to_sword/ \
    --output data/swot_merit_geometry.nc
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

NA_PFAF_REGIONS = [71, 72, 73, 74, 75, 76, 77, 78]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build SWOT geometry NetCDF for MERIT COMIDs.")
    parser.add_argument(
        "--eiv-fits-dir",
        type=Path,
        required=True,
        help="Directory containing swot_vol_fits_*.csv files.",
    )
    parser.add_argument(
        "--crosswalk-dir",
        type=Path,
        required=True,
        help="Directory containing mb_to_sword_pfaf_*_translate.nc files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NetCDF path (e.g. data/swot_merit_geometry.nc).",
    )
    parser.add_argument(
        "--min-nobs",
        type=int,
        default=10,
        help="Minimum number of SWOT observations to include a reach (default: 10).",
    )
    return parser.parse_args()


def load_eiv_fits(eiv_dir: Path, pfaf_regions: list[int], min_nobs: int) -> pd.DataFrame:
    """Load and concatenate EIV_fits CSVs for the specified PFAF regions.

    Returns DataFrame indexed by reach_id with med_width, m_1, nobs, med_flow_area.
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
        raise FileNotFoundError(f"No EIV_fits CSVs found in {eiv_dir} for PFAF regions {pfaf_regions}")

    eiv = pd.concat(frames, ignore_index=True)

    # Filter by minimum observations
    n_before = len(eiv)
    eiv = eiv[eiv["nobs"] >= min_nobs].copy()
    print(f"  Filtered {n_before - len(eiv)} reaches with nobs < {min_nobs} ({len(eiv)} remaining)")

    # Drop duplicates on reach_id (keep first occurrence)
    eiv = eiv.drop_duplicates(subset="reach_id", keep="first")
    eiv = eiv.set_index("reach_id")
    print(f"  Total unique SWOT reaches: {len(eiv)}")
    return eiv[["med_width", "m_1", "nobs", "med_flow_area"]]


def load_crosswalk(crosswalk_dir: Path, pfaf_regions: list[int]) -> pd.DataFrame:
    """Load MERIT-SWORD crosswalk NetCDFs and return COMID → primary SWORD reach_id mapping.

    Uses only the primary match (sword_1) where sword_1 != 0.
    """
    frames = []
    for pfaf in pfaf_regions:
        nc_path = crosswalk_dir / f"mb_to_sword_pfaf_{pfaf}_translate.nc"
        if not nc_path.exists():
            print(f"  WARNING: crosswalk not found: {nc_path}")
            continue

        ds = xr.open_dataset(nc_path)
        comids = ds["mb"].values
        sword_1 = ds["sword_1"].values
        ds.close()

        df = pd.DataFrame({"COMID": comids, "reach_id": sword_1})
        # Only keep entries with a valid SWORD match
        df = df[df["reach_id"] != 0]
        frames.append(df)
        print(f"  Loaded {nc_path.name}: {len(df)} COMIDs with SWORD match")

    if not frames:
        raise FileNotFoundError(
            f"No crosswalk files found in {crosswalk_dir} for PFAF regions {pfaf_regions}"
        )

    crosswalk = pd.concat(frames, ignore_index=True)
    # If a COMID appears in multiple PFAF files, keep first
    crosswalk = crosswalk.drop_duplicates(subset="COMID", keep="first")
    print(f"  Total COMIDs with SWORD match: {len(crosswalk)}")
    return crosswalk


def derive_geometry(eiv: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Join crosswalk to EIV data and derive top_width and side_slope per COMID."""
    merged = crosswalk.merge(eiv, left_on="reach_id", right_index=True, how="inner")
    print(f"  Matched {len(merged)} COMIDs to SWOT EIV data")

    merged["top_width"] = merged["med_width"].clip(lower=1.0, upper=5000.0)
    merged["side_slope"] = (merged["m_1"] / 2.0).clip(lower=0.5, upper=50.0)

    return merged[["COMID", "top_width", "side_slope", "nobs", "med_flow_area"]].set_index("COMID")


def save_netcdf(geometry: pd.DataFrame, output_path: Path) -> None:
    """Save geometry DataFrame as NetCDF with COMID dimension."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            "top_width": ("COMID", geometry["top_width"].values.astype(np.float32)),
            "side_slope": ("COMID", geometry["side_slope"].values.astype(np.float32)),
            "nobs": ("COMID", geometry["nobs"].values.astype(np.int32)),
            "med_flow_area": ("COMID", geometry["med_flow_area"].values.astype(np.float32)),
        },
        coords={"COMID": geometry.index.values},
        attrs={
            "description": "SWOT satellite geometry mapped to MERIT COMIDs via MERIT-SWORD crosswalk",
            "top_width_units": "meters",
            "side_slope_units": "H:V ratio (dimensionless)",
            "top_width_derivation": "med_width from EIV_fits, clamped [1.0, 5000.0]",
            "side_slope_derivation": "m_1 / 2 from EIV_fits, clamped [0.5, 50.0]",
            "source_eiv": "SWOT River Volume (Cerbelaud et al., 2026)",
            "source_crosswalk": "MERIT-SWORD (Zenodo 14675925)",
        },
    )
    ds.to_netcdf(output_path)
    print(f"\nSaved {output_path} ({len(geometry)} COMIDs)")


def main() -> None:
    """Build SWOT geometry NetCDF by joining EIV fits to MERIT COMIDs."""
    args = parse_args()

    print("Step 1: Loading EIV fits...")
    eiv = load_eiv_fits(args.eiv_fits_dir, NA_PFAF_REGIONS, args.min_nobs)

    print("\nStep 2: Loading MERIT-SWORD crosswalk...")
    crosswalk = load_crosswalk(args.crosswalk_dir, NA_PFAF_REGIONS)

    print("\nStep 3: Deriving geometry...")
    geometry = derive_geometry(eiv, crosswalk)

    # Summary statistics
    print("\nSummary:")
    print(
        f"  top_width:  median={geometry['top_width'].median():.1f}m, "
        f"range=[{geometry['top_width'].min():.1f}, {geometry['top_width'].max():.1f}]"
    )
    print(
        f"  side_slope: median={geometry['side_slope'].median():.1f}, "
        f"range=[{geometry['side_slope'].min():.1f}, {geometry['side_slope'].max():.1f}]"
    )
    print(
        f"  nobs:       median={geometry['nobs'].median():.0f}, "
        f"range=[{geometry['nobs'].min()}, {geometry['nobs'].max()}]"
    )

    print("\nStep 4: Saving NetCDF...")
    save_netcdf(geometry, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
