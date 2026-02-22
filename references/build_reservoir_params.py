"""Build reservoir parameters CSV from HydroLAKES-MERIT intersection shapefiles.

Reads pre-intersected shapefiles from the MERIT-HydroLAKES data, aggregates
lake parameters per COMID (many-to-many: multiple lakes per COMID), and derives
level pool routing parameters (weir + orifice outflow model).

Usage
-----
    uv run python references/build_reservoir_params.py

Output
------
    data/merit_reservoir_params.csv
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Input / output paths
SHAPEFILE_DIR = Path("/projects/mhpi/data/hydroLakes/merit_intersected_data")
SHAPEFILE_IDS = range(71, 79)  # RIV_lake_intersection_71..78
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "merit_reservoir_params.csv"

# Physical constants
G = 9.81  # gravitational acceleration [m/s^2]
C_W_DEFAULT = 0.4  # broad-crested weir discharge coefficient
C_O_DEFAULT = 0.1  # orifice discharge coefficient (NWM/RFC-DA conservative default)
SHORE_FRAC = 0.01  # fraction of shoreline used as weir length
MIN_WEIR_LENGTH = 1.0  # minimum weir length [m]

# RFC-DA elevation fractions (nhf-builds convention)
CREST_FRAC = 0.90  # weir crest at 90% of pool height from base
INVERT_FRAC = 0.15  # orifice invert at 15% of pool height from base


def load_all_shapefiles() -> gpd.GeoDataFrame:
    """Load and concatenate all intersection shapefiles."""
    frames = []
    for sid in SHAPEFILE_IDS:
        path = SHAPEFILE_DIR / f"RIV_lake_intersection_{sid}.shp"
        if not path.exists():
            log.warning(f"Shapefile not found: {path}")
            continue
        gdf = gpd.read_file(path)
        log.info(f"Loaded {len(gdf)} records from {path.name}")
        frames.append(gdf)

    if not frames:
        raise FileNotFoundError(f"No shapefiles found in {SHAPEFILE_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Total records: {len(combined)}")
    return combined


def aggregate_per_comid(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Aggregate lake parameters per COMID.

    Multiple lakes can intersect a single COMID. We aggregate:
    - lake_area_m2: sum of Lake_area (km^2 -> m^2)
    - depth_avg_m: area-weighted mean of Depth_avg
    - elevation_m: area-weighted mean of Elevation
    - dis_avg_m3s: sum of Dis_avg
    - shore_len_m: sum of Shore_len (km -> m)
    """
    # Drop rows with missing COMID or Lake_area
    valid = gdf.dropna(subset=["COMID", "Lake_area"]).copy()
    valid["COMID"] = valid["COMID"].astype(int)

    # Area weights for weighted averages
    valid["_weight"] = valid["Lake_area"]  # km^2

    grouped = valid.groupby("COMID")

    result = pd.DataFrame(
        {
            "lake_area_m2": grouped["Lake_area"].sum() * 1e6,  # km^2 -> m^2
            "depth_avg_m": grouped.apply(
                lambda g: np.average(g["Depth_avg"], weights=g["_weight"]) if g["_weight"].sum() > 0 else 0.0,
                include_groups=False,
            ),
            "elevation_m": grouped.apply(
                lambda g: np.average(g["Elevation"], weights=g["_weight"]) if g["_weight"].sum() > 0 else 0.0,
                include_groups=False,
            ),
            "dis_avg_m3s": grouped["Dis_avg"].sum(),
            "shore_len_m": grouped["Shore_len"].sum() * 1000,  # km -> m
        }
    )

    result.index.name = "COMID"
    return result


def derive_reservoir_params(agg: pd.DataFrame) -> pd.DataFrame:
    """Derive level pool reservoir parameters from aggregated lake data.

    Uses RFC-DA elevation conventions (nhf-builds / Lynker hydrofabric):
    - Orifice invert at 15% of pool height from base (dead storage below)
    - Weir crest at 90% of pool height from base
    - C_o = 0.1 (NWM conservative default)

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated lake parameters indexed by COMID.

    Returns
    -------
    pd.DataFrame
        Reservoir parameters indexed by COMID.
    """
    depth = agg["depth_avg_m"].clip(lower=0.1)
    elev = agg["elevation_m"]
    dis_avg = agg["dis_avg_m3s"].clip(lower=1e-6)
    base = elev - depth  # lake bottom

    # RFC-DA elevation conventions
    weir_elevation = base + CREST_FRAC * depth  # 90% of pool height
    orifice_elevation = base + INVERT_FRAC * depth  # 15% of pool height (dead storage below)

    weir_length = (agg["shore_len_m"] * SHORE_FRAC).clip(lower=MIN_WEIR_LENGTH)

    # Back-calculate orifice area from average discharge at half-depth equilibrium
    # Head above orifice at half-depth pool: (0.5 - INVERT_FRAC) * depth = 0.35 * depth
    h_mid = (0.5 - INVERT_FRAC) * depth
    orifice_area = dis_avg / (C_O_DEFAULT * np.sqrt(2 * G * h_mid) + 1e-8)

    # Initial pool elevation at half-full
    initial_pool_elevation = elev - 0.5 * depth

    params = pd.DataFrame(
        {
            "lake_area_m2": agg["lake_area_m2"],
            "weir_elevation": weir_elevation,
            "orifice_elevation": orifice_elevation,
            "weir_coeff": C_W_DEFAULT,
            "weir_length": weir_length,
            "orifice_coeff": C_O_DEFAULT,
            "orifice_area": orifice_area,
            "initial_pool_elevation": initial_pool_elevation,
        },
        index=agg.index,
    )
    return params


def main() -> None:
    """Build reservoir parameters CSV."""
    log.info("Loading HydroLAKES-MERIT intersection shapefiles...")
    gdf = load_all_shapefiles()

    log.info("Aggregating per COMID...")
    agg = aggregate_per_comid(gdf)
    log.info(f"Unique COMIDs with lakes: {len(agg)}")

    # Filter out physically invalid entries (HydroLAKES Elevation=0 artifacts)
    valid_mask = agg["elevation_m"] > 0
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        log.warning(f"Filtered {n_invalid} COMIDs with non-positive elevation")
        agg = agg[valid_mask]
        log.info(f"Remaining COMIDs: {len(agg)}")

    log.info("Deriving level pool parameters...")
    params = derive_reservoir_params(agg)

    # Sanity checks
    log.info(f"Output shape: {params.shape}")
    log.info(
        f"Lake area [m^2]: median={params['lake_area_m2'].median():.0f}, "
        f"range=[{params['lake_area_m2'].min():.0f}, {params['lake_area_m2'].max():.0f}]"
    )
    log.info(
        f"Orifice area [m^2]: median={params['orifice_area'].median():.2f}, "
        f"range=[{params['orifice_area'].min():.2f}, {params['orifice_area'].max():.2f}]"
    )
    log.info(
        f"Weir length [m]: median={params['weir_length'].median():.1f}, "
        f"range=[{params['weir_length'].min():.1f}, {params['weir_length'].max():.1f}]"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    params.to_csv(OUTPUT_PATH)
    log.info(f"Saved reservoir parameters to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
