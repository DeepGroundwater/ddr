"""Build reservoir parameters CSV from HydroLAKES RFC-DA.

Reads the HydroLAKES RFC-DA CSV (with Pour_long/Pour_lat coordinates), constructs
point geometries, and spatially joins to MERIT catchment polygons (cat_pfaf_7) via
point-in-polygon in EPSG:5070 to obtain the MERIT COMID for each dam.  Outputs
level pool routing parameters indexed by MERIT COMID.

The input CSV is produced by the DeepGroundwater/references ``lakes`` package,
which derives RFC-DA hydraulic parameters (WeirE, OrificeE, WeirC, WeirL, OrificeC,
OrificeA) from HydroLAKES + GRanD attributes following NOAA-OWP NHF conventions.

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
from shapely.geometry import Point

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Input / output paths
RESERVOIR_CSV = Path("/projects/mhpi/tbindas/ddr/data/hydrolakes_rfc_da.csv")
MERIT_CATCHMENT_SHP = Path(
    "/projects/mhpi/data/MERIT/raw/continent/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
)
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "merit_reservoir_params.csv"

# Physical caps
MIN_WEIR_LENGTH = 1.0  # minimum weir length [m]
MAX_ORIFICE_AREA = 100.0  # physical cap [m²] — prevents forward Euler instability

# CRS for spatial operations (Albers Equal Area CONUS)
PROJECTED_CRS = "EPSG:5070"


def build_reservoir_params() -> pd.DataFrame:
    """Build reservoir parameters from HydroLAKES RFC-DA CSV.

    1. Load HydroLAKES RFC-DA CSV (6,797 reservoirs with RFC-DA attributes).
    2. Construct point geometries from Pour_long/Pour_lat.
    3. Load MERIT catchment polygons (cat_pfaf_7).
    4. Reproject both to EPSG:5070 and spatial-join (point-in-polygon).
    5. Map columns to output CSV schema.

    Returns
    -------
    pd.DataFrame
        Reservoir parameters indexed by MERIT COMID.
    """
    # Load reservoir CSV and construct point geometries
    log.info(f"Loading HydroLAKES RFC-DA from {RESERVOIR_CSV}...")
    df = pd.read_csv(RESERVOIR_CSV)
    geometry = [Point(lon, lat) for lon, lat in zip(df["Pour_long"], df["Pour_lat"], strict=False)]
    reservoirs = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    log.info(f"Reservoir records: {len(reservoirs)}")

    log.info(f"Loading MERIT catchment polygons from {MERIT_CATCHMENT_SHP}...")
    catchments = gpd.read_file(MERIT_CATCHMENT_SHP)
    if catchments.crs is None:
        catchments = catchments.set_crs("EPSG:4326")
    log.info(f"MERIT catchments: {len(catchments)}")

    # Reproject both to EPSG:5070 for accurate spatial join
    res_proj = reservoirs.to_crs(PROJECTED_CRS)
    cat_proj = catchments[["COMID", "geometry"]].to_crs(PROJECTED_CRS)

    # Point-in-polygon: reservoir dam points → MERIT catchment polygons
    log.info("Spatial join (EPSG:5070): reservoir points within MERIT catchments...")
    joined = gpd.sjoin(res_proj, cat_proj, how="inner", predicate="within")

    # Resolve MERIT COMID column name (sjoin appends _right on collision)
    if "COMID_right" in joined.columns:
        joined = joined.rename(columns={"COMID_right": "merit_comid"})
    elif "COMID" in joined.columns:
        joined = joined.rename(columns={"COMID": "merit_comid"})
    else:
        raise ValueError(f"Cannot find MERIT COMID column: {list(joined.columns)}")

    joined["merit_comid"] = joined["merit_comid"].astype(int)
    log.info(f"Matched: {len(joined)} dams → {joined['merit_comid'].nunique()} unique MERIT COMIDs")

    # Filter: drop rows with non-positive max pool elevation (bad data)
    valid_mask = joined["LkMxE"] > 0
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        log.warning(f"Filtered {n_invalid} dams with non-positive LkMxE")
        joined = joined[valid_mask]

    # Filter: drop rows with missing lake area (NaN LkArea → NaN in forward Euler)
    nan_area = joined["LkArea"].isna()
    if nan_area.any():
        log.warning(f"Filtered {nan_area.sum()} dams with NaN LkArea")
        joined = joined[~nan_area]

    # Filter: drop rows with inverted elevations (WeirE < OrificeE → pool clamp inversion)
    inverted = joined["WeirE"] < joined["OrificeE"]
    if inverted.any():
        log.warning(f"Filtered {inverted.sum()} dams with inverted elevations (WeirE < OrificeE)")
        joined = joined[~inverted]

    # Deduplicate: keep largest lake area per MERIT catchment
    joined = joined.sort_values("LkArea", ascending=False).drop_duplicates(subset="merit_comid", keep="first")
    joined = joined.set_index("merit_comid")
    joined.index.name = "COMID"

    # Map columns → output CSV schema (LkArea is already in m²)
    params = pd.DataFrame(
        {
            "lake_area_m2": joined["LkArea"].values,
            "weir_elevation": joined["WeirE"].values,
            "orifice_elevation": joined["OrificeE"].values,
            "weir_coeff": joined["WeirC"].values,
            "weir_length": np.clip(joined["WeirL"].values, MIN_WEIR_LENGTH, None),
            "orifice_coeff": joined["OrificeC"].values,
            "orifice_area": np.clip(joined["OrificeA"].values, 0, MAX_ORIFICE_AREA),
            "initial_pool_elevation": (joined["OrificeE"].values + joined["WeirE"].values) / 2,
        },
        index=joined.index,
    )

    log.info(f"Output reservoirs: {len(params)}")
    return params


def main() -> None:
    """Build reservoir parameters CSV."""
    params = build_reservoir_params()

    # Sanity checks
    log.info(f"Output shape: {params.shape}")
    log.info(
        f"Lake area [m²]: median={params['lake_area_m2'].median():.0f}, "
        f"range=[{params['lake_area_m2'].min():.0f}, {params['lake_area_m2'].max():.0f}]"
    )
    log.info(
        f"Orifice coeff: median={params['orifice_coeff'].median():.3f}, "
        f"non-default={(params['orifice_coeff'] != 0.1).sum()}"
    )
    log.info(
        f"Orifice area [m²]: median={params['orifice_area'].median():.2f}, "
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
