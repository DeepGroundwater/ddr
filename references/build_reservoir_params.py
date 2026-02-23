"""Build reservoir parameters CSV from RFC-DA Hydraulics v2.

Spatially joins RFC-DA dam points to MERIT catchment polygons (cat_pfaf_7) via
point-in-polygon in EPSG:5070 to obtain the MERIT COMID for each dam.  Outputs
level pool routing parameters directly from per-dam RFC-DA attributes.

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
RFC_DA_GPKG = Path(
    "/projects/mhpi/tbindas/hydrofabric-builds/data/reservoirs/output/rfc-da-hydraulics-v2.gpkg"
)
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
    """Build reservoir parameters from RFC-DA Hydraulics v2.

    1. Load RFC-DA GeoPackage (8,571 dams with per-dam hydraulic attributes).
    2. Load MERIT catchment polygons (cat_pfaf_7).
    3. Reproject both to EPSG:5070 and spatial-join (point-in-polygon).
    4. Map RFC-DA columns to output CSV schema.

    Returns
    -------
    pd.DataFrame
        Reservoir parameters indexed by MERIT COMID.
    """
    # Load data sources
    log.info(f"Loading RFC-DA Hydraulics v2 from {RFC_DA_GPKG}...")
    rfc = gpd.read_file(RFC_DA_GPKG)
    log.info(f"RFC-DA records: {len(rfc)}")

    log.info(f"Loading MERIT catchment polygons from {MERIT_CATCHMENT_SHP}...")
    catchments = gpd.read_file(MERIT_CATCHMENT_SHP)
    if catchments.crs is None:
        catchments = catchments.set_crs("EPSG:4326")
    log.info(f"MERIT catchments: {len(catchments)}")

    # Reproject both to EPSG:5070 for accurate spatial join
    rfc_proj = rfc.to_crs(PROJECTED_CRS)
    cat_proj = catchments[["COMID", "geometry"]].to_crs(PROJECTED_CRS)

    # Point-in-polygon: RFC-DA dam points → MERIT catchment polygons
    log.info("Spatial join (EPSG:5070): RFC-DA dam points within MERIT catchments...")
    joined = gpd.sjoin(rfc_proj, cat_proj, how="inner", predicate="within")

    # Resolve MERIT COMID column name (sjoin appends _right on collision)
    if "COMID_right" in joined.columns:
        joined = joined.rename(columns={"COMID_right": "merit_comid"})
    elif "COMID" in joined.columns and "comid" in joined.columns:
        joined = joined.rename(columns={"COMID": "merit_comid"})
    else:
        raise ValueError(f"Cannot find MERIT COMID column in joined result: {list(joined.columns)}")

    joined["merit_comid"] = joined["merit_comid"].astype(int)
    log.info(f"Matched: {len(joined)} RFC-DA dams → {joined['merit_comid'].nunique()} unique MERIT COMIDs")

    # Filter: drop rows with non-positive max pool elevation (bad data)
    valid_mask = joined["LkMxE"] > 0
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        log.warning(f"Filtered {n_invalid} dams with non-positive LkMxE")
        joined = joined[valid_mask]

    # Deduplicate: keep largest lake area per MERIT catchment
    joined = joined.sort_values("LkArea", ascending=False).drop_duplicates(subset="merit_comid", keep="first")
    joined = joined.set_index("merit_comid")
    joined.index.name = "COMID"

    # Map RFC-DA columns → output CSV schema
    params = pd.DataFrame(
        {
            "lake_area_m2": joined["LkArea"].values,
            "weir_elevation": joined["WeirE"].values,
            "orifice_elevation": joined["OrficeE"].values,
            "weir_coeff": joined["WeirC"].values,
            "weir_length": np.clip(joined["WeirL"].values, MIN_WEIR_LENGTH, None),
            "orifice_coeff": joined["OrficeC"].values,
            "orifice_area": np.clip(joined["OrficeA"].values, 0, MAX_ORIFICE_AREA),
            "initial_pool_elevation": (joined["OrficeE"].values + joined["WeirE"].values) / 2,
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
