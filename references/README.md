# References

Reference datasets and preprocessing scripts for building DDR input data. Includes gage-to-COMID mapping (which gages are eligible for training and how lateral inflow is scaled), and reservoir parameter derivation from HydroLAKES.

## Directory Structure

```
references/
├── gage_info/                   Gage reference CSVs (GAGES-II, CAMELS, gages_3000, dhbv2)
├── geo_io/                      Scripts to build and patch gage reference CSVs
├── analysis/                    Drainage area error distribution plots
├── dhbv2_merit/                 Notebooks for downloading/converting dHBV2.0 lateral inflow
└── build_reservoir_params.py    Derives level pool parameters from HydroLAKES-MERIT intersection
```

## `gage_info/`

CSV files mapping USGS stream gages to MERIT Hydro COMIDs. Each row is a gage with its assigned river reach, drainage area comparison metrics, and a flow scale factor.

See [`gage_info/README.md`](gage_info/README.md) for column definitions and dataset citations.

**Key columns used at runtime:**
- `ABS_DIFF` — filtered by `max_area_diff_sqkm` (default 50 km²) to exclude poorly matched gages
- `DRAIN_SQKM`, `COMID_DRAIN_SQKM`, `COMID_UNITAREA_SQKM` — used to compute flow scaling per batch

## `geo_io/`

| Script | Purpose |
|--------|---------|
| `build_gage_references.py` | Spatial join of GAGES-II points to MERIT catchments → writes `GAGES-II.csv`, `camels_670.csv`, `gages_3000.csv` |
| `patch_dhbv2_gages.py` | Patches `dhbv2_gages.csv` with standard columns (`ABS_DIFF`, `DA_VALID`, `FLOW_SCALE`, etc.) |

## `analysis/`

| File | Purpose |
|------|---------|
| `plot_gage_distributions.py` | Generates `ABS_DIFF_distributions.png` — per-dataset histograms + CDF overlay |

## `dhbv2_merit/`

Jupyter notebooks for downloading dHBV2.0 unit-hydrograph retrospective lateral inflow data and converting it from NetCDF to Icechunk format.

## `build_reservoir_params.py`

Preprocessing script that derives level pool reservoir routing parameters from HydroLAKES-MERIT intersection shapefiles.

**Input:** 8 shapefiles from `/projects/mhpi/data/hydroLakes/merit_intersected_data/RIV_lake_intersection_{71-78}.shp` (80,597 records)

**Output:** `data/merit_reservoir_params.csv` — 46,095 COMIDs with 8 columns:

| Column | Units | Description |
|--------|-------|-------------|
| `lake_area_m2` | m^2 | Aggregated lake surface area |
| `weir_elevation` | m | Weir crest elevation (75% of full pool) |
| `orifice_elevation` | m | Orifice center elevation (lake bottom) |
| `weir_coeff` | - | Broad-crested weir coefficient (0.4) |
| `weir_length` | m | Effective weir length (1% of shoreline) |
| `orifice_coeff` | - | Orifice discharge coefficient (0.6) |
| `orifice_area` | m^2 | Back-calculated from average discharge |
| `initial_pool_elevation` | m | Initial pool elevation (half-full) |

**Usage:**
```bash
uv run python references/build_reservoir_params.py
```

See [`docs/reservoir.md`](../docs/reservoir.md) for full physics documentation.

## Supported Geospatial Datasets

- **NOAA-OWP Hydrofabric v2.2** — Lynker-based river network
- **MERIT Hydro Vectorized Flowlines** — MERIT-based river network (primary for national-scale DDR)
