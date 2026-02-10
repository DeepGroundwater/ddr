# References

Reference datasets and scripts for mapping USGS streamflow gages to river network reaches. These gage-to-COMID assignments determine which gages are eligible for DDR training and how lateral inflow is scaled at each gage location.

## Directory Structure

```
references/
├── gage_info/          Gage reference CSVs (GAGES-II, CAMELS, gages_3000, dhbv2)
├── geo_io/             Scripts to build and patch gage reference CSVs
├── analysis/           Drainage area error distribution plots
└── dhbv2_merit/        Notebooks for downloading/converting dHBV2.0 lateral inflow
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

## Supported Geospatial Datasets

- **NOAA-OWP Hydrofabric v2.2** — Lynker-based river network
- **MERIT Hydro Vectorized Flowlines** — MERIT-based river network (primary for national-scale DDR)
