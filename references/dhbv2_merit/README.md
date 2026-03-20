# dHBV2.0 MERIT Lateral Inflow

Notebooks for downloading dHBV2.0 retrospective runoff predictions and converting
them to the Icechunk format required by DDR.

## Prerequisites

### 1. MERIT Hydro catchment shapefile

The download notebook requires the MERIT Hydro level-7 catchment basins shapefile
to look up unit catchment areas for unit conversion (mm/day → m³/s).

**Download:** https://drive.google.com/drive/folders/1pRnMGvRL94cXi4JhVUruj9JbbVTEHGIZ

Expected file: `cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`

Place it (and its companion `.dbf`, `.prj`, `.shx` files) at:
```
data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp
```
or update the path in cell 3 of `download.ipynb`.

### 2. Install notebook dependencies

The notebooks use separate dependency pins from the main DDR workspace because
they need `zarr<3` (the source data is zarr v2) while DDR itself requires `zarr>=3`:

```bash
uv pip install "zarr<3" "s3fs" netcdf4
```

## Workflow

Run the notebooks in order:

### Step 1 — `download.ipynb`

Downloads dHBV2.0 runoff data from S3 (`s3://psu-diff-water-models/dhbv2.0_40yr_dataset/`),
converts mm/day → m³/s using MERIT catchment areas, and saves the result as a local
NetCDF file:

```
data/merit_dhbv2_UH_retrospective.nc
```

**Why NetCDF as an intermediate?** The source data is stored in zarr v2 format.
DDR uses zarr v3 (via Icechunk). There is no direct zarr v2 → v3 conversion path,
so the data is materialized to NetCDF first.

**MERIT zones covered:** `71, 72, 73, 74, 75, 77, 78`

These are Pfafstetter level-2 basins covering CONUS. Zone 76 is excluded
(Gulf Coast / interior drainage with no USGS gage coverage in the training set).

### Step 2 — `nc_to_icechunk.ipynb`

Reads the NetCDF output from Step 1 and writes it to a local Icechunk store:

```
data/merit_dhbv2_UH_retrospective/   ← Icechunk repo (directory)
```

Point your DDR config at this path:

```yaml
data_sources:
  streamflow: data/merit_dhbv2_UH_retrospective
```

## Output format

The resulting Icechunk store contains a single variable `Qr` with dimensions
`(divide_id, time)`, units `m^3/s`, at daily resolution from 1980-01-01 to 2020-12-31.
`divide_id` values are integer MERIT COMIDs.
