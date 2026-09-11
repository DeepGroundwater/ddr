---
icon: lucide/grid-3x3
---

# Gridded (ISIMIP DDM30) inputs

How to obtain every input the gridded routing path needs, from nothing. The
[Juniata gridded example](https://github.com/DeepGroundwater/ddr/tree/master/examples/juniata_gridded)
ships its own 1.6 MB bundle and needs none of this; follow this page to rebuild the
CONUS or global stores, or to train on your own gauges.

Sources follow Song et al. (2025, WRR, doi:10.1029/2024WR038928), whose attribute
tables define the KAN inputs. **All gridded attributes are aggregated from native
rasters onto 0.5° cells; basin-averaged values are never resampled onto the grid.**

## Prerequisites

    uv sync --all-packages          # installs extractrs, geedim, pyogrio, rioxarray

A Google Earth Engine account is needed for most rasters. Register a free cloud
project at [code.earthengine.google.com/register](https://code.earthengine.google.com/register),
then authenticate once:

    uv run --with earthengine-api earthengine authenticate

Everything below writes under a data root you choose; the scripts default to
`/mnt/ssd1/data`. Total footprint is about 35 GB for CONUS, 55 GB for global.

## 1. The routing network

The ISIMIP3b river-routing product is DDM30 (Döll & Lehner 2002): a global 0.5°
D8 drainage-direction grid, 67,424 land cells.

    mkdir -p data/ddm30 && cd data/ddm30
    curl -O https://files.isimip.org/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_flowdir_cru_neva.nc
    curl -O https://files.isimip.org/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_basins_cru_neva.nc
    curl -O https://files.isimip.org/ISIMIP3b/InputData/geo_conditions/river_routing/ddm30_slopes_cru_neva.nc

The search portal at `data.isimip.org` is often unreachable; the file server above
works. Then build the adjacency and any per-gauge subsets:

    uv run python -m ddr_engine.gridded data/ddm30 --path data/ddm30 --gages data/ddm30/juniata_gages.csv

This writes `ddm30_adjacency.zarr` (67,424 cells, 55,866 edges) and a per-gauge
`ddm30_gages_adjacency.zarr`. Cell ids are `row * 720 + col`, row 0 at 55.75°S.

!!! note "netCDF deviations from the ISIMIP documentation"
    The documented flow-direction code 9 never appears. The 1,326 CRU-only cells carry
    `flowdir = 0` with basin ≥ 70,001 and slope 0.001, and 782 cells carry `flowdir = −1`
    for inland terminals. Treat any code outside 1–8 as terminal. Lake Ladoga is basin
    17102, not the documented 17012.

### Sub-reach network (optional, recommended)

Splits each cell so Muskingum-Cunge runs near Courant 1:

    uv run python scripts/build_subdivided_adjacency.py \
        --qprime /mnt/ssd1/data/icechunk/ddm30_conus_uh_retrospective_regridded.ic \
        --sizing-flow mean --out data/ddm30/ddm30_subreach_adjacency.zarr

Needs the forcing store from step 3, since reach length is set from celerity at mean
flow. Use `--global` for the whole grid, or `--target-m 6000` for a fixed length.

## 2. Attributes

### Earth Engine rasters

One script pulls GMTED2010 elevation, six HiHydroSoil layers, GLWD open water, a
MODIS NDVI climatology and a MODIS snow-cover climatology:

    uv run scripts/acquire_gee_attributes.py --project <your-gee-project>          # CONUS, ~3 GB
    uv run scripts/acquire_gee_attributes.py --project <your-gee-project> --global # ~4 GB

It is resumable, so rerun it if a layer fails. Do **not** run two copies at once:
parallel jobs trip Earth Engine's rate limit and layers fail with HTTP 429.

### Direct downloads

| Source | Command | Size |
|---|---|---|
| SoilGrids 2.0 clay/sand/silt, 0–5 cm, 1 km | `curl -O https://files.isric.org/soilgrids/latest/data_aggregated/1000m/{clay,sand,silt}/{clay,sand,silt}_0-5cm_mean_1000.tif` | 560 MB |
| WorldClim 2.1 30″ bio, tmin, tmax, prec | `curl -O https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_30s_{bio,tmin,tmax,prec}.zip` | 20 GB |
| GLHYMPS 1.0 (porosity + permeability) | `curl -L -o GLHYMPS.zip "https://borealisdata.ca/api/access/datafile/72026"` | 1.1 GB |
| MERIT unit catchments (for the forcing) | [reachhydro.org](https://www.reachhydro.org/home/params/merit-basins), `cat_pfaf_7` | 2 GB |

Unzip the WorldClim archives once, into `worldclim/tif/`. Reading windows straight
from the zips is roughly 20× slower because each read decompresses from the start of a
deflate block:

    cd worldclim && mkdir -p tif && for v in bio tmin tmax prec; do unzip -j -o wc2.1_30s_$v.zip -d tif; done

Unzip GLHYMPS to `glhymps/extracted/`. SoilGrids is in Goode Homolosine and GLHYMPS in
World Cylindrical Equal Area; both are reprojected automatically.

### Build the store

    uv run python scripts/build_gridded_attributes.py            # CONUS,  ~10 min
    uv run python scripts/build_gridded_attributes.py --global   # global, ~1 h

Aggregation is coverage-weighted zonal means via [extractrs](https://github.com/taddyb/extractrs),
processed in 30° blocks so a world raster is never held in memory. Three outputs:

| File | Dimension | Reader |
|---|---|---|
| `ddm30_*_attributes.nc` | `COMID` = cell id | ddr's MERIT attribute path, and ddrs |
| `ddm30_*_attributes.ic` | `divide_id` = cell id | ddr's icechunk attribute path |
| `ddm30_*_attributes_grid.nc` | `(lat, lon)` 280×720 | ISIMIP-style tooling and maps |

25 variables: elevation, slope, three SoilGrids textures, six HiHydroSoil properties,
NDVI, snow fraction, snowfall fraction, open-water fraction, porosity, permeability,
mean precipitation and temperature, potential evapotranspiration, aridity, two
seasonality indices, cell area and upstream area.

## 3. Lateral inflow (Q′)

Any per-catchment runoff works. For the dHBV2 UH retrospective:

    uv run python scripts/build_gridded_qprime.py

!!! warning "Write `Qr(divide_id, time)`, and read by dimension name"
    The store follows the source contract, `Qr(divide_id, time)`. A store written
    transposed is still *readable*: an out-of-range read returns fill values instead of
    raising, so a mismatched reader silently yields an all-NaN lateral inflow and a
    baseline of zero rather than an error. Readers here call
    `.transpose("time", "divide_id")`, which resolves by name and is correct either way.
    Do not index axes positionally.

Q′ is volumetric per catchment (m³/s), so moving it onto cells is mass-conserving
aggregation: each catchment's discharge is split across the cells it overlaps in
proportion to overlapping area. About 40% of catchments straddle a cell edge, so
assigning a whole catchment to the cell holding its flowline midpoint misplaces water.
Verified to 8e-8 relative on volume. Catchment-to-cell weights cache to parquet, so the
polygon overlay runs once.

## 4. Observations and gauges

Two inputs: USGS daily observations in an icechunk store with `streamflow(gage_id, time)`,
and a gauge CSV in the `gages_3000.csv` schema.

### Selection

`ddr_benchmarks.gridded.snap_gauges` does the matching, and `train_conus.py` applies the
filters. The chain, with the counts from the shipped runs:

| Step | Gauges | Control |
|---|---|---|
| In `gages_3000.csv` | 3,211 | — |
| Drainage area > 2,000 km² | 890 | `--min-da-km2` |
| Snap within area tolerance | 667 | `DA_TOLERANCE` in `ddr_benchmarks.gridded` |
| Observation coverage > 90% | 620 | `min_cov` in `build_network`, currently fixed at 0.9 |

Snapping searches the 3×3 cell neighbourhood of the gauge coordinate and keeps the cell
that minimises `|log(cell_upstream_area / gauge_drainage_area)|`, where the cell's upstream
area is the topological accumulation of cell areas over the DDM30 network. Candidates
outside a ratio of [0.7, 1.4] are dropped, and when two gauges snap to the same cell the
one with the smaller log error wins. Median accepted ratio is 1.01.

!!! warning "A 0.5° cell is about 2,350 km²"
    A basin smaller than one routing element cannot be represented. 2,416 of the 3,211
    gauges are sub-cell. This is a property of the grid, not a tunable threshold: lowering
    `--min-da-km2` admits gauges whose drainage-area match is noise. Above 1,000 km² only
    730 of 1,354 candidates snap within tolerance, against 667 of 890 above 2,000 km².

!!! note "Why not nearest-centre snapping"
    It was tried first and misplaced about half of the large gauges, with drainage-area
    errors up to a factor of 201. A gauge near a cell edge is frequently closest to the
    centre of a cell its river never enters, so proximity alone is not evidence of
    hydrologic connection. Matching on accumulated area tests the connection directly.

Residual area mismatch is handled by scaling predictions by the inverse of the ratio,
applied identically in training and evaluation so the parameterization is never asked to
absorb it.

## 5. Validate

    uv run python scripts/validate_gridded.py --store /mnt/ssd1/data/icechunk/ddm30_conus_attributes.nc

67 checks across four families: flow-direction topology (single downstream, strictly
lower-triangular order, neighbours only, no cross-basin edges, area conserved),
documented unit ranges per attribute, cross-variable identities (texture sums, field
capacity below saturation, aridity = PET/P), and Muskingum-Cunge stability including a
synthetic flood routed through the real network. Exits non-zero on failure.

## Known gotchas

- Every geedim GeoTIFF carries a trailing `FILL_MASK` band; band 1 is the data.
- The Earth Engine `dir` band is uint8, so inland sinks (−1) arrive as 255 and river
  mouths (0) share a value with ocean nodata. Treat both as trace terminals.
- HiHydroSoil values are stored as integers ×10,000, and `WCsat` has an unflagged
  int32-max overflow pixel; the builder bounds them before aggregating.
- SoilGrids fills 0 over permanent water where HiHydroSoil returns nodata; all-zero
  texture is set to NaN.
- Hyper-arid cells round to 0 mm/yr precipitation, so aridity floors P at 1 mm/yr.
- MERIT Hydro stops at 60°N, which excludes 17,238 cells of the global grid. Elevation
  and slope therefore come from GMTED2010.
