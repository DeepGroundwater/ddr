# Gage Info

This folder contains CSV files mapping USGS streamflow gages to MERIT Hydro COMIDs for use in DDR benchmarking.

## Files

### `camels_670.csv`

670 gages from the CAMELS / HCDN-2009 reference basin set. Derived by:

1. Filtering `data/conus_3000_gages.gpkg` to the 671 HCDN-2009 gage IDs found in `data/HCDN_nhru_final_671.shp`.
2. Performing a spatial intersection (point-in-polygon) with MERIT Hydro basins (`data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`) in EPSG:5070 to assign each gage a MERIT `COMID`.
3. Dropping 1 coastal gage (`01195100`, Indian River near Clinton, CT) that falls outside MERIT basin coverage.

**Citation:**
> A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett, 2014.
> A large-sample watershed-scale hydrometeorological dataset for the contiguous USA.
> Boulder, CO: UCAR/NCAR. https://dx.doi.org/10.5065/D6MW2F4D

### `gages_3000.csv`

3211 gages curated by Ouyang et al. (2021) for continental-scale streamflow modeling. Derived by:

1. Matching the 3213 gage IDs in `data/gages3000Info.csv` to `data/conus_3000_gages.gpkg` (zero-filled to 8 characters).
2. Performing a spatial intersection (point-in-polygon) with MERIT Hydro basins in EPSG:5070 to assign each gage a MERIT `COMID`.
3. Dropping 2 coastal gages (`01305500`, Swan River at East Patchogue NY; `01195100`, Indian River near Clinton, CT) that fall outside MERIT basin coverage.

**Citation:**
> Ouyang, W., Lawson, K., Feng, D., Ye, L., Zhang, C., & Shen, C. (2021).
> Continental-scale streamflow modeling of basins with reservoirs: Towards a coherent
> deep-learning-based strategy. Journal of Hydrology, 599, 126455.
> https://doi.org/10.1016/j.jhydrol.2021.126455

### `GAGES-II.csv`

8945 gages from the GAGES-II dataset (Falcone, 2011), which provides geospatial data and classifications for 9,322 stream gages maintained by the USGS. Derived by:

1. Performing a spatial intersection (point-in-polygon) with MERIT Hydro basins (`data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp`) in EPSG:5070 to assign each gage a MERIT `COMID`.
2. Dropping 19 coastal gages that fall outside MERIT basin coverage.
3. Dropping 103 gages with non-standard station IDs (9–10 digits) that have no matching USGS streamflow observations. These are likely non-reference gages or sub-stations without available observation records.

**Citation:**
> Falcone, J. A. GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow, 2011.
> https://doi.org/10.3133/70046617

### Columns

| Column       | Description                          |
|--------------|--------------------------------------|
| `STAID`      | USGS station ID                      |
| `STANAME`    | Station name                         |
| `DRAIN_SQKM` | Drainage area (km²)                 |
| `LAT_GAGE`   | Gage latitude (NAD83)               |
| `LNG_GAGE`   | Gage longitude (NAD83)              |
| `COMID`      | MERIT Hydro basin COMID             |

## Data Sources

- **CAMELS** (Catchment Attributes and Meteorology for Large-sample Studies):
  A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett, 2014.
  A large-sample watershed-scale hydrometeorological dataset for the contiguous USA.
  Boulder, CO: UCAR/NCAR. <https://dx.doi.org/10.5065/D6MW2F4D>

- **GAGES-II** (Geospatial Attributes of Gages for Evaluating Streamflow):
  Falcone, J. A. GAGES-II: Geospatial Attributes of Gages for Evaluating Streamflow, 2011.
  <https://doi.org/10.3133/70046617>

- **Ouyang et al. (2021)**:
  Ouyang, W., Lawson, K., Feng, D., Ye, L., Zhang, C., & Shen, C. (2021).
  Continental-scale streamflow modeling of basins with reservoirs: Towards a coherent
  deep-learning-based strategy. Journal of Hydrology, 599, 126455.
  <https://doi.org/10.1016/j.jhydrol.2021.126455>

- **MERIT Hydro**:
  Lin, P., Pan, M., Wood, E. F., Yamazaki, D., & Allen, G. H. (2021).
  A new vector-based global river network dataset accounting for variable drainage density.
  Scientific Data, 8(1), 28.
  <https://doi.org/10.1038/s41597-021-00819-9>
