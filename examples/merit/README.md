# MERIT-Hydro Global River Channel Geometry (DDR v0.5.2)

Predicted trapezoidal channel geometry for 2,939,404 reaches in the global Multi-Error-Removed-Improved-Terrain (MERIT) Hydro DEM based unit basins data set (Lin et al., 2021).  Geometry statistics (depth, width, etc.) are computed for 346,321 CONUS reaches using predicted accumulated streamflow from dHBV2.0UH (Song et al. 2025) over water year 2000 (1999-10-01 to 2000-09-30).

## Released files

- `merit_geometry_predictions.nc`: NetCDF dataset with predicted channel parameters and geometry statistics
- `ddr-v0.5.2-merit-geometry-weights.pt`: Trained KAN checkpoint (PyTorch) used to generate the predictions
- `merit_geometry_config.yaml`: Hydra configuration file for reproducing or modifying predictions
- `geometry_predictor.ipynb`: Notebook that explores the NetCDF output: distributions of learned KAN parameters, depth/width statistics, geometry-vs-discharge scatter plots, and Leopold & Maddock width-depth curves
- `plot_parameter_map.ipynb`: Notebook that joins predictions to MERIT-Hydro catchment polygons and produces spatial parameter maps (requires the MERIT shapefile)

## How the predictions were made

A KAN (Kolmogorov-Arnold Network is a neural network that replaces fixed activation functions with learnable spline-based functions on each edge) was trained end-to-end inside the [DDR](https://github.com/DeepGroundwater/ddr) (Distributed Differentiable Routing) framework.  During training the KAN learns to predict three physical river channel parameters from catchment attributes by back-propagating through differentiable Muskingum-Cunge routing (a physically-based flood routing method that approximates the 1D Saint-Venant equations) and comparing routed streamflow against USGS observations.

Once trained, the KAN predicts Manning's roughness (*n*), Leopold & Maddock width coefficient (*p*), and width-depth exponent (*q*) for every reach in the global MERIT-Hydro network using only the 10 catchment attributes listed below.  These parameters are discharge-independent, meaning they depend only on static landscape properties.

For the 346,321 CONUS reaches where streamflow data is available, the parameters are combined with daily accumulated discharge to derive trapezoidal cross-section geometry for each day of the water year.  The daily values are then summarized as min/max/median/mean statistics.

The generation script is [`scripts/geometry_predictor.py`](https://github.com/DeepGroundwater/ddr/blob/main/scripts/geometry_predictor.py) in the DDR repository.

## Streamflow source

Daily lateral inflows come from dHBV2.0UH (Song et al. 2025), a high-resolution distributed hydrologic model producing retrospective streamflow predictions across CONUS at the unit-catchment scale. The dHBV2.0 "unit-hydrograph" retrospective dataset provides lateral inflow (Q') for each MERIT unit catchment, covering 1980-2020 at daily resolution.

The dHBV2.0 lateral inflow data is freely available on Zenodo: <https://doi.org/10.5281/zenodo.15784945>

Lateral inflows are accumulated through the MERIT network topology by solving:

Non-CONUS reaches (2,593,083 of 2,939,404) lack dHBV2.0 coverage and have `NaN` for all geometry statistics and slope.  Their KAN parameters (*n*, *p*, *q*) are still valid and can be combined with any user-supplied discharge and slope.

## Spatial framework
The MERIT CONUS subset covers COMIDs 71000001 through 78028489 (346,321 reaches).

The NetCDF contains no geometries.  To map the data spatially, join on `COMID` to the MERIT-Hydro shapefiles available at <https://www.reachhydro.org/home/params/merit-basins>.

## KAN input attributes

The 10 catchment attributes used by the KAN, sourced from the MERIT global attributes file:

- `SoilGrids1km_clay` -- Soil clay content (%)
- `aridity` -- Aridity index, P/PET (--)
- `meanelevation` -- Mean catchment elevation (m)
- `meanP` -- Mean annual precipitation (mm/year)
- `NDVI` -- Normalized Difference Vegetation Index (--)
- `meanslope` -- Mean catchment slope (degrees)
- `log10_uparea` -- Log10 upstream drainage area (log10(km^2))
- `SoilGrids1km_sand` -- Soil sand content (%)
- `ETPOT_Hargr` -- Hargreaves mean annual potential evapotranspiration (mm/year)
- `Porosity` -- Soil porosity (%)

Attributes are z-score normalized using training statistics before input to the KAN.

## Output variables and units

### Learned parameters (all 2,939,404 reaches)

- `COMID` -- MERIT-Hydro reach identifier (int64 dimension/join key)
- `n` -- Manning's roughness coefficient (m^(-1/3) s, range [0.015, 0.25])
- `p_spatial` -- Leopold & Maddock width coefficient (range [1, 200], log-space)
- `q_spatial` -- Leopold & Maddock width-depth exponent (range [0, 1]; 0 = rectangular, 1 = triangular)

### Static reach property (346,321 CONUS reaches; NaN elsewhere)

- `slope` -- Channel bed slope (m/m, clamped min 0.001). Single value per reach, no temporal variants.

### Geometry statistics (346,321 CONUS reaches; NaN elsewhere)

Each variable below has four variants: `_min`, `_max`, `_median`, `_mean`, representing temporal statistics across the 365 days of water year 2000.

- `depth` -- Flow depth (m)
- `top_width` -- Water-surface width (m)
- `bottom_width` -- Channel bed width (m)
- `side_slope` -- Side slope, horizontal:vertical (H:V ratio, clamped [0.5, 50])
- `hydraulic_radius` -- Cross-sectional area / wetted perimeter (m)
- `discharge` -- Topologically accumulated streamflow (m^3/s)

28 data variables total: (3 KAN parameters + 1 slope + 6 geometry vars x 4 stats).

## How geometry is derived

Given *n*, *p*, *q*, discharge *Q*, and slope *S0*, the trapezoidal cross-section is computed by inverting Manning's equation for depth:

```
depth = [ Q * n * (q + 1) / (p * sqrt(S0)) ] ^ [ 3 / (5 + 3q) ]
```

Then the Leopold & Maddock power law gives top width:

```
top_width = p * depth^q
```

The side slope is derived by differentiating the power law with respect to depth -- the rate at which width increases with depth gives the horizontal run per unit vertical rise:

The cross-section is a trapezoid where the water surface (top_width) is wider than the bed (bottom_width), with sloped banks on each side.

- side_slope = top_width * q / (2 * depth) [H:V]
- bottom_width = top_width - 2 * side_slope * depth [m]
- area = (top_width + bottom_width) * depth / 2 [m^2]
- wetted_perimeter = bottom_width + 2 * depth * sqrt(1 + side_slope^2) [m]
- hydraulic_radius = area / wetted_perimeter [m]

> **Note on numerical stability:** The implementation adds small epsilon
> values (1e-6 to *q*, 1e-8 to the denominator) to avoid division by zero.

## Quick start

```python
import xarray as xr

ds = xr.open_dataset("merit_geometry_predictions.nc")

# All 2.9M reaches have KAN parameters
print(ds["n"])           # Manning's roughness
print(ds["p_spatial"])   # width coefficient
print(ds["q_spatial"])   # width-depth exponent

# 346K CONUS reaches have geometry statistics
conus = ds.where(ds["slope"].notnull(), drop=True)
print(conus["depth_mean"])
print(conus["top_width_mean"])
print(conus["discharge_mean"])
```

## Reproducing the predictions

The inputs used to reproduce this file will be relesed in an upcoming paper. To aid the community, we decided to attach the trained model weights, information on the attributes used, and the statistics file so anyone can bring their own attributes and run the NN.
