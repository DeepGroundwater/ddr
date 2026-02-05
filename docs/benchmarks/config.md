---
icon: lucide/settings
---

# Configuration Reference

The benchmark configuration extends DDR's standard Hydra configuration with additional options for model comparison.

## Full Configuration

```yaml
# benchmarks/config/benchmark.yaml

defaults:
  - _self_
  - hydra: settings

# === Standard DDR Configuration ===

mode: testing
geodataset: merit
name: benchmark-${now:%Y-%m-%d_%H-%M-%S}
device: cpu  # or cuda device number

data_sources:
  attributes: /path/to/attributes.nc
  geospatial_fabric_gpkg: /path/to/river_network.shp
  conus_adjacency: /path/to/conus_adjacency.zarr
  gages_adjacency: /path/to/gages_adjacency.zarr
  statistics: /path/to/statistics
  streamflow: /path/to/streamflow
  observations: /path/to/observations
  gages: /path/to/gages.csv

params:
  parameter_ranges:
    n: [0.02, 0.2]
    q_spatial: [0.0, 1.0]
    top_width: [1.0, 6000.0]
    side_slope: [0.5, 50.0]
  log_space_parameters: [top_width, side_slope]
  defaults:
    p_spatial: 21
  tau: 3

experiment:
  batch_size: 64
  start_time: 1995/10/01
  end_time: 2010/09/30
  warmup: 3
  checkpoint: /path/to/trained_model.pt  # Required for meaningful comparison

kan:
  hidden_size: 21
  input_var_names:
    - SoilGrids1km_clay
    - aridity
    - meanelevation
    - meanP
    - NDVI
    - meanslope
    - log10_uparea
    - SoilGrids1km_sand
    - ETPOT_Hargr
    - Porosity
  num_hidden_layers: 2
  learnable_parameters:
    - n
    - q_spatial
    - top_width
    - side_slope
  grid: 50
  k: 2

# === DiffRoute Configuration ===

diffroute:
  enabled: true
  irf_fn: muskingum
  max_delay: 100
  dt: 0.0416667
  k: 0.0416667
  x: 0.3
```

## DiffRoute Options

### `diffroute.enabled`

**Type**: `bool`
**Default**: `true`

Enable or disable DiffRoute comparison. Set to `false` to run DDR-only benchmarks (useful on CPU-only systems since DiffRoute requires CUDA).

### `diffroute.irf_fn`

**Type**: `str`
**Default**: `"muskingum"`

Impulse Response Function model to use. Options:

| IRF | Parameters | Description |
|-----|------------|-------------|
| `muskingum` | k, x | Classic Muskingum routing |
| `linear_storage` | tau | Single linear reservoir |
| `nash_cascade` | tau, n | Cascade of n linear reservoirs |
| `pure_lag` | delay | Pure time delay |
| `hayami` | D, L, c | Diffusive wave approximation |

### `diffroute.max_delay`

**Type**: `int`
**Default**: `100`

Maximum number of timesteps for the LTI router's impulse response. Larger values allow longer travel times but increase memory usage.

### `diffroute.dt`

**Type**: `float`
**Default**: `0.0416667` (1 hour in days)

Timestep size in **days**. Must match DDR's internal timestep (1 hour).

Common values:

| Timestep | dt (days) |
|----------|-----------|
| 15 min | 0.0104167 |
| 1 hour | 0.0416667 |
| 3 hours | 0.125 |
| 1 day | 1.0 |

### `diffroute.k`

**Type**: `float` or `null`
**Default**: `0.0416667` (1 hour in days)

Muskingum k parameter (wave travel time through reach) in **days**.

- Must be in same units as `dt`
- For stability: `k >= dt / (2*(1-x))`
- Physical interpretation: `k = reach_length / wave_celerity`

If `null`, defaults to the value of `dt`.

### `diffroute.x`

**Type**: `float`
**Default**: `0.3`

Muskingum x parameter (weighting factor).

- Range: 0.0 to 0.5
- `x = 0`: Pure reservoir (maximum attenuation)
- `x = 0.5`: Pure translation (no attenuation)
- Typical values: 0.1 - 0.3

## Command-Line Overrides

Override any configuration option from the command line:

```bash
# Change DiffRoute parameters
python -m ddr_benchmarks.benchmark diffroute.k=0.1 diffroute.x=0.2

# Change experiment settings
python -m ddr_benchmarks.benchmark \
    experiment.start_time=2000/10/01 \
    experiment.end_time=2005/09/30

# Use different checkpoint
python -m ddr_benchmarks.benchmark \
    experiment.checkpoint=/path/to/other_model.pt

# Run on GPU
python -m ddr_benchmarks.benchmark device=0

# Disable DiffRoute (DDR only)
python -m ddr_benchmarks.benchmark diffroute.enabled=false
```

## Output Directory

Results are saved to:

```
output/benchmark-<timestamp>/
├── plots/
│   ├── nse_cdf_comparison.png
│   ├── nse_boxplot_comparison.png
│   └── kge_cdf_comparison.png
├── benchmark_results.zarr
└── .hydra/
    ├── config.yaml
    └── overrides.yaml
```
