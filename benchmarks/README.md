# DDR Benchmarks

Benchmarking tools for comparing DDR against other routing models on identical input data.

## Features

- **DiffRoute Comparison**: Compare DDR against [DiffRoute](https://github.com/TristHas/DiffRoute) LTI routing
- **Summed Q' Baseline**: Optional non-routed baseline to quantify routing value
- **Standardized Evaluation**: Uses DDR's `Metrics` class (NSE, KGE, RMSE, Bias, FHV, FLV) for consistent comparison
- **Publication-Quality Plots**: Multi-metric boxplots, CDF curves, gauge maps, and per-gage hydrographs
- **Mass Balance Logging**: Accumulation comparison between models logged to console
- **Zarr Output**: Saves predictions from all models for further analysis

## Installation

```bash
# From repository root - full workspace
uv sync --all-packages

# Or with CUDA support for DiffRoute
uv sync --all-packages --extra cuda
```

## Quick Start

```bash
cd benchmarks

# Copy the example config and customize paths
cp config/example_benchmark.yaml config/benchmark.yaml

# Run benchmark
uv run python scripts/benchmark.py

# Override DiffRoute parameters
uv run python scripts/benchmark.py diffroute.k=0.1 diffroute.x=0.25

# Disable DiffRoute (DDR only, useful on CPU)
uv run python scripts/benchmark.py diffroute.enabled=false

# Include summed Q' baseline
uv run python scripts/benchmark.py summed_q_prime=/path/to/summed_q_prime.zarr
```

## Configuration

Edit `config/benchmark.yaml` or override via command line. See `config/example_benchmark.yaml` for a fully commented template.

```yaml
# DiffRoute-specific options
diffroute:
  enabled: true
  irf_fn: muskingum        # muskingum, linear_storage, nash_cascade, pure_lag, hayami
  max_delay: 100           # Maximum timesteps for impulse response
  dt: 0.0416667            # Timestep in days (1 hour)
  k: 0.1042                # Wave travel time in days (9000s = 2.5h, RAPID default)
  x: 0.3                   # Weighting factor (0-0.5)

# Optional baseline
summed_q_prime: null        # Path to summed Q' zarr store, or null to skip
```

### Muskingum k Parameter

The k parameter represents wave travel time through a reach:

- **Units**: Days (must match dt)
- **Stability**: k >= dt / (2*(1-x))
- **Physical meaning**: k = reach_length / wave_celerity

For dt = 1 hour (0.0417 days) and x = 0.3, k_min ~ 0.03 days. Default k = 0.1042 days (9000s, RAPID default).

## Output

Results are saved to `output/<name>/<timestamp>/`:

```
output/benchmarks-v0.1.0-merit/2026-02-06_12-00-00/
├── plots/
│   ├── nse_cdf_comparison.png            # CDF of NSE across all gauges
│   ├── kge_cdf_comparison.png            # CDF of KGE across all gauges
│   ├── metric_boxplot_comparison.png     # 6-panel boxplot (bias, rmse, fhv, flv, nse, kge)
│   ├── gauge_map_ddr_NSE.png            # Map of gauges colored by DDR NSE
│   ├── gauge_map_diffroute_NSE.png      # Map of gauges colored by DiffRoute NSE
│   ├── gauge_map_sqp_NSE.png            # Map of gauges colored by summed Q' NSE (if enabled)
│   └── hydrographs/                     # One hydrograph per gauge with all models overlaid
│       ├── 01234567.png
│       └── ...
├── benchmark_results.zarr                # xarray Dataset with predictions + observations
└── .hydra/
    └── config.yaml
```

### Console Output

The benchmark also logs a mass balance accumulation comparison:

```
=== Mass Balance Accumulation Comparison ===
DDR vs Obs       — Mean rel. error: 0.1234, Median: 0.0567
DiffRoute vs Obs — Mean rel. error: 0.2345, Median: 0.1234
DDR vs summed Q' — Mean rel. error: 0.0456, Median: 0.0234
```

## Package Structure

```
benchmarks/
├── scripts/
│   └── benchmark.py             # Entry point (Hydra CLI)
├── src/ddr_benchmarks/
│   ├── __init__.py              # Package exports
│   ├── benchmark.py             # Benchmark runner and plotting
│   ├── diffroute_adapter.py     # COO -> NetworkX conversion for DiffRoute
│   └── validation/
│       ├── __init__.py
│       ├── benchmark.py         # BenchmarkConfig validation
│       └── diffroute.py         # DiffRouteConfig
├── config/
│   ├── benchmark.yaml           # Active Hydra configuration
│   ├── example_benchmark.yaml   # Example config template (fully commented)
│   └── hydra/
│       └── settings.yaml
└── pyproject.toml
```

## Documentation

See the full documentation at the [DDR Benchmarks docs](https://deepgroundwater.com/ddr/benchmarks/).

## Requirements

- DDR core package
- DiffRoute (requires CUDA for GPU acceleration)
- Trained DDR model checkpoint for meaningful comparison
