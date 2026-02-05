# DDR Benchmarks

Benchmarking tools for comparing DDR against other routing models on identical input data.

## Features

- **DiffRoute Comparison**: Compare DDR against [DiffRoute](https://github.com/TristHas/DiffRoute) LTI routing
- **Standardized Evaluation**: Uses DDR's `Metrics` class (NSE, KGE, RMSE) for consistent comparison
- **Comparison Plots**: Generates CDF and boxplot comparisons of metric distributions
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

# Run benchmark with default configuration
python -m ddr_benchmarks.benchmark

# Override DiffRoute parameters
python -m ddr_benchmarks.benchmark diffroute.k=0.1 diffroute.x=0.25

# Disable DiffRoute (DDR only, useful on CPU)
python -m ddr_benchmarks.benchmark diffroute.enabled=false
```

## Configuration

Edit `config/benchmark.yaml` or override via command line:

```yaml
# DiffRoute-specific options
diffroute:
  enabled: true
  irf_fn: muskingum        # muskingum, linear_storage, nash_cascade, pure_lag, hayami
  max_delay: 100           # Maximum timesteps for impulse response
  dt: 0.0416667            # Timestep in days (1 hour)
  k: 0.0416667             # Wave travel time in days (same units as dt)
  x: 0.3                   # Weighting factor (0-0.5)
```

### Muskingum k Parameter

The k parameter represents wave travel time through a reach:

- **Units**: Days (must match dt)
- **Stability**: k >= dt / (2*(1-x))
- **Physical meaning**: k = reach_length / wave_celerity

For dt = 1 hour (0.0417 days) and x = 0.3, k_min ≈ 0.03 days.

## Output

Results are saved to `output/benchmark-<timestamp>/`:

```
output/benchmark-2026-02-05_12-00-00/
├── plots/
│   ├── nse_cdf_comparison.png
│   ├── nse_boxplot_comparison.png
│   └── kge_cdf_comparison.png
├── benchmark_results.zarr
└── .hydra/
    └── config.yaml
```

## Documentation

See the full documentation at the [DDR Benchmarks docs](https://deepgroundwater.com/ddr/benchmarks/).

## Requirements

- DDR core package
- DiffRoute (requires CUDA for GPU acceleration)
- Trained DDR model checkpoint for meaningful comparison
