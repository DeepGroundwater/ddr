---
icon: lucide/gauge
---

# Benchmarks

The `ddr-benchmarks` package provides tools for comparing DDR against other routing models on identical input data. This enables rigorous, apples-to-apples performance evaluation.

!!! note
    Benchmarking is currently only supported on the **MERIT** geodataset.

## Overview

Benchmarking routing models requires:

1. **Identical input data** - Same lateral inflows (Q'), network topology, and time period
2. **Consistent evaluation** - Same metrics (NSE, KGE, RMSE) computed on same observations
3. **Fair comparison** - Account for differences in model formulations and parameters

The benchmarks package addresses all three by reusing DDR's existing data infrastructure while providing adapters for other routing models.

## Supported Models

| Model | Type | Status | Description |
|-------|------|--------|-------------|
| **DDR** | Differentiable Muskingum-Cunge | Baseline | Physics-based with learned parameters |
| **DiffRoute** | Differentiable LTI Routing | Supported | Linear Time-Invariant routing with multiple IRF options |
| **Summed Q'** | Lateral flow summation | Supported | Optional non-routed baseline comparison |
| *RAPID* | Muskingum | Planned | Traditional non-differentiable routing |

## Architecture

The benchmark runs in two phases:

1. **Phase 1 — DDR**: Runs the full time-batched DataLoader loop (same as `scripts/test.py`), accumulating predictions across all gages.
2. **Phase 2 — DiffRoute**: Iterates over each gage independently, building a connected NetworkX graph from its zarr subgroup. This avoids the disconnected-graph problem that arises from the full CONUS adjacency matrix.

An optional **Summed Q'** baseline can be included. This loads pre-computed lateral flow sums (from `scripts/summed_q_prime.py`) and compares them alongside DDR and DiffRoute.

## Installation

The benchmarks package is installed as an optional dependency:

```bash
# Install with benchmarks support
pip install ddr[benchmarks]

# Or install separately
pip install ddr-benchmarks
```

**Note**: DiffRoute requires CUDA. The benchmarks will skip DiffRoute comparisons on CPU-only systems.

## Quick Start

```bash
# Copy the example config and customize paths
cp benchmarks/config/example_benchmark.yaml benchmarks/config/benchmark.yaml

# Run benchmark
cd benchmarks
uv run python scripts/benchmark.py

# Override configuration options
uv run python scripts/benchmark.py \
    experiment.checkpoint=/path/to/model.pt \
    diffroute.k=0.1 \
    diffroute.x=0.25

# Include summed Q' baseline
uv run python scripts/benchmark.py \
    summed_q_prime=/path/to/summed_q_prime.zarr
```

## Output

The benchmark produces publication-quality plots and console diagnostics:

### Plots (saved to `output/<run>/plots/`)

| File | Description |
|------|-------------|
| `nse_cdf_comparison.png` | CDF of NSE across all gauges |
| `kge_cdf_comparison.png` | CDF of KGE across all gauges |
| `metric_boxplot_comparison.png` | 6-panel boxplot (Bias, RMSE, FHV, FLV, NSE, KGE) |
| `gauge_map_ddr_NSE.png` | Map of gauges colored by DDR NSE |
| `gauge_map_diffroute_NSE.png` | Map of gauges colored by DiffRoute NSE |
| `gauge_map_sqp_NSE.png` | Map of gauges colored by summed Q' NSE (if enabled) |
| `hydrographs/*.png` | Per-gage time series with all models overlaid |

### Console Output

Mass balance accumulation comparison is logged for each model:

```
=== Mass Balance Accumulation Comparison ===
DDR vs Obs       — Mean rel. error: 0.1234, Median: 0.0567
DiffRoute vs Obs — Mean rel. error: 0.2345, Median: 0.1234
DDR vs summed Q' — Mean rel. error: 0.0456, Median: 0.0234
```

### Results (saved to `output/<run>/benchmark_results.zarr`)

```python
import xarray as xr

ds = xr.open_zarr("output/<run>/benchmark_results.zarr")
# <xarray.Dataset>
# Dimensions:                 (gage_ids: N, time: T)
# Data variables:
#     ddr_predictions         (gage_ids, time) float64
#     diffroute_predictions   (gage_ids, time) float64
#     observations            (gage_ids, time) float64
```

## Package Structure

```
benchmarks/
├── scripts/
│   └── benchmark.py             # Entry point (Hydra CLI)
├── src/ddr_benchmarks/
│   ├── __init__.py              # Package exports
│   ├── benchmark.py             # Benchmark runner and plotting
│   ├── diffroute_adapter.py     # COO → NetworkX conversion for DiffRoute
│   └── validation/
│       ├── __init__.py
│       ├── benchmark.py         # BenchmarkConfig (DDR + model configs)
│       └── diffroute.py         # DiffRouteConfig
├── config/
│   ├── benchmark.yaml           # Active Hydra configuration
│   ├── example_benchmark.yaml   # Example config template (fully commented)
│   └── hydra/
│       └── settings.yaml
└── pyproject.toml
```

## Key Components

### Benchmark Runner (`benchmark.py`)

The main benchmark script follows the same pattern as `scripts/test.py`:

1. Load dataset using DDR's `geodataset.get_dataset_class()`
2. Initialize DDR models (KAN, DMC, StreamflowReader)
3. **Phase 1**: Run DDR on time-batched DataLoader, accumulate predictions
4. **Phase 2**: Run DiffRoute per-gage using zarr subgroup graphs
5. Optionally load summed Q' predictions for baseline comparison
6. Compute metrics using DDR's `Metrics` class
7. Log mass balance accumulation comparison
8. Generate comparison plots (CDF, boxplots, gauge maps, hydrographs)
9. Save results to zarr

### DiffRoute Adapter (`diffroute_adapter.py`)

Converts DDR's sparse COO adjacency matrices to DiffRoute-compatible NetworkX graphs:

- `zarr_group_to_networkx()` - Load zarr subgroup → NetworkX DiGraph
- `create_param_df()` - Create parameter DataFrame for DiffRoute
- `build_diffroute_inputs()` - End-to-end conversion utility

## Next Steps

- [DiffRoute Comparison](diffroute.md) - Detailed guide to comparing DDR vs DiffRoute
- [Configuration Reference](config.md) - Full configuration options
