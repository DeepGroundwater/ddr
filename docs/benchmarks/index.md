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
python -m ddr_benchmarks

# Override configuration options
python -m ddr_benchmarks \
    experiment.checkpoint=/path/to/model.pt \
    diffroute.k=0.1 \
    diffroute.x=0.25

# Include summed Q' baseline
python -m ddr_benchmarks \
    summed_q_prime=/path/to/summed_q_prime.zarr
```

## Package Structure

```
benchmarks/
├── src/ddr_benchmarks/
│   ├── __init__.py           # Package exports
│   ├── benchmark.py          # Main benchmark runner
│   ├── diffroute_adapter.py  # COO → NetworkX conversion for DiffRoute
│   └── configs/
│       ├── benchmark.py      # BenchmarkConfig (DDR + model configs)
│       └── diffroute.py      # DiffRouteConfig
├── config/
│   ├── benchmark.yaml        # Active Hydra configuration
│   └── example_benchmark.yaml  # Example config template
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
7. Generate comparison plots and save results

### DiffRoute Adapter (`diffroute_adapter.py`)

Converts DDR's sparse COO adjacency matrices to DiffRoute-compatible NetworkX graphs:

- `zarr_group_to_networkx()` - Load zarr subgroup → NetworkX DiGraph
- `create_param_df()` - Create parameter DataFrame for DiffRoute
- `build_diffroute_inputs()` - End-to-end conversion utility

## Next Steps

- [DiffRoute Comparison](diffroute.md) - Detailed guide to comparing DDR vs DiffRoute
- [Configuration Reference](config.md) - Full configuration options
