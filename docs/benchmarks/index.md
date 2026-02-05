---
icon: lucide/gauge
---

# Benchmarks

The `ddr-benchmarks` package provides tools for comparing DDR against other routing models on identical input data. This enables rigorous, apples-to-apples performance evaluation.

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
| *RAPID* | Muskingum | Planned | Traditional non-differentiable routing |

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
# Run benchmark with default configuration
cd benchmarks
python -m ddr_benchmarks.benchmark

# Override configuration options
python -m ddr_benchmarks.benchmark \
    experiment.checkpoint=/path/to/model.pt \
    diffroute.k=0.1 \
    diffroute.x=0.25
```

## Package Structure

```
benchmarks/
├── src/ddr_benchmarks/
│   ├── __init__.py           # Package exports
│   ├── benchmark.py          # Main benchmark runner
│   └── diffroute_adapter.py  # COO → NetworkX conversion for DiffRoute
├── config/
│   └── benchmark.yaml        # Hydra configuration
└── pyproject.toml
```

## Key Components

### Benchmark Runner (`benchmark.py`)

The main benchmark script follows the same pattern as `scripts/test.py`:

1. Load dataset using DDR's `geodataset.get_dataset_class()`
2. Initialize DDR models (KAN, DMC, StreamflowReader)
3. Run both DDR and DiffRoute on each batch
4. Compute metrics using DDR's `Metrics` class
5. Generate comparison plots and save results

### DiffRoute Adapter (`diffroute_adapter.py`)

Converts DDR's sparse COO adjacency matrices to DiffRoute-compatible NetworkX graphs:

- `zarr_to_networkx()` - Load zarr store → NetworkX DiGraph
- `create_param_df()` - Create parameter DataFrame for DiffRoute
- `build_diffroute_inputs()` - End-to-end conversion utility

## Next Steps

- [DiffRoute Comparison](diffroute.md) - Detailed guide to comparing DDR vs DiffRoute
- [Configuration Reference](config.md) - Full configuration options
