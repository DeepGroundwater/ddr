# DDR Benchmarks

Benchmarking tools for comparing DDR against other routing models.

## Setup

From repository root:

```bash
# Full workspace (includes benchmarks and engine)
uv sync --all-packages

# Or just DDR core (skip benchmarks/engine)
uv sync --package ddr
```

## Planned Features

- Export DDR test outputs to NetCDF (common format)
- Compare against DiffRoute and other routing models
- Standardized evaluation metrics
