---
icon: lucide/git-compare
---

# DiffRoute Comparison

[DiffRoute](https://github.com/TristHas/DiffRoute) is a differentiable river routing library that implements Linear Time-Invariant (LTI) routing with various Impulse Response Functions (IRFs). This page explains how to compare DDR against DiffRoute.

## Background

Both DDR and DiffRoute implement differentiable routing, but with different approaches:

| Aspect | DDR | DiffRoute |
|--------|-----|-----------|
| **Method** | Muskingum-Cunge (finite difference) | LTI convolution with IRFs |
| **Parameters** | Manning's n, channel geometry | k (travel time), x (weighting) |
| **Learning** | Neural network predicts physical params | Parameters can be learned or fixed |
| **Timestep** | Typically 1 hour | Configurable (dt in days) |

## Per-Gage Routing Architecture

The benchmark routes each gage independently using its zarr subgroup from `gages_adjacency`. This avoids the disconnected-graph problem that arises when loading the full CONUS adjacency matrix (~77K nodes) as a single NetworkX graph.

For each gage:

1. Load the gage's subgroup from `gages_adjacency.zarr`
2. Resolve CONUS-level sparse indices to COMIDs using the full `conus_adjacency` order array (see [Binsparse format](../engine/binsparse.md))
3. Build a connected NetworkX DiGraph for the gage's upstream catchment
4. Create a `RivTree` and route lateral inflows through it
5. Extract discharge at the gage node

This per-gage approach means each subgraph is small (dozens to hundreds of nodes), keeping memory usage low and ensuring valid graph connectivity.

## Muskingum Parameters

DiffRoute's Muskingum IRF uses two parameters:

### k - Wave Travel Time

The **k** parameter represents the time for a flood wave to travel through a reach:

$$k = \frac{L}{c}$$

where:

- $L$ = reach length (m)
- $c$ = wave celerity (m/s)

**Units**: k must be in **days** (same as dt)

**Stability constraint**: For numerical stability, k must satisfy:

$$k \geq \frac{dt}{2(1-x)}$$

For dt = 1 hour (0.0417 days) and x = 0.3:

$$k_{min} = \frac{0.0417}{2 \times 0.7} \approx 0.03 \text{ days}$$

### x - Weighting Factor

The **x** parameter controls the balance between inflow and outflow weighting:

- **x = 0**: Pure reservoir behavior (maximum attenuation)
- **x = 0.5**: Pure translation (no attenuation)
- **Typical range**: 0.1 - 0.3

## Configuration

The benchmark configuration (`benchmark.yaml`) includes DiffRoute-specific options:

```yaml
diffroute:
  # Enable/disable DiffRoute comparison
  enabled: true

  # Impulse Response Function model
  # Options: muskingum, linear_storage, nash_cascade, pure_lag, hayami
  irf_fn: muskingum

  # Maximum delay for LTI router (number of timesteps)
  max_delay: 100

  # Timestep in days (3600/86400 = 1 hour)
  dt: 0.0416667

  # Muskingum k parameter (wave travel time) in days
  # 0.1042 days = 9000s = 2.5 hours (RAPID default)
  k: 0.1042

  # Muskingum x parameter (weighting factor)
  x: 0.3
```

## Running the Comparison

### Basic Usage

```bash
cd benchmarks
uv run python scripts/benchmark.py
```

### With Custom Parameters

```bash
# Faster wave propagation (smaller k)
uv run python scripts/benchmark.py diffroute.k=0.02

# More attenuation (smaller x)
uv run python scripts/benchmark.py diffroute.x=0.1

# Different IRF model
uv run python scripts/benchmark.py diffroute.irf_fn=linear_storage
```

### Disable DiffRoute

```bash
# Run DDR only (useful on CPU-only systems)
uv run python scripts/benchmark.py diffroute.enabled=false
```

## Output

The benchmark produces:

### Metrics (logged)

```
=== DDR Metrics ===
----------------------------------------
Metric     |         Mean |       Median
----------------------------------------
NSE        |       0.7234 |       0.7891
RMSE       |      12.3456 |       8.7654
KGE        |       0.6543 |       0.7012
----------------------------------------

=== DiffRoute Metrics ===
----------------------------------------
Metric     |         Mean |       Median
----------------------------------------
NSE        |       0.6891 |       0.7456
...

=== Summed Q' Metrics ===
...
```

### Mass Balance (logged)

```
=== Mass Balance Accumulation Comparison ===
DDR vs Obs       — Mean rel. error: 0.1234, Median: 0.0567
DiffRoute vs Obs — Mean rel. error: 0.2345, Median: 0.1234
DDR vs summed Q' — Mean rel. error: 0.0456, Median: 0.0234
```

### Plots (saved to `output/<run>/plots/`)

| File | Description |
|------|-------------|
| `nse_cdf_comparison.png` | CDF comparison of NSE distributions |
| `kge_cdf_comparison.png` | CDF comparison of KGE distributions |
| `metric_boxplot_comparison.png` | 6-panel boxplot (Bias, RMSE, FHV, FLV, NSE, KGE) |
| `gauge_map_ddr_NSE.png` | Map of gauges colored by DDR NSE |
| `gauge_map_diffroute_NSE.png` | Map of gauges colored by DiffRoute NSE |
| `gauge_map_sqp_NSE.png` | Map of gauges colored by summed Q' NSE (if enabled) |
| `hydrographs/*.png` | Per-gage hydrographs with all models overlaid |

When summed Q' is enabled, all plots include it as an additional series.

### Results (saved to `output/<run>/benchmark_results.zarr`)

```python
import xarray as xr

ds = xr.open_zarr("output/<run>/benchmark_results.zarr")
print(ds)
# <xarray.Dataset>
# Dimensions:                 (gage_ids: N, time: T)
# Data variables:
#     ddr_predictions         (gage_ids, time) float64
#     diffroute_predictions   (gage_ids, time) float64
#     observations            (gage_ids, time) float64
```

## Data Flow

The benchmark uses a two-phase architecture to ensure both models receive identical lateral inflows:

```
Phase 1: DDR                          Phase 2: DiffRoute
──────────────────                    ─────────────────────

┌─────────────────┐                   For each gage:
│ DDR Dataset     │                   ┌─────────────────┐
│ (DataLoader,    │                   │ Gage Zarr       │
│  365-day batch) │                   │ Subgroup        │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│ StreamflowReader│                   │ Build connected │
│ (full CONUS Q') │                   │ NX DiGraph      │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│ KAN → DMC       │                   │ StreamflowReader│
│ (learned params)│                   │ (gage subset Q')│
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         ▼                                     ▼
┌─────────────────┐                   ┌─────────────────┐
│ DDR Predictions │                   │ LTIRouter +     │
│ (all gages)     │                   │ RivTree         │
└────────┬────────┘                   └────────┬────────┘
         │                                     │
         └──────────────┬──────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │ Metrics Class   │
              │ (NSE, KGE, RMSE)│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Plots & Results │
              │ (CDF, boxplots, │
              │  gauge maps,    │
              │  hydrographs)   │
              └─────────────────┘
```

## Reordering

DDR and DiffRoute use different node orderings:

- **DDR**: Topological order from adjacency matrix
- **DiffRoute**: Depth-first search (DFS) order from RivTree

The benchmark handles this automatically using reordering functions:

```python
from ddr_benchmarks import reorder_to_diffroute, reorder_to_topo

# DDR topo order → DiffRoute DFS order
runoff_diffroute = reorder_to_diffroute(runoff_ddr, topo_order, riv)

# DiffRoute DFS order → DDR topo order
discharge_ddr = reorder_to_topo(discharge_diffroute, topo_order, riv)
```

## References

- [DiffRoute GitHub](https://github.com/TristHas/DiffRoute)
- [DiffRoute Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000760): "Differentiable river routing for end-to-end learning of hydrological processes at diverse scales"
- [DDR Paper](https://doi.org/10.1029/2023WR035337): "Improving River Routing Using a Differentiable Muskingum-Cunge Model"
