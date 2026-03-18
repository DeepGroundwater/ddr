---
icon: lucide/sigma
---

# Summed Lateral Flow

The summed lateral flow (Summed Q') baseline computes streamflow at gauge locations by simply summing all upstream lateral inflows — no routing physics applied. This provides a lower-bound benchmark: if DDR's routed predictions don't improve over this sum, the routing model isn't adding value.

## Why It Matters

Routing redistributes flow in time — it delays and attenuates flood waves as they travel downstream. The Summed Q' baseline skips this step entirely, giving you a direct measure of how much your unit catchment predictions (from dHBV, NWM, or any lumped model) contribute to the total signal vs. how much routing improves it.

Comparing DDR against Summed Q' quantifies the effect of routing on the predicted hydrograph relative to a simple summation baseline.

## Quick Start

```bash
ddr summed-q-prime --config-name your_config
```

## Configuration

Summed Q' uses the same config structure as training/testing, but only requires:

```yaml
mode: testing  # or training — the mode doesn't affect summed Q' computation
geodataset: lynker_hydrofabric  # or merit

experiment:
  start_time: 1995/10/01
  end_time: 2010/09/30

data_sources:
  streamflow: "s3://mhpi-spatial/hydrofabric_v2.2_dhbv_retrospective"
  observations: "s3://mhpi-spatial/usgs_streamflow_observations/"
  gages: /path/to/gauges.csv
  gages_adjacency: /path/to/gages_adjacency.zarr
```

No KAN or trained checkpoint is needed — this is a pure data summation.

## How It Works

For each gauge location:

1. Look up all upstream catchments from the gauge's adjacency subgraph
2. Sum the lateral inflow (Q') across all upstream catchments for each timestep
3. Compare the summed flow against USGS observations

```
Q_summed(t) = Σ Q'_i(t)   for all catchments i upstream of the gauge
```

This is equivalent to assuming instantaneous flow propagation — all upstream runoff arrives at the gauge on the same day it was generated.

## Output

Results are saved to the Hydra output directory:

```
output/<run_name>/
├── summed_q_prime.zarr          # Predictions and observations
├── metrics_summary_<timestamp>.json    # Aggregate statistics
└── detailed_metrics_<timestamp>.csv    # Per-gauge metrics
```

### Loading Results

```python
import xarray as xr

ds = xr.open_zarr("output/<run>/summed_q_prime.zarr")
print(ds)
# <xarray.Dataset>
# Dimensions:       (gage_ids: N, time: T)
# Data variables:
#     predictions   (gage_ids, time) float32
#     observations  (gage_ids, time) float32
```

## Relationship to Hot-Start

The [hot-start initialization](hot_start.md) solves the same mathematical problem — summing upstream contributions via the sparse triangular solve $(\mathbf{I} - \mathbf{N}) \cdot \mathbf{Q}_0 = \mathbf{Q}'_0$ — but at runtime within the routing engine. Summed Q' does it as a standalone diagnostic without any routing physics applied.
