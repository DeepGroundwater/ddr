---
icon: lucide/play
---

# Routing with Trained Weights

This guide covers running DDR inference (routing) on new domains or time periods using trained model weights.

## Overview

Routing mode runs forward simulation without computing metrics or requiring observations:

1. Load trained model checkpoint
2. Route flow through specified catchments or entire network
3. Save predictions to zarr

This is useful for:

- **Operational forecasting**: Route flow in near-real-time
- **Ungauged basins**: Generate predictions where no observations exist
- **Scenario analysis**: Route different lateral inflow scenarios

## Quick Start

```bash
python scripts/router.py --config-name your_routing_config.yaml
```

## Configuration

### Essential Routing Options

```yaml
mode: routing
geodataset: lynker_hydrofabric

experiment:
  batch_size: 64
  start_time: 2020/01/01
  end_time: 2020/12/31
  checkpoint: /path/to/trained_model.pt

data_sources:
  # Option 1: Route specific catchments
  target_catchments:
    - wb-1234
    - wb-5678

  # Option 2: Route all gauged locations
  gages: /path/to/gages.csv
  gages_adjacency: /path/to/gages_adjacency.zarr

  # Option 3: Route entire network (no target_catchments or gages)
  conus_adjacency: /path/to/conus_adjacency.zarr
```

### Routing Targets

| Configuration | Behavior |
|---------------|----------|
| `target_catchments` specified | Route upstream of listed catchments |
| `gages` + `gages_adjacency` | Route upstream of all gauges |
| Neither specified | Route entire river network |

## Routing Process

### 1. Network Selection

DDR selects the appropriate subnetwork based on configuration:

```python
if cfg.data_sources.target_catchments:
    # Route specific catchments
    num_outputs = len(routing_dataclass.outflow_idx)
elif cfg.data_sources.gages:
    # Route gauge locations
    num_outputs = len(routing_dataclass.outflow_idx)
else:
    # Route all segments
    num_outputs = routing_dataclass.adjacency_matrix.shape[0]
```

### 2. Forward Pass

```python
nn = nn.eval()
with torch.no_grad():
    for routing_dataclass in dataloader:
        streamflow_predictions = flow(routing_dataclass=routing_dataclass)
        spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes)
        dmc_output = routing_model(
            routing_dataclass=routing_dataclass,
            spatial_parameters=spatial_params,
            streamflow=streamflow_predictions,
        )
```

### 3. Output

Predictions are saved as zarr:

```python
ds = xr.Dataset(
    data_vars={"predictions": pred_da},
    attrs={
        "start_time": start_time,
        "end_time": end_time,
        "model": checkpoint_path,
    },
)
ds.to_zarr(output_path / "router_output.zarr")
```

## Output Format

```
output/<run_name>/
├── router_output.zarr/
│   ├── predictions/          # (segments, time) discharge values
│   └── .zattrs               # Metadata
└── .hydra/config.yaml
```

### Loading Results

```python
import xarray as xr

ds = xr.open_zarr("output/<run>/router_output.zarr")
predictions = ds.predictions  # (segments, time) DataArray
```

## Performance Tips

1. **Use GPU**: Set `device: 0` (or appropriate GPU ID)
2. **Batch size**: Larger batches = faster, but more memory
3. **Target specific catchments**: Routing subnetworks is faster than full CONUS

## Example: Operational Routing

```yaml
# Route daily forecasts for a single basin
mode: routing
geodataset: lynker_hydrofabric
device: 0

experiment:
  batch_size: 1
  start_time: 2024/01/01
  end_time: 2024/01/07
  checkpoint: /models/production_model.pt

data_sources:
  target_catchments:
    - wb-12345  # Basin outlet
  streamflow: /forecasts/nwm_forecast_20240101/
```

## Next Steps

- [Benchmarks](../benchmarks/index.md): Compare routing results against other models
- [Model Testing](test.md): Evaluate model performance with observations
