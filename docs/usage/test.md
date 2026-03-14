---
icon: lucide/test-tube
---

# Model Testing

This guide covers evaluating trained DDR models on held-out test periods.

## Overview

Model testing evaluates a trained DDR model on a different time period than training:

1. Load trained model checkpoint
2. Run forward pass on test period data
3. Compare predictions against observations
4. Generate evaluation outputs

## Quick Start

```bash
ddr test --config-name your_test_config
```

## Configuration

### Essential Test Options

```yaml
mode: testing
geodataset: lynker_hydrofabric  # Must match training

experiment:
  batch_size: 64
  start_time: 1995/10/01        # Test period start
  end_time: 2010/09/30          # Test period end
  warmup: 3                     # Days excluded from evaluation during spin-up
  checkpoint: /path/to/trained_model.pt  # Required!
```

Ensure the checkpoint matches the KAN configuration used during training.

## Evaluation Process

### 1. Load Checkpoint

```python
state = torch.load(checkpoint_path, map_location=device)
nn.load_state_dict(state["model_state_dict"])
nn = nn.eval()  # Set to evaluation mode
```

### 2. Run Inference

```python
with torch.no_grad():
    for routing_dataclass in dataloader:
        streamflow_predictions = flow(routing_dataclass=routing_dataclass)
        spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes)
        dmc_output = routing_model(
            routing_dataclass=routing_dataclass,
            spatial_parameters=spatial_params,
            streamflow=streamflow_predictions,
        )
        predictions[:, indices] = dmc_output["runoff"].cpu().numpy()
```

### 3. Evaluate Predictions

Use the `Metrics` class from `ddr.validation` to compare predictions against observations:

```python
from ddr.validation.metrics import Metrics

metrics = Metrics(pred=daily_runoff[:, warmup:], target=observations[:, warmup:])
```

See the `Metrics` class for available evaluation attributes.

## Output

Test results are saved to:

```
output/<run_name>/
├── model_test.zarr           # Predictions and observations
├── plots/                    # Optional evaluation plots
└── .hydra/config.yaml        # Configuration used
```

### Loading Results

```python
import xarray as xr

ds = xr.open_zarr("output/<run>/model_test.zarr")
print(ds)
# <xarray.Dataset>
# Dimensions:       (gage_ids: N, time: T)
# Data variables:
#     predictions   (gage_ids, time) float64
#     observations  (gage_ids, time) float64
```

## Next Steps

- [Routing](routing.md): Run inference on new domains
- [Benchmarks](../benchmarks/index.md): Compare against DiffRoute and other models
