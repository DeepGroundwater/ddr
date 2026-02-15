---
icon: lucide/brain-cog
---

# Model Training

This guide covers training DDR models to learn optimal routing parameters from observed streamflow data.

## Overview

DDR training optimizes two neural networks — a KAN (static spatial parameters) and a CudaLSTM (time-varying parameters) — to predict physical routing parameters from catchment attributes and meteorological forcings. The training loop:

1. Reads lateral inflow (Q') from unit catchment predictions
2. Predicts static parameters (channel geometry, K_D) using the KAN
3. Predicts time-varying parameters (Manning's n, d_gw) using the CudaLSTM
4. Routes flow through the river network using Muskingum-Cunge
5. Computes loss against observed streamflow
6. Backpropagates gradients through the entire system

## Quick Start

```bash
python scripts/train.py --config-name your_config.yaml
```

## Configuration

### Essential Training Options

```yaml
mode: training
geodataset: lynker_hydrofabric  # or merit

experiment:
  epochs: 5                    # Number of training epochs
  batch_size: 64               # Gauges per batch
  rho: 365                     # Training window (days)
  warmup: 3                    # Warmup days excluded from loss
  shuffle: true                # Shuffle training data
  checkpoint: null             # Resume from checkpoint (optional)
```

### KAN Configuration

```yaml
kan:
  hidden_size: 21              # Hidden layer size (recommend 2n+1)
  num_hidden_layers: 2         # Number of hidden layers
  input_var_names:             # Catchment attributes as inputs
    - aridity
    - meanelevation
    - meanP
    - log10_uparea
    # ... more attributes
  learnable_parameters:        # Static spatial parameters to learn
    - q_spatial                # Shape factor
    - top_width                # Channel width
    - side_slope               # Channel side slope
    - K_D_delta                # Hydraulic conductivity correction (leakance)
  grid: 50                     # KAN grid size
  k: 2                         # KAN spline order
```

## Training Process

### 1. Data Loading

DDR uses PyTorch DataLoaders with custom collate functions:

```python
dataset = cfg.geodataset.get_dataset_class(cfg=cfg)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=cfg.experiment.batch_size,
    sampler=RandomSampler(dataset),
    collate_fn=dataset.collate_fn,
)
```

### 2. Forward Pass

For each batch:

```python
# Get lateral inflows
streamflow_predictions = flow(routing_dataclass=routing_dataclass)

# Select attribute columns for each network
attr_names = routing_dataclass.attribute_names
normalized_attrs = routing_dataclass.normalized_spatial_attributes
kan_attrs = select_columns(normalized_attrs, cfg.kan.input_var_names, attr_names)
lstm_attrs = select_columns(normalized_attrs, cfg.cuda_lstm.input_var_names, attr_names)

# Predict static spatial parameters (channel geometry, K_D)
spatial_params = nn(inputs=kan_attrs)

# Predict time-varying parameters (Manning's n, d_gw)
lstm_params = lstm_nn(forcings=forcing_data, attributes=lstm_attrs)

# Route flow through network
dmc_output = routing_model(
    routing_dataclass=routing_dataclass,
    spatial_parameters=spatial_params,
    streamflow=streamflow_predictions,
    lstm_params=lstm_params,
)
```

### 3. Loss Computation

Loss is computed on daily-averaged discharge after warmup:

```python
# Downsample to daily
daily_runoff = ddr_functions.downsample(dmc_output["runoff"], rho=num_days)

# Compute MSE loss (excluding warmup period)
loss = mse_loss(daily_runoff[:, warmup:], observations[:, warmup:])
```

### 4. Checkpointing

Models are saved periodically:

```
output/<run_name>/saved_models/
├── ddr_..._epoch_1_mb_0.pt
├── ddr_..._epoch_1_mb_64.pt
└── ...
```

## Resuming Training

To resume from a checkpoint:

```yaml
experiment:
  checkpoint: /path/to/checkpoint.pt
```

The training will resume from the saved epoch and mini-batch.

## Monitoring

Training logs include:

- Loss values per mini-batch
- NSE, RMSE, KGE metrics periodically
- Parameter statistics (e.g., median Manning's n)

## Tips

1. **Start with smaller batch sizes** (8-16) for debugging
2. **Use warmup** (3+ days) to allow routing to stabilize
3. **Monitor for NaN losses** - may indicate unstable parameters
4. **Save checkpoints frequently** - training can take hours/days

## Next Steps

- [Model Testing](test.md): Evaluate trained models
- [Benchmarks](../benchmarks/index.md): Compare against other models
