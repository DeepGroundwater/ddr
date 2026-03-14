# DDR Examples

Each subdirectory contains a dataset-specific example with pre-trained weights and a config file.

## Directory Structure

```
examples/
├── merit/                      # MERIT-Hydro examples
│   ├── example_config.yaml     # Config pointing to v0.5.2 trained weights
│   ├── ddr-v0.5.2_merit_trained_weights.pt
│   └── plot_parameter_map.ipynb
├── lynker_hydrofabric/         # Lynker Hydrofabric v2.2 examples
│   ├── example_config.yaml     # Config pointing to v0.5.2 trained weights
│   ├── ddr-v0.5.2.lynker_hydrofabric_trained_weights.pt
│   └── plot_parameter_map.ipynb
├── eval/                       # Evaluation notebook (dataset-agnostic)
│   └── evaluate.ipynb
└── parameter_maps/             # Legacy v0.1.0a2 example (Lynker only)
    └── plot_parameter_map.ipynb
```

## Quick Start

1. Set `DDR_DATA_DIR` to your local data directory (or edit the config).
2. Open the notebook for your dataset and run all cells.

Each `example_config.yaml` uses `${oc.env:DDR_DATA_DIR,./../../data}` so paths
resolve relative to the repo root's `data/` folder by default.
