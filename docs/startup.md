---
icon: lucide/rocket
---

# Getting Started

This guide will walk you through installing DDR and running your first routing experiment.

## Prerequisites

Before installing DDR, ensure you have:

- **Python 3.11+**: DDR requires Python 3.11 or later
- **uv**: The fast Python package manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
- **Git**: For cloning the repository

For GPU support (optional but recommended):
- **CUDA 12.4+**: Required for GPU acceleration
- **CuPy**: Will be installed automatically with GPU extras

## Installation

### Clone the Repository

```bash
git clone https://github.com/DeepGroundwater/ddr.git
cd ddr
```

### Install Dependencies

DDR uses `uv` for dependency management. The repository is organized as a workspace with three packages:

| Package | Description |
|---------|-------------|
| `ddr` | Core routing library |
| `ddr-engine` | Geospatial data preparation tools |
| `ddr-benchmarks` | Benchmarking tools for model comparison |

Choose the appropriate installation based on your needs:

=== "Full Workspace"

    ```bash
    # Installs ddr, ddr-engine, and ddr-benchmarks, all dev tools, local doc builds
    uv sync --all-packages --all-extras
    ```

=== "Core Only"

    ```bash
    # Installs only the ddr package (skip engine and benchmarks)
    uv sync --package ddr
    ```

=== "GPU (CUDA 12.4)"

    ```bash
    # Full workspace with GPU support
    uv sync --all-packages
    ```

The full workspace is recommended for development and paper verification. Use core-only for production routing.

### Verify Installation

```python
import ddr
print(ddr.__version__)
```

## Data Preparation

Before training, you need to create the sparse adjacency matrices that define the river network topology.

### Step 1: Download Geospatial Data

DDR requires a geospatial fabric defining the river network. Currently supported:

**NOAA-OWP Hydrofabric v2.2** (Recommended for CONUS):

- Download from Lynker-Spatial
- File: `conus_nextgen.gpkg`

**MERIT Hydro** (Global coverage):

- Download from [MERIT Hydro website](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/)
- Or use the [Google Drive mirror](https://drive.google.com/drive/folders/1DhLXCdMYVkRtlgHBHkiFmpPjTQJX5k1g?usp=sharing)

### Step 2: Prepare Gauge Information

Create a CSV file with gauge information. Required columns:

| Column | Description | Example |
|--------|-------------|---------|
| `STAID` | USGS Station ID (7-8 digits, not zero padded) | `1563500` |
| `DRAIN_SQKM` | Drainage area in km² | `2847.5` |
| `LAT_GAGE` | Latitude of gauge | `40.2345` |
| `LNG_GAGE` | Longitude of gauge | `-76.8901` |
| `STANAME` | Station name (optional) | `Susquehanna River`

__NOTE:__ to use MERIT you will need to have COMID also specified and mapped to each river gage

You can find pre-prepared gauge lists in the [streamflow_datasets repository](https://github.com/DeepGroundwater/datasets).

### Step 3: Build Adjacency Matrices

Run the engine script to create the sparse network matrices:

=== "Hydrofabric v2.2"

    ```bash
    uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py \
        <PATH/TO/conus_nextgen.gpkg> \
        data/ \
        --gages streamflow_datasets/mhpi/dHBV2.0UH/training_gauges.csv
    ```

=== "MERIT Hydro"

    ```bash
    uv run python engine/scripts/build_merit_matrices.py \
        <PATH/TO/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp> \
        data/ \
        --gages your_gauges.csv
    ```

This creates two zarr stores:

| File | Description |
|------|-------------|
| `*_conus_adjacency.zarr` | Full CONUS river network in sparse COO format |
| `*_gages_conus_adjacency.zarr` | Per-gauge upstream subnetworks |

These zarr stores are coo matrices stored using the [binsparse-python](https://github.com/ivirshup/binsparse-python) specification

## Configuration

DDR uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are in YAML format.

### Configuration Structure

The most important part of your config is shown below. These are the data sources for DDR to work

```yaml
mode: training               # training, testing, or routing
geodataset: lynker_hydrofabric  # the geodataset used to determine river connectivity
name: ddr-v${oc.env:DDR_VERSION,dev}-${geodataset}-${mode}  # The name of the training run

data_sources:
  attributes: "s3://mhpi-spatial/hydrofabric_v2.2_attributes/"  # The path to your geodataset attributes
  geospatial_fabric_gpkg: /path/to/conus_nextgen.gpkg  # the path to your geodataset for river connectivity
  conus_adjacency: /path/to/hydrofabric_v2.2_conus_adjacency.zarr  # the output of engine/ using your geodataset
  gages: /path/to/training_gages.csv  # the training gages used
  gages_adjacency: /path/to/hydrofabric_v2.2_gages_conus_adjacency.zarr  # the output of engine/ using your geodataset for gage subgraph connectivity
  streamflow: "s3://mhpi-spatial/hydrofabric_v2.2_dhbv_retrospective"  # the unit catchment streamflow prediction output you'll be using
  observations: "s3://mhpi-spatial/usgs_streamflow_observations/"  # the USGS observations to train your model against
  target_catchments:
    - 1234   # the river ID for the catchment you want to route upstream of (MERIT EXAMPLE)
    - wb-1234  # the river ID for the catchment you want to route upstream of (Lynker Hydrofabric EXAMPLE)
```

### Key Configuration Options

#### Training

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | Operating mode: `training`, `testing`, `routing` | Required |
| `geodataset` | Dataset type: `lynker_hydrofabric`, `merit` | Required |
| `experiment.rho` | Time window length for training (days) | `None` (full period) |
| `experiment.warmup` | Days excluded from loss calculation to allow the model to warm up | `3` |
| `experiment.batch_size` | Number of gauges per batch | `1` |
| `kan.hidden_size` | KAN hidden layer size (recommend 2n+1 where n=input features) | `21` |
| `device` | GPU ID or `"cpu"` | `0` |

#### Testing/Rouing (Inference)

| Option | Description | Default |
|--------|-------------|---------|
| `mode` | Operating mode: `training`, `testing`, `routing` | Required |
| `geodataset` | Dataset type: `lynker_hydrofabric`, `merit` | Required |
| `experiment.warmup` | Days excluded from loss calculation to allow the model to warm up | `3` |
| `experiment.batch_size` | Number of days included in the batch | `1` |
| `kan.hidden_size` | KAN hidden layer size (recommend 2n+1 where n=input features) | `21` |
| `device` | GPU ID or `"cpu"` | `0` |


__NOTE:__ More geodataset support is coming soon

## Running Your First Model

### Quick Start

```bash
# Training
python scripts/train.py --config-name example_config.yaml

# Testing
python scripts/test.py --config-name example_config.yaml

# Routing a trained model over specified catchments / the whole dataset
python scripts/router.py --config-name example_config.yaml

# Checking the baseline unit catchment metrics
python scripts/summed_q_prime.py --config-name example_config.yaml
```

__NOTE:__ Please change the example config to match what mode/geodataset/method you need to work with. The config in the example is for structure only

### Monitoring

DDR logs progress including:

- Loss values per epoch and mini-batch
- NSE, RMSE, and KGE metrics
- Parameter statistics

Model checkpoints are saved to the `params.save_path` directory.

### Expected Model outputs

After running DDR, you'll have:

```
output/ddr-{version}-{geodataset}-{mode}/YYYY-MM-DD_HH-MM-SS/
├── model/      # KAN model states
├── plots/      # any DDR generated plots
├── saved_models/
    ├── ddr_{version}-{geodataset}-{mode}_epoch_1_mb_0.pt   # Checkpoint file
    ...
    └── ddr_{version}-{geodataset}-{mode}_epoch_5_mb_42.pt   # Checkpoint file
├── pydantic_config.yaml     # Validated configuration
└── ddr-{version}-{geodataset}-{mode}.log  # the log file of all information generated during training
```

## Next Steps

- [Model Training](usage/train.md): Detailed training guide
- [Model Testing](usage/test.md): Evaluate your trained model
- [Routing](usage/routing.md): Run inference with trained weights
- [Engine](engine/index.md): Learn about data preparation
