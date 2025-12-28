---
icon: lucide/rocket
---


# Getting started

### Dependencies

The following commands will allow you to install all required dependencies for DDR

```sh
# CPU
uv sync --all-packages
. .venv/bin/activate

# or GPU
uv sync --all-packages --extra cu124
. .venv/bin/activate
```

### Data Engine

Next, you need to create the necessary data files for running a routing across your domain.
- The example below is for the NOAA-OWP Hydrofabric v2.2 (Dataset is not included in the repo)
- This requires the `ddr-engine` local package to be installed (which is done automatically through the above `uv sync`)
- The gauges.csv can be found [here](https://github.com/DeepGroundwater/datasets/tree/master/mhpi/dHBV2.0UH)

```sh
uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py <PATH/TO/conus_nextgen.gpkg> data/ --gages datasets/mhpi/dHBV2.0UH/training_gauges.csv
```

This will create two files used for routing
- `hydrofabric_v2.2_conus_adjacency.zarr`
  - a sparse COO matrix containing the whole river network for Hydrofabric v2.2 across CONUS
- `hydrofabric_v2.2_gages_conus_adjacency.zarr`
  - a zarr.Group of sparse coo matrices for river networks upstream of USGS Gauges

### Model Train

All that's left is to train a routing model
```sh
# Train a model using the MHPI S3 defaults
python scripts/train.py --config-name example_config.yaml
```
