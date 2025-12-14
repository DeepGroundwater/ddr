---
icon: lucide/rocket
---

!!! note
    This repo is a work in progress and will be updating frequently. Be sure to be using the most recent release version

<p align="center">
  <img src="images/ddr_logo.png" width="40%"/>
</p>


# Getting started

The following commands will allow you to install all required dependencies for DDR

```sh
# CPU
uv synv
. .venv/bin/activate

# or GPU
uv sync --extra cu124
. .venv/bin/activate

```

Now, you can create the necessary data files for running on the NOAA-OWP Hydrofabric v2.2 (Dataset is not included in the repo):
```sh
# Create CONUS adjacency
python engine/adjacency.py PATH/TO/conus_nextgen.gpkg data/conus_adjacency.zarr

# Create gauges adjacency
python engine/gages_adjacency.py PATH/TO/conus_nextgen.gpkg PATH/TO/TRAINING_GAUGES.csv data/gages_adjacency.zarr --conus-adj data/conus_adjacency.zarr

# Train a model using the MHPI S3 defaults
python scripts/train.py --config-name example_config.yaml
```
