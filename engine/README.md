# DDR Engine

This folder contains scripts and tools meant to format versions of the hydrofabric and create objects which help our routing. Examples include:
- Creaing adjacency matrices for implicit muskingum cunge routing. Matrices are indexed based on the CONUS geopackage.
- Creating pyiceberg warehouses for data structure storage

To install these dependencies, please run the following command from the project root
```sh
uv sync
```

To run the scripts, please run the following commands from the project root:
```python
# CONUS adjacency script
python engine/adjacency.py <path to hydrofabric gpkg> <store path>
```

```python
# # gages adjacency script
python engine/gages_adjacency.py <path to hydrofabric gpkg> <path to training gauges csv> <store path> --conus-adj <path to CONUS adjacency store>
```

To get a list of gauges to train your model on, please see the https://github.com/DeepGroundwater/datasets repo.
