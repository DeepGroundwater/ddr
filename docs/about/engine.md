---
icon: lucide/dam
---

# Geospatial Engine

# DDR Engine

This folder contains scripts and tools meant to format geospatial datasets and create objects which help our routing. Examples include:
- Creaing adjacency matrices for implicit muskingum cunge routing
- Creating adjacency matrices for mapping gauge locations within geospatial datasets

To install these dependencies, please run the following command from the project root
```sh
uv sync
```

## Why have an `engine/` folder?

## Why use a COO matrix?

## Examples:

### CONUS v2.2 Hydrofabric

```python
# CONUS adjacency script
python engine/hydrofabric/v2.2/adjacency.py <path to hydrofabric gpkg> <store path>
```

```python
# gages adjacency script
python engine/hydrofabric/v2.2/gages_adjacency.py <path to hydrofabric gpkg> <path to training gauges csv> <store path> --conus-adj <path to CONUS adjacency store>
```

### MERIT Flowlines

```python
# CONUS adjacency script
uv run python engine/merit/adjacency.py <path to MERIT continental shp file> <store path>
```

```python
# gages adjacency script
uv run python engine/merit/gages_adjacency.py <path to merit shp> <path to training gauges csv> <store path> --conus-adj <path to CONUS adjacency store>
```
