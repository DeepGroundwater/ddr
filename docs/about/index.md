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

!!! warning
    Dataset is not included in the repo and needs to be downloaded

```sh
uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py <PATH/TO/conus_nextgen.gpkg> data/ --gages datasets/mhpi/dHBV2.0UH/training_gauges.csv
```

### MERIT Flowlines

!!! note
    Dataset is not included in the repo and can be downloded from the [following location](https://drive.google.com/drive/folders/1DhLXCdMYVkRtlgHBHkiFmpPjTQJX5k1g?usp=sharing)

```sh
uv run python engine/scripts/build_merit_matrices.py <PATH/TO/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp> data/ --gages datasets/mhpi/dHBV2.0UH/training_gauges.csv
```
