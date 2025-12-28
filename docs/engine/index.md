---
icon: lucide/dam
---

# Geospatial DDR Engine

The `engine/` workspace package contains build scripts and tools meant to format geospatial datasets and create sparse matrices which help our routing. The following sparse matrices are created:
- Network/adjacency matrices for implicit muskingum cunge routing
- Network/adjacency matrices for mapping gauge locations within geospatial datasets

## Why have an `engine/` folder?

The `engine/` folder is meant to contain all necessary code for generating end-to-end routing without having the need for extra repositories. The user only needs to provide the dataset (and unit-catchment flow predictions), and the code will have all of the tools needed for routing.

## Why use a COO matrix?

As explained [here](http://rapid-hub.org/docs/RAPID_Parallel_Computing.pdf#page=5.00) routing can be efficiently solved using a sparse network (otherwise known as an adjacency) matrix and a backwards linear solution. Storing these matrixes then becomes a choice of what format to use for universal readability and efficient storage. Thus, a sparse COO was chosen as COO is fast to turn into other formats, and is readable given only coordinates are stored.

It was necessary to build the tools to convert datasets to their matrix form as river networks don't often ship in sparse form. We used the [Binsparse](https://github.com/ivirshup/binsparse-python) specification for storing the matrices.

## Setup

To install these dependencies, please run the following command from the project root
```sh
uv sync --all-packages
```
which will install the `ddr-engine` package

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
