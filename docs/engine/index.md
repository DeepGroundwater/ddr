---
icon: lucide/dam
---

# Geospatial DDR Engine

The `engine/` workspace package contains build scripts and tools meant to format geospatial datasets and create sparse matrices which help our routing. The following sparse matrices are created:
- Network/adjacency matrices for implicit muskingum cunge routing
- Network/adjacency matrices for mapping gauge locations within geospatial datasets

The goal for the `engine/` package is to give all of the necessary data functions for DDR to the user. The user only needs to provide the dataset (and unit-catchment flow predictions), and the code will have all of the tools needed for routing.

## Why use a COO matrix?

As explained [here](http://rapid-hub.org/docs/RAPID_Parallel_Computing.pdf#page=5.00) routing can be efficiently solved using a sparse network (otherwise known as an adjacency) matrix and a backwards linear solution. Storing these matrixes then becomes a choice of what format to use for universal readability and efficient storage. Thus, a sparse COO was chosen as COO is fast to turn into other formats, and is readable given only coordinates are stored.

It was necessary to build the tools to convert datasets to their matrix form as river networks don't often ship in sparse form. We used the Binsparse specification for storing the matrices. See the [Binsparse COO Format](binsparse.md) documentation for details on the storage format and API.

## Package API

The `ddr_engine` package exports I/O functions at the package level:

```python
from ddr_engine import (
    # Primary API (recommended)
    coo_to_zarr,      # Write COO matrix (pass geodataset name)
    coo_from_zarr,    # Read COO matrix (auto-detects geodataset)
    coo_to_zarr_group,  # Write gauge subset

    # Converter registry
    list_geodatasets,    # List available geodatasets
    register_converter,  # Register custom geodataset
)
```

### Example Usage

```python
from ddr_engine import coo_to_zarr, coo_from_zarr

# Write - specify geodataset name
coo_to_zarr(coo, ts_order, Path("output.zarr"), "merit")

# Read - auto-detects from metadata
coo, ts_order = coo_from_zarr(Path("output.zarr"))
```

## Setup

To install these dependencies, please run the following command from the project root
```sh
uv sync --all-packages
```
which will install the `ddr-engine` package

## Examples:

NOTE: by default:
- `--path` will be set to `data/`
- `--gages` will be set to `references/gage_info/dhbv2_gages.csv`


### CONUS v2.2 Hydrofabric

!!! warning
    Dataset is not included in the repo and needs to be downloaded

```sh
uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py <PATH/TO/conus_nextgen.gpkg>
```

### MERIT Flowlines

!!! note
    Dataset is not included in the repo and can be downloded from the [following location](https://drive.google.com/drive/folders/1DhLXCdMYVkRtlgHBHkiFmpPjTQJX5k1g?usp=sharing)

```sh
uv run python -m ddr_engine.merit /path/to/riv_pfaf_X_MERIT_Hydro_v07_Basins_v01_bugfix1.shp --path data/ --gages references/gage_info/dhbv2_gages.csv
```
