# DDR Engine

This folder contains scripts and tools meant to format versions of the hydrofabric and preprocess data structures for usage in DDR

## Getting started:
To install the endpoints needed for the engine repo, you can run the following installation from project root:
```sh
uv sync --extra engine
```

To use the engine, you will need a copy of the [v2.2 NOAA/USGS Hydrofabric](https://www.lynker-spatial.com/data/hydrofabric/v2.2/conus/)

Next, the hydrofabric will need to be written into pyiceberg format as this is significantly faster than using geopandas
```sh
python engine/build_warehouse.py --file conus_nextgen.gpkg
```

Lastly, the adjacency matrix for CONUS can be created and written to zarr
```sh
python engine/adjacency.py <path to hydrofabric gpkg> <store path>
```
