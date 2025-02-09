# DDR Engine

This folder contains scripts and tools meant to format versions of the hydrofabric and create objects which help our routing. Examples include:
- Mapping MERIT streamflow predictions to the hydrofabric
- Creaing adjacency matrices for implicit muskingum cunge routing 

How to run adjacency matrix:
```python 
python engine/adjacency.py <path to hydrofabric gpkg> <store key> <store path>
```
