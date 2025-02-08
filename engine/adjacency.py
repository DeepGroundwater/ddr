#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date Febuary 8, 2025
@version 0.2

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric.
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import graphlib as gl
import sys

from scipy import sparse
import xarray as xr

# pkg = sys.argv[1]
# out_path = sys.argv[2]

gauge = "01563500"

pkg = "/projects/mhpi/data/hydrofabric/v2.2/JRB.gpkg"
out_path = "/projects/mhpi/tbindas/ddr/data/networks.zarr"

# Useful for some debugging, not needed for algorithm
# nexi = gpd.read_file(pkg, layer='nexus').set_index('id') 
fp = gpd.read_file(pkg, layer='flowpaths').set_index('id')
network = gpd.read_file(pkg, layer='network').set_index('id')

# Toposort for the win
sorter = gl.TopologicalSorter()

for id in fp.index:
    nex = fp.loc[id]['toid']
    try:
        ds_wb = network.loc[nex]['toid']
    except:
        print("Terminal nex???", nex)
        continue
    # Add a node to the sorter, ds_wb is the node, id is its predesessor
    sorter.add(ds_wb, id)

# There are possibly more than one correct topological sort orders
# Just grab one and go...
ts_order = list(sorter.static_order())

# Reindex the flowpaths based on the topo order
fp = fp.reindex(ts_order)

# Create matrix, "indexed" the same as the re-ordered fp dataframe
matrix = np.zeros( (len(fp), len(fp) ) )

for wb in ts_order:
    nex = fp.loc[wb]['toid']
    if isinstance(nex, float) and np.isnan(nex):
        continue
    # Use the network to find wb -> wb topology
    try:
        ds_wb = network.loc[nex]['toid']
    except KeyError:
        print("Terminal nex???", nex)
        continue
    # Find the inicies of the adajcent flowpaths
    idx = fp.index.get_loc(wb)
    idxx = fp.index.get_loc(ds_wb)
    fp['matrix_idxx'] = idxx
    fp['matrix_idx'] = idx
    # print(wb, " -> ", nex, " -> ", ds_wb)
    # Set the matrix value
    matrix[idxx][idx] = 1

# Ensure, within tolerance, that this is a lower triangular matrix
assert np.allclose(matrix, np.tril(matrix))

out_path = Path(out_path)

if out_path.exists():
    dt = xr.open_datatree(out_path)
else:
    dt = xr.DataTree()

coo = sparse.coo_matrix(matrix)
metadata = {
    "format": "COO",
    "shape": list(coo.shape),
    "data_types": {
        "indices_0": coo.row.dtype.__str__(),
        "indices_1": coo.col.dtype.__str__(),
        "values": coo.data.dtype.__str__(),
    },
}

dt[gauge] = xr.Dataset(
    data_vars={
        'indices_0': ('index', coo.row),
        'indices_1': ('index', coo.col),
        'values': ('index', coo.data)
    },
    coords={
        'index': np.arange(len(coo.data))
    },
    attrs=metadata
)

dt.to_zarr(out_path)

print(f"Gauge {gauge} written to zarr")

# np.save(out_path, matrix)

# Visual verification
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(fp)
# print(matrix)
