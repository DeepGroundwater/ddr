#!/usr/bin/env python

"""
@author Nels Frazier
@author Tadd Bindas

@date Febuary 8, 2025
@version 0.2

An introduction script for building a lower triangular adjancency matrix
from a NextGen hydrofabric and writing a sparse zarr group
"""
from pathlib import Path

import geopandas as gpd
import numpy as np
import graphlib as gl
import sys

from scipy import sparse
import zarr

pkg = sys.argv[1]
gauge = sys.argv[2]
out_path = sys.argv[3]

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

# Comverting to a sparse COO matrix, and saving the output in many arrays within a zarr v3 group
out_path = Path(out_path)
store = zarr.storage.LocalStore(root=out_path)
if out_path.exists():
    root = zarr.open_group(store=store) 
else:
    root = zarr.create_group(store=store)   

coo = sparse.coo_matrix(matrix)
zarr_order = np.array([int(_id.split("-")[1]) for _id in ts_order], dtype=np.int32)

gauge_root = root.create_group(name=gauge)
indices_0 = gauge_root.create_array(
    name='indices_0', shape=coo.row.shape, dtype=coo.row.dtype
)
indices_1 = gauge_root.create_array(
    name='indices_1', shape=coo.col.shape, dtype=coo.row.dtype
)
values = gauge_root.create_array(
    name='values', shape=coo.data.shape, dtype=coo.data.dtype
)
order = gauge_root.create_array(
    name='order', shape=zarr_order.shape, dtype=zarr_order.dtype
)
indices_0[:] = coo.row
indices_1[:] = coo.col
values[:] = coo.data
order[:] = zarr_order

gauge_root.attrs["format"] = "COO"
gauge_root.attrs["shape"] = list(coo.shape)
gauge_root.attrs["data_types"] = {
    "indices_0": coo.row.dtype.__str__(),
    "indices_1": coo.col.dtype.__str__(),
    "values": coo.data.dtype.__str__(),
}
print(f"Gauge {gauge} written to zarr")

# Visual verification
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)
# print(fp)
# print(matrix)
