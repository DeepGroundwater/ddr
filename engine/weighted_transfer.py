#!/usr/bin/env python

"""
@author Tadd Bindas

@date Febuary 17, 2025
@version 0.2

A script to find the weighted-intersection of merit basins to CONUS catchments
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import sparse
import zarr

zone = "73"
path_1 = f"/projects/mhpi/data/MERIT/raw/basins/cat_pfaf_{zone}_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
# path_2 = "/projects/mhpi/data/hydrofabric/v2.2/jrb_2.gpkg"
path_2 = "/projects/mhpi/data/hydrofabric/v2.2/conus_nextgen.gpkg"
out_path = Path(f"/projects/mhpi/data/hydrofabric/v2.2/conus_transition_matrices.zarr")

print("Reading shp files")
gdf1 = gpd.read_file(path_1).set_crs(epsg=4326).to_crs(epsg=5070)
gdf2 = gpd.read_file(path_2, layer="divides").to_crs(epsg=5070)

gdf1['gdf1_orig_area'] = gdf1.geometry.area
gdf2['gdf2_orig_area'] = gdf2.geometry.area

print("Running gdf intersection")
intersection = gpd.overlay(gdf1, gdf2, how='intersection')
intersection['intersection_area'] = intersection.geometry.area
intersection['gdf1_pct'] = (intersection['intersection_area'] / intersection['gdf1_orig_area'])

print("Running generating weighted transfer matrix")
weight_matrix = pd.pivot_table(intersection, 
                             values='gdf1_pct',
                             index='COMID',  # replace with your actual column name from gdf2
                             columns='divide_id',  # replace with your actual column name from gdf1
                             fill_value=0)

print("Saving to sparse zarr store")
store = zarr.storage.LocalStore(root=out_path)
if out_path.exists():
    root = zarr.open_group(store=store) 
else:
    root = zarr.create_group(store=store)  
    
coo = sparse.coo_matrix(weight_matrix.to_numpy())

comid_order = np.array([int(float(_id.split("-")[1])) for _id in weight_matrix.columns.to_numpy()], dtype=np.int32)
merit_basin_order = weight_matrix.index.to_numpy().astype(np.int32)

gauge_root = root.create_group(name=zone)
indices_0 = gauge_root.create_array(
    name='indices_0', shape=coo.row.shape, dtype=coo.row.dtype
)
indices_1 = gauge_root.create_array(
    name='indices_1', shape=coo.col.shape, dtype=coo.row.dtype
)
values = gauge_root.create_array(
    name='values', shape=coo.data.shape, dtype=coo.data.dtype
)
comid_zarr_order = gauge_root.create_array(
    name='comid_order', shape=comid_order.shape, dtype=comid_order.dtype
)
merit_basins_zarr_order = gauge_root.create_array(
    name='merit_basins_order', shape=merit_basin_order.shape, dtype=merit_basin_order.dtype
)
indices_0[:] = coo.row
indices_1[:] = coo.col
values[:] = coo.data
comid_zarr_order[:] = comid_order
merit_basins_zarr_order[:] = merit_basin_order

gauge_root.attrs["format"] = "COO"
gauge_root.attrs["shape"] = list(coo.shape)
gauge_root.attrs["data_types"] = {
    "indices_0": coo.row.dtype.__str__(),
    "indices_1": coo.col.dtype.__str__(),
    "values": coo.data.dtype.__str__(),
}
print(f"{out_path} written to zarr")

# weight_matrix.to_csv("/projects/mhpi/data/hydrofabric/v2.2/73_conus_transition_matrix.csv")
# print("Created transition matrix @ /projects/mhpi/data/hydrofabric/v2.2/73_conus_transition_matrix.csv")
