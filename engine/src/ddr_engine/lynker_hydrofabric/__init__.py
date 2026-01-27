"""Functions for building data matrices from the Lynker Hydrofabric v2.2"""

from .build import build_lynker_hydrofabric_adjacency, build_lynker_hydrofabric_gages_adjacency
from .graph import build_graph_from_wb_network, find_origin, preprocess_river_network, subset, subset_upstream
from .io import coo_to_zarr, coo_to_zarr_group, create_coo, create_matrix, index_matrix

__all__ = [
    # build
    "build_lynker_hydrofabric_adjacency",
    "build_lynker_hydrofabric_gages_adjacency",
    # graph
    "find_origin",
    "subset",
    "preprocess_river_network",
    # io
    "create_matrix",
    "coo_to_zarr",
    "coo_to_zarr_group",
    "create_coo",
    "index_matrix",
]
