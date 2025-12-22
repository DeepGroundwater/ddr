"""Functions for building data matrices from the merit flowpaths"""

from .adjacency import coo_to_zarr, create_matrix
from .gages_adjacency import coo_to_zarr_group, create_coo, subset
from .utils import _build_rustworkx_object, _build_upstream_dict_from_merit

__all__ = [
    "coo_to_zarr",
    "create_matrix",
    "coo_to_zarr_group",
    "subset",
    "create_coo",
    "_build_rustworkx_object",
    "_build_upstream_dict_from_merit",
]
