"""Functions for building data matrices from MERIT flowpaths"""

from .build import (
    build_gauge_adjacencies,
    build_merit_adjacency,
    create_adjacency_matrix,
)
from .graph import (
    build_graph,
    build_upstream_dict,
    subset_upstream,
)
from .io import (
    coo_from_zarr,
    coo_to_zarr,
    coo_to_zarr_group,
    create_subset_coo,
)

__all__ = [
    # build
    "build_gauge_adjacencies",
    "build_merit_adjacency",
    "create_adjacency_matrix",
    # graph
    "build_graph",
    "build_upstream_dict",
    "subset_upstream",
    # io
    "coo_from_zarr",
    "coo_to_zarr",
    "coo_to_zarr_group",
    "create_subset_coo",
]
