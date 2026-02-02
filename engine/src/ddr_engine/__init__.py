"""DDR Engine - Geodataset preparation tools.

This package provides tools for building sparse COO adjacency matrices from
geodatasets (MERIT Hydro, Lynker Hydrofabric v2.2).

Primary API
-----------
The recommended functions for reading and writing COO matrices:

- ``coo_to_zarr``: Write a COO matrix (pass geodataset name)
- ``coo_from_zarr``: Read a COO matrix (auto-detects geodataset)
- ``coo_to_zarr_group``: Write a gauge subset COO matrix

Converter Registry
------------------
Custom geodatasets can be registered with ``register_converter()``.
List available geodatasets with ``list_geodatasets()``.

Example
-------
>>> from ddr_engine import coo_to_zarr, coo_from_zarr
>>> from scipy import sparse
>>> coo = sparse.coo_matrix(...)
>>> coo_to_zarr(coo, ts_order, Path("output.zarr"), "merit")
>>> coo, ts_order = coo_from_zarr(Path("output.zarr"))  # Auto-detects
"""

from ._version import __version__
from .core import (
    # Primary API
    coo_from_zarr,
    # Generic API (low-level)
    coo_from_zarr_generic,
    coo_to_zarr,
    coo_to_zarr_generic,
    coo_to_zarr_group,
    coo_to_zarr_group_generic,
    # Converter registry
    get_converter,
    list_geodatasets,
    # Converter instances (for advanced use)
    lynker_converter,
    merit_converter,
    register_converter,
)

__all__ = [
    "__version__",
    # Primary API (recommended)
    "coo_to_zarr",
    "coo_from_zarr",
    "coo_to_zarr_group",
    # Generic API (low-level)
    "coo_to_zarr_generic",
    "coo_from_zarr_generic",
    "coo_to_zarr_group_generic",
    # Converter registry
    "get_converter",
    "register_converter",
    "list_geodatasets",
    # Converter instances (for advanced use)
    "merit_converter",
    "lynker_converter",
]
