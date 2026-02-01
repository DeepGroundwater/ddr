"""DDR Engine - Hydrofabric data preparation tools.

This package provides tools for building sparse COO adjacency matrices from
hydrofabric datasets (MERIT Hydro, Lynker Hydrofabric v2.2).

Core I/O Functions
------------------
The following generic I/O functions work with any hydrofabric dataset:

- ``coo_to_zarr_generic``: Write a COO matrix to zarr
- ``coo_from_zarr_generic``: Read a COO matrix from zarr
- ``coo_to_zarr_group_generic``: Write a gauge subset COO matrix

Order Converters
----------------
Use the appropriate converter for your dataset:

- ``merit_converter``: For MERIT Hydro (integer COMIDs)
- ``lynker_converter``: For Lynker Hydrofabric (wb-* string IDs)

Example
-------
>>> from ddr_engine import coo_to_zarr_generic, merit_converter
>>> from scipy import sparse
>>> coo = sparse.coo_matrix(...)
>>> coo_to_zarr_generic(coo, ts_order, Path("output.zarr"), merit_converter)
"""

from ._version import __version__
from .core import (
    coo_from_zarr_generic,
    coo_to_zarr_generic,
    coo_to_zarr_group_generic,
    lynker_converter,
    merit_converter,
)

__all__ = [
    "__version__",
    # Core I/O functions
    "coo_to_zarr_generic",
    "coo_from_zarr_generic",
    "coo_to_zarr_group_generic",
    # Order converters
    "merit_converter",
    "lynker_converter",
]
