"""Core module for shared engine functionality.

This module provides COO zarr I/O functions and converters for translating
between domain-specific IDs (MERIT COMIDs, Lynker wb-* strings) and the
integer format used in zarr storage.

Primary API
-----------
The recommended functions auto-detect or accept geodataset names:

- ``coo_to_zarr``: Write a COO matrix (takes geodataset name)
- ``coo_from_zarr``: Read a COO matrix (auto-detects geodataset)
- ``coo_to_zarr_group``: Write a gauge subset (takes geodataset name)

Converter Registry
------------------
Register custom converters with ``register_converter()`` and list available
geodatasets with ``list_geodatasets()``.
"""

from .converters import (
    LynkerOrderConverter,
    MeritOrderConverter,
    OrderConverter,
    get_converter,
    list_geodatasets,
    lynker_converter,
    merit_converter,
    register_converter,
)
from .zarr_io import (
    coo_from_zarr,
    coo_from_zarr_generic,
    coo_to_zarr,
    coo_to_zarr_generic,
    coo_to_zarr_group,
    coo_to_zarr_group_generic,
)

__all__ = [
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
    # Converter classes and instances
    "OrderConverter",
    "MeritOrderConverter",
    "LynkerOrderConverter",
    "merit_converter",
    "lynker_converter",
]
