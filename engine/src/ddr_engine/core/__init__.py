"""Core module for shared engine functionality.

This module provides generic COO zarr I/O functions and converters for
translating between domain-specific IDs (MERIT COMIDs, Lynker wb-* strings)
and the integer format used in zarr storage.
"""

from .converters import (
    LynkerOrderConverter,
    MeritOrderConverter,
    OrderConverter,
    lynker_converter,
    merit_converter,
)
from .zarr_io import (
    coo_from_zarr_generic,
    coo_to_zarr_generic,
    coo_to_zarr_group_generic,
)

__all__ = [
    # Converters
    "OrderConverter",
    "MeritOrderConverter",
    "LynkerOrderConverter",
    "merit_converter",
    "lynker_converter",
    # Zarr I/O
    "coo_to_zarr_generic",
    "coo_from_zarr_generic",
    "coo_to_zarr_group_generic",
]
