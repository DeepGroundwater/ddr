"""Converters for translating between domain-specific IDs and zarr storage format.

Each engine (MERIT, Lynker) has its own ID format. These converters handle the
translation between domain IDs and the integer format used in zarr storage.
"""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class OrderConverter(Protocol):
    """Protocol for order converters."""

    def to_zarr(self, ids: list) -> NDArray[np.int32]:
        """Convert domain IDs to zarr order array."""
        ...

    def from_zarr(self, order: NDArray[np.int32]) -> list:
        """Convert zarr order array back to domain IDs."""
        ...


class MeritOrderConverter:
    """Converter for MERIT COMIDs (integers)."""

    def to_zarr(self, comids: list[int]) -> NDArray[np.int32]:
        """Convert COMID list to zarr order array.

        MERIT COMIDs are already integers, so this is an identity conversion.

        Parameters
        ----------
        comids : list[int]
            List of COMID integers.

        Returns
        -------
        NDArray[np.int32]
            Array of COMIDs as int32.
        """
        return np.array(comids, dtype=np.int32)

    def from_zarr(self, order: NDArray[np.int32]) -> Any:
        """Convert zarr order array back to COMID list.

        Parameters
        ----------
        order : NDArray[np.int32]
            Array of COMIDs from zarr.

        Returns
        -------
        Any
            List of COMID integers.
        """
        return order.tolist()


class LynkerOrderConverter:
    """Converter for Lynker Hydrofabric IDs (wb-* strings)."""

    def to_zarr(self, wb_ids: list[str]) -> NDArray[np.int32]:
        """Convert watershed boundary ID list to zarr order array.

        Extracts the numeric portion from "wb-123" or "ghost-0" format strings.

        Parameters
        ----------
        wb_ids : list[str]
            List of watershed boundary IDs (e.g., ["wb-123", "wb-456"]).

        Returns
        -------
        NDArray[np.int32]
            Array of extracted numeric IDs as int32.
        """
        return np.array(
            [int(float(_id.split("-")[1])) for _id in wb_ids],
            dtype=np.int32,
        )

    def from_zarr(self, order: NDArray[np.int32]) -> list[str]:
        """Convert zarr order array back to watershed boundary ID list.

        Reconstructs "wb-{n}" format strings from numeric values.

        Parameters
        ----------
        order : NDArray[np.int32]
            Array of numeric IDs from zarr.

        Returns
        -------
        list[str]
            List of watershed boundary IDs (e.g., ["wb-123", "wb-456"]).

        Notes
        -----
        Ghost nodes (negative IDs or IDs that were originally "ghost-*") cannot
        be distinguished from regular wb-* IDs after zarr storage. This function
        always reconstructs as "wb-{n}".
        """
        return [f"wb-{n}" for n in order.tolist()]


# Convenience instances
merit_converter = MeritOrderConverter()
lynker_converter = LynkerOrderConverter()

# Geodataset registry - maps geodataset names to converter instances
_GEODATASET_REGISTRY: dict[str, OrderConverter] = {
    "merit": merit_converter,
    "lynker": lynker_converter,
    "hydrofabric_v2.2": lynker_converter,  # Alias for lynker
}


def get_converter(geodataset: str) -> OrderConverter:
    """Get a converter by geodataset name.

    Parameters
    ----------
    geodataset : str
        Name of the geodataset (e.g., "merit", "lynker", "hydrofabric_v2.2").

    Returns
    -------
    OrderConverter
        The converter instance for the specified geodataset.

    Raises
    ------
    ValueError
        If the geodataset name is not registered.

    Examples
    --------
    >>> converter = get_converter("merit")
    >>> converter.to_zarr([12345, 12346])
    array([12345, 12346], dtype=int32)
    """
    if geodataset not in _GEODATASET_REGISTRY:
        available = ", ".join(sorted(_GEODATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown geodataset '{geodataset}'. Available: {available}")
    return _GEODATASET_REGISTRY[geodataset]


def register_converter(geodataset: str, converter: OrderConverter) -> None:
    """Register a custom converter for a geodataset.

    Parameters
    ----------
    geodataset : str
        Name of the geodataset to register.
    converter : OrderConverter
        The converter instance to use for this geodataset.

    Examples
    --------
    >>> class MyConverter:
    ...     def to_zarr(self, ids):
    ...         return np.array(ids, dtype=np.int32)
    ...
    ...     def from_zarr(self, order):
    ...         return order.tolist()
    >>> register_converter("my_geodataset", MyConverter())
    """
    _GEODATASET_REGISTRY[geodataset] = converter


def list_geodatasets() -> list[str]:
    """List all registered geodataset names.

    Returns
    -------
    list[str]
        Sorted list of registered geodataset names.
    """
    return sorted(_GEODATASET_REGISTRY.keys())
