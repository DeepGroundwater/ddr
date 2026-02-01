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
