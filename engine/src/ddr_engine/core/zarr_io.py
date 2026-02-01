"""Generic zarr I/O utilities for COO adjacency matrices.

This module provides engine-agnostic functions for reading and writing
sparse COO matrices to zarr format. Engine-specific behavior is handled
via order converters.

Binsparse COO Format (zarr v3)
==============================

This module implements a zarr-based storage format for sparse COO matrices,
inspired by the binsparse specification (https://github.com/ivirshup/binsparse-python).

Arrays
------
Each zarr group contains the following arrays:

- ``indices_0`` : int32
    Row indices of non-zero elements (downstream segment indices).
- ``indices_1`` : int32
    Column indices of non-zero elements (upstream segment indices).
- ``values`` : uint8
    Matrix values (typically 1 for adjacency).
- ``order`` : int32
    Topological sort order as domain-specific IDs. For MERIT, these are
    COMIDs (integers). For Lynker, these are numeric portions of wb-* strings.

Attributes
----------
The zarr group stores the following attributes:

- ``format`` : str
    Always "COO" to indicate coordinate format.
- ``shape`` : list[int, int]
    Matrix dimensions [rows, cols].
- ``data_types`` : dict
    Dtype strings for indices_0, indices_1, and values.
- ``gage_catchment`` : int | str (gauge subsets only)
    Origin catchment ID for the gauge.
- ``gage_idx`` : int (gauge subsets only)
    Index of the gauge catchment in the CONUS adjacency matrix.

Matrix Structure
----------------
The adjacency matrix is lower triangular, where A[i, j] = 1 indicates that
flow goes from segment j (column) to segment i (row). This structure ensures
topological ordering: upstream segments always have lower indices than
downstream segments.

Order Converters
----------------
Different hydrofabric datasets use different ID formats:

- **MERIT**: Integer COMIDs (stored directly as int32)
- **Lynker Hydrofabric**: String IDs like "wb-123" (numeric portion extracted)

Use the appropriate converter when reading/writing:

- ``merit_converter``: For MERIT Hydro datasets
- ``lynker_converter``: For Lynker Hydrofabric v2.2

Example
-------
Writing a COO matrix:

    >>> from ddr_engine import coo_to_zarr_generic, merit_converter
    >>> from scipy import sparse
    >>> coo = sparse.coo_matrix(...)
    >>> ts_order = [12345, 12346, 12347]  # MERIT COMIDs
    >>> coo_to_zarr_generic(coo, ts_order, Path("output.zarr"), merit_converter)

Reading a COO matrix:

    >>> from ddr_engine import coo_from_zarr_generic, merit_converter
    >>> coo, ts_order = coo_from_zarr_generic(Path("output.zarr"), merit_converter)
    >>> print(ts_order)  # [12345, 12346, 12347]
"""

from pathlib import Path
from typing import Any

import zarr
from scipy import sparse

from .converters import OrderConverter


def coo_to_zarr_generic(
    coo: sparse.coo_matrix,
    ts_order: list,
    out_path: Path,
    converter: OrderConverter,
) -> None:
    """
    Save a COO adjacency matrix to a zarr group.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list
        Topological sort order of flowpaths (domain-specific IDs).
    out_path : Path
        Path to save the zarr group.
    converter : OrderConverter
        Converter to translate domain IDs to zarr storage format.
    """
    store = zarr.storage.LocalStore(root=out_path)
    root = zarr.create_group(store=store)

    zarr_order = converter.to_zarr(ts_order)

    indices_0 = root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order = root.create_array(name="order", shape=zarr_order.shape, dtype=zarr_order.dtype)

    indices_0[:] = coo.row
    indices_1[:] = coo.col
    values[:] = coo.data
    order[:] = zarr_order

    root.attrs["format"] = "COO"
    root.attrs["shape"] = list(coo.shape)
    root.attrs["data_types"] = {
        "indices_0": str(coo.row.dtype),
        "indices_1": str(coo.col.dtype),
        "values": str(coo.data.dtype),
    }

    print(f"Adjacency matrix written to zarr at {out_path}")


def coo_from_zarr_generic(
    zarr_path: Path,
    converter: OrderConverter,
) -> tuple[sparse.coo_matrix, list]:
    """
    Load a COO adjacency matrix from a zarr group.

    Parameters
    ----------
    zarr_path : Path
        Path to the zarr group.
    converter : OrderConverter
        Converter to translate zarr storage format back to domain IDs.

    Returns
    -------
    tuple[sparse.coo_matrix, list]
        The COO matrix and topological order (in domain-specific ID format).
    """
    root = zarr.open_group(store=zarr_path, mode="r")

    row = root["indices_0"][:]
    col = root["indices_1"][:]
    data = root["values"][:]
    shape = tuple(root.attrs["shape"])
    order_array = root["order"][:]

    coo = sparse.coo_matrix((data, (row, col)), shape=shape)
    ts_order = converter.from_zarr(order_array)

    return coo, ts_order


def coo_to_zarr_group_generic(
    coo: sparse.coo_matrix,
    ts_order: list,
    origin: Any,
    gauge_root: zarr.Group,
    mapping: dict,
    converter: OrderConverter,
) -> None:
    """
    Save a COO matrix to a zarr group for a gauge subset.

    Parameters
    ----------
    coo : sparse.coo_matrix
        Lower triangular adjacency matrix.
    ts_order : list
        Domain-specific IDs in topological sort order.
    origin : Any
        The origin ID of the gauge (domain-specific type).
    gauge_root : zarr.Group
        The zarr group for the subset COO matrix.
    mapping : dict
        Mapping of domain ID to its position in the CONUS array.
    converter : OrderConverter
        Converter to translate domain IDs to zarr storage format.
    """
    zarr_order = converter.to_zarr(ts_order)

    indices_0 = gauge_root.create_array(name="indices_0", shape=coo.row.shape, dtype=coo.row.dtype)
    indices_1 = gauge_root.create_array(name="indices_1", shape=coo.col.shape, dtype=coo.row.dtype)
    values = gauge_root.create_array(name="values", shape=coo.data.shape, dtype=coo.data.dtype)
    order_array = gauge_root.create_array(name="order", shape=zarr_order.shape, dtype=zarr_order.dtype)

    indices_0[:] = coo.row
    indices_1[:] = coo.col
    values[:] = coo.data
    order_array[:] = zarr_order

    gauge_root.attrs["format"] = "COO"
    gauge_root.attrs["shape"] = list(coo.shape)
    gauge_root.attrs["gage_catchment"] = origin if isinstance(origin, int | str) else str(origin)
    gauge_root.attrs["gage_idx"] = int(mapping[origin])
    gauge_root.attrs["data_types"] = {
        "indices_0": str(coo.row.dtype),
        "indices_1": str(coo.col.dtype),
        "values": str(coo.data.dtype),
    }
