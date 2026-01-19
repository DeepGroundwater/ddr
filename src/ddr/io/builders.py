"""A file to handle all building from data sources"""

import logging
from typing import Any

import numpy as np
import rustworkx as rx
import xarray as xr
import zarr
import zarr.storage
from scipy import sparse

from ddr.geodatazoo.dataclasses import Dates

log = logging.getLogger(__name__)


def _build_network_graph(conus_adjacency: dict[str, Any]) -> tuple[rx.PyDiGraph, dict[int, int], np.ndarray]:
    """Build a rustworkx directed graph from the CONUS adjacency matrix.

    Parameters
    ----------
    conus_adjacency : dict
        Zarr group containing COO matrix data (indices_0, indices_1, values, order)

    Returns
    -------
    tuple[rx.PyDiGraph, dict[int, int], np.ndarray]
        graph : The directed graph where edges point downstream
        hf_id_to_node : Mapping from hydrofabric ID to graph node index
        hf_ids : Array of hydrofabric IDs in topological order
    """
    hf_ids = conus_adjacency["order"][:]
    rows = conus_adjacency["indices_0"][:]
    cols = conus_adjacency["indices_1"][:]
    n = len(hf_ids)

    # Create graph - edges go from col (upstream) to row (downstream)
    graph = rx.PyDiGraph(check_cycle=False, node_count_hint=n, edge_count_hint=len(rows))

    # Add all nodes
    graph.add_nodes_from(list(range(n)))

    # Create mapping from hf_id to node index
    hf_id_to_node = {int(hf_id): idx for idx, hf_id in enumerate(hf_ids)}

    # Add edges (upstream -> downstream)
    # In lower triangular matrix: row >= col, so col is upstream of row
    edges = [(int(col), int(row)) for col, row in zip(cols, rows, strict=False)]
    graph.add_edges_from_no_data(edges)

    return graph, hf_id_to_node, hf_ids


def construct_network_matrix(
    batch: list[str], subsets: zarr.Group
) -> tuple[sparse.coo_matrix, list[str], list[str]]:
    """Creates a sparse coo matrix from many subset basins from `engine/gages_adjacency.py`

    Parameters
    ----------
    batch : list[str]
        The gauges contained in the current batch
    subsets : zarr.Group
        The subset basins from `engine/gages_adjacency.py`

    Returns
    -------
    tuple[sparse.coo_matrix, list[str], list[str]]
        The sparse network matrix and lists of the idx of the gauge and its wb id

    Raises
    ------
    KeyError
        Cannot find a gauge from the batch in the gages_adjacency.zarr Group
    """
    coordinates: set[tuple[int, int]] = set()
    output_idx: list[str] = []
    output_wb: list[str] = []
    _attrs: dict[str, Any] = {}
    for _id in batch:
        try:
            gauge_root = subsets[_id]
        except KeyError:
            msg = f"Cannot find gage {_id} in subsets zarr store. Skipping"
            log.info(msg)
        _r: list[int] = gauge_root["indices_0"][:].tolist()
        _c: list[int] = gauge_root["indices_1"][:].tolist()
        for row, col in zip(_r, _c, strict=False):
            coordinates.add((row, col))
        try:
            _attrs = dict(gauge_root.attrs)
            output_idx.append(_attrs["gage_idx"])
            output_wb.append(_attrs["gage_catchment"])
        except KeyError:
            msg = f"Cannot find gauge attributes for gage {_id}. Skipping"
            log.info(msg)
    if coordinates:
        rows_tuple, cols_tuple = zip(*coordinates, strict=False)
        rows: list[int] = list(rows_tuple)
        cols: list[int] = list(cols_tuple)
    else:
        raise ValueError("No coordinate-pairs found. Cannot construct a matrix")
    shape = tuple(_attrs["shape"])
    coo = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=shape,
    )
    return coo, output_idx, output_wb


def create_hydrofabric_observations(
    dates: Dates,
    gage_ids: np.ndarray,
    observations: xr.Dataset,
) -> xr.Dataset:
    """Select a subset of hydrofabric observations.

    Parameters
    ----------
    dates : Dates
        Object of dates to select from the observations.
    gage_ids : np.ndarray
        Array of gage IDs to select from the observations.
    observations : xr.Dataset
        The observations dataset.
    """
    ds = observations.sel(time=dates.batch_daily_time_range, gage_id=gage_ids).compute()
    return ds
