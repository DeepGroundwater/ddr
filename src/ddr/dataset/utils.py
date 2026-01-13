import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import icechunk as ic
import numpy as np
import rustworkx as rx
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
import zarr.storage
from scipy import sparse

from ddr.dataset.Dates import Dates

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    """Hydrofabric data class."""

    adjacency_matrix: torch.Tensor | None = field(default=None)
    spatial_attributes: torch.Tensor | None = field(default=None)
    length: torch.Tensor | None = field(default=None)
    slope: torch.Tensor | None = field(default=None)
    side_slope: torch.Tensor | None = field(default=None)
    top_width: torch.Tensor | None = field(default=None)
    x: torch.Tensor | None = field(default=None)
    dates: Dates | None = field(default=None)
    normalized_spatial_attributes: torch.Tensor | None = field(default=None)
    observations: xr.Dataset | None = field(default=None)
    divide_ids: np.ndarray | None = field(default=None)
    outflow_idx: list[np.ndarray] | None = field(
        default=None
    )  # Has to be list[np.ndarray] since idx are ragged arrays
    gage_wb: list[str] | None = field(default=None)


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


def read_coo(path: Path, key: str) -> tuple[sparse.coo_matrix, zarr.Group]:
    """Reading a Binsparse specified coo matrix from zarr.

    Parameters
    ----------
    path : Path
        Path to zarr store.
    key : str
    Gage ID to read from the zarr store.
    """
    if path.exists():
        store = zarr.storage.LocalStore(root=path, read_only=True)
        root = zarr.open_group(store, mode="r")
        try:
            gauge_root = root[key]
        except KeyError as e:
            raise KeyError(f"Cannot find key: {key}") from e

        attrs = dict(gauge_root.attrs)
        shape = tuple(attrs["shape"])

        coo = sparse.coo_matrix(
            (
                gauge_root["values"][:],
                (
                    gauge_root["indices_0"][:],
                    gauge_root["indices_1"][:],
                ),
            ),
            shape=shape,
        )
        return coo, gauge_root
    else:
        raise FileNotFoundError(f"Cannot find file: {path}")


def read_zarr(path: Path) -> zarr.Group:
    """Reads a zarr group from store.

    Parameters
    ----------
    path : Path
        Path to zarr store.

    Returns
    -------
    zarr.Group
        The saved group object
    """
    if path.exists():
        store = zarr.storage.LocalStore(root=path, read_only=True)
        root = zarr.open_group(store, mode="r")
        return root
    else:
        raise FileNotFoundError(f"Cannot find file: {path}")


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
    coordinates = set()  # (indices_0, indices_1)
    output_idx = []
    output_wb = []
    for _id in batch:
        try:
            gauge_root = subsets[_id]
            _r = gauge_root["indices_0"][:].tolist()  # type: ignore
            _c = gauge_root["indices_1"][:].tolist()  # type: ignore
            for row, col in zip(_r, _c, strict=False):
                coordinates.add((row, col))
            _attrs: dict[str, Any] = dict(gauge_root.attrs)
            output_idx.append(_attrs["gage_idx"])
            output_wb.append(_attrs["gage_wb"])
        except KeyError:
            msg = f"Cannot find gauge {_id} in subsets zarr store. Skipping"
            log.info(msg)
            pass
    if coordinates:
        rows, cols = zip(*coordinates, strict=False)
        rows = list(rows)
        cols = list(cols)
    else:
        raise ValueError("No coordinate-pairs found. Cannot construct a matrix")
    shape = tuple(_attrs["shape"])  # type: ignore
    coo = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)),
        shape=shape,
    )
    return coo, output_idx, output_wb


def downsample(data: torch.Tensor, rho: int) -> torch.Tensor:
    """Downsamples data from hourly to daily resolution.

    Parameters
    ----------
    data : torch.Tensor
        The data to downsample.
    rho : int
        The number of days to downsample to.

    Returns
    -------
    torch.Tensor
        The downsampled daily data.
    """
    downsampled_data = F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)
    return downsampled_data


def naninfmean(arr) -> np.ndarray:
    """Finds the mean of an array if there are both nan and inf values

    Parameters
    ----------
    attr : torch.Tensor
        The tensor to fill nan values in.

    Returns
    -------
    np.ndarray
        The array with nan values filled.
    """
    finite_vals = arr[np.isfinite(arr)]
    return np.mean(finite_vals) if len(finite_vals) > 0 else np.nan


def fill_nans(attr, row_means=None):
    """Fills nan values in a tensor using the mean.

    Parameters
    ----------
    attr : torch.Tensor
        The tensor to fill nan values in.
    row_means : torch.Tensor, optional
        Per-row means to use for filling. If None, uses global mean.

    Returns
    -------
    torch.Tensor
        The tensor with nan values filled.
    """
    original_shape = attr.shape
    if row_means is None:
        result = torch.where(torch.isnan(attr), torch.nanmean(attr), attr)
    else:
        row_means = row_means.to(attr.device)

        # Ensuring row_means will work if we have multiple rows and row_means needs to be broadcast across them
        if attr.dim() == 2 and row_means.dim() == 1 and len(row_means) > 1:
            row_means = row_means.unsqueeze(-1)

        result = torch.where(torch.isnan(attr), row_means, attr)

    # Ensure output shape matches input shape
    return result.view(original_shape)


def read_ic(store: str, region="us-east-2") -> xr.Dataset:
    """Reads an icechunk repo either from a local store or an S3 bucket

    Parameters
    ----------
    store: str
        The path to the icechunk store

    Returns
    -------
    xr.Dataset
        The icechunk store via xarray.Dataset
    """
    if "s3://" in store:
        # Getting the bucket and prefix from an s3:// URI
        log.info(f"Reading icechunk repo from {store}")
        path_parts = store[5:].split("/")
        bucket = path_parts[0]
        prefix = (
            "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
        )  # Join all remaining parts as the prefix
        storage_config = ic.s3_storage(bucket=bucket, prefix=prefix, region=region, anonymous=True)
    else:
        # Assuming Local Icechunk Store
        log.info(f"Reading icechunk store from local disk: {store}")
        storage_config = ic.local_filesystem_storage(store)
    repo = ic.Repository.open(storage_config)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)


def _build_network_graph(conus_adjacency: dict) -> tuple[rx.PyDiGraph, dict[int, int], np.ndarray]:
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

    return graph, hf_id_to_node
