import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import icechunk as ic
import numpy as np
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
    gage_idx: list[np.ndarray] | None = field(default=None)
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
    r = []  # indices_0
    c = []  # indices_1
    output_idx = []
    output_wb = []
    for _id in batch:
        try:
            gauge_root = subsets[_id]
            _r = gauge_root["indices_0"][:].tolist()  # type: ignore
            _c = gauge_root["indices_1"][:].tolist()  # type: ignore
            r.extend(_r)
            c.extend(_c)
            _attrs: dict[str, Any] = dict(gauge_root.attrs)
            output_idx.append(_attrs["gage_idx"])
            output_wb.append(_attrs["gage_wb"])
        except KeyError:
            msg = f"Cannot find gauge {_id} in subsets zarr store. Skipping"
            log.info(msg)
            pass
    shape = tuple(_attrs["shape"])  # type: ignore
    coo = sparse.coo_matrix(
        (
            np.ones(len(r)),
            (
                r,
                c,
            ),
        ),
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


def fill_nans(attr):
    """Fills nan values in a tensor using the mean.

    Parameters
    ----------
    attr : torch.Tensor
        The tensor to fill nan values in.

    Returns
    -------
    torch.Tensor
        The tensor with nan values filled.
    """
    row_means = torch.nanmean(attr)
    nan_mask = torch.isnan(attr)
    attr[nan_mask] = row_means
    return attr


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
