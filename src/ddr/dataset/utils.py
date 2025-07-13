import logging
from dataclasses import dataclass, field
from pathlib import Path

import icechunk as ic
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
import zarr
import zarr.storage
from scipy import sparse
from scipy.sparse import csc_matrix

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
    transition_matrix: csc_matrix | None = field(default=None)
    divide_ids: np.ndarray | None = field(default=None)


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
        log.info(f"Reading icechunk streamflow predictions from {store}")
        path_parts = store[5:].split("/")
        bucket = path_parts[0]
        prefix = (
            "/".join(path_parts[1:]) if len(path_parts) > 1 else ""
        )  # Join all remaining parts as the prefix
        storage_config = ic.s3_storage(bucket=bucket, prefix=prefix, region=region, anonymous=True)
    else:
        # Assuming Local Icechunk Store
        log.info(f"Reading icechunk streamflow predictions from local disk: {store}")
        storage_config = ic.local_filesystem_storage(store)
    repo = ic.Repository.open(storage_config)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, consolidated=False)
