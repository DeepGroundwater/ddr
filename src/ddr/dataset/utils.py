from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse
import zarr
import zarr.storage

def read_coo(path: Path, key: str) -> tuple[sparse.coo_matrix, zarr.Group]:
    """Reading a Binsparse specified coo matrix"""
    if path.exists():
        store = zarr.storage.LocalStore(root=path, read_only=True)
        root = zarr.open_group(store, mode="r")
        try:
            gauge_root = root[key]
        except KeyError as e:
            raise e(f"Cannot find key: {key}")
        
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
    """Downsamples from hourly to daily data using torch.nn.functional.interpolate

    Parameters
    ----------
    data : torch.Tensor
        The data to downsample
    rho : int
        The number of days to downsample to

    Returns
    -------
    torch.Tensor
        The downsampled daily data
    """
    downsampled_data = F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)
    return downsampled_data


def fill_nans(attr):
    """Fills the nan values in a tensor using the mean
    """
    row_means = torch.nanmean(attr)
    nan_mask = torch.isnan(attr)
    attr[nan_mask] = row_means
    return attr
