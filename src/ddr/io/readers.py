"""A file to handle all reading from data sources"""

import logging
from pathlib import Path
from typing import Any

import icechunk as ic
import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr
import zarr.storage
from scipy import sparse

from ddr.geodatazoo.dataclasses import Dates
from ddr.validation.configs import Config, GeoDataset

log = logging.getLogger(__name__)


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


def convert_ft3_s_to_m3_s(flow_rates_ft3_s: np.ndarray) -> np.ndarray:
    """Convert a 2D tensor of flow rates from cubic feet per second (ft³/s) to cubic meters per second (m³/s)."""
    conversion_factor = 0.0283168
    return flow_rates_ft3_s * conversion_factor


def read_gage_info(gage_info_path: Path) -> dict[str, list[str]]:
    """Reads gage information from a specified file.

    Parameters
    ----------
    gage_info_path : Path
        The path to the CSV file containing gage information.

    Returns
    -------
    Dict[str, List[str]]: A dictionary containing the gage information.

    Raises
    ------
        FileNotFoundError: If the specified file path is not found.
        KeyError: If the CSV file is missing any of the expected column headers.
    """
    expected_column_names = [
        "STAID",
        "STANAME",
        "DRAIN_SQKM",
        "LAT_GAGE",
        "LNG_GAGE",
    ]

    try:
        df = pd.read_csv(gage_info_path, delimiter=",")

        if not set(expected_column_names).issubset(set(df.columns)):
            missing_headers = set(expected_column_names) - set(df.columns)
            if len(missing_headers) == 1 and "STANAME" in missing_headers:
                df["STANAME"] = df["STAID"]
            else:
                raise KeyError(f"The CSV file is missing the following headers: {list(missing_headers)}")

        df["STAID"] = df["STAID"].astype(str)

        out = {
            field: df[field].tolist() if field == "STANAME" else df[field].values.tolist()
            for field in expected_column_names
            if field in df.columns
        }
        return out
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {gage_info_path}") from e


def naninfmean(arr: np.ndarray) -> np.floating[Any]:
    """Finds the mean of an array if there are both nan and inf values

    Parameters
    ----------
    arr : np.ndarray
        The array to compute the mean of.

    Returns
    -------
    np.floating
        The mean of finite values, or np.nan if no finite values exist.
    """
    finite_vals = arr[np.isfinite(arr)]
    return np.mean(finite_vals) if len(finite_vals) > 0 else np.nan


def fill_nans(attr: torch.Tensor, row_means: torch.Tensor | None = None) -> torch.Tensor:
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


def read_ic(store: str, region: str = "us-east-2") -> xr.Dataset:
    """Reads an icechunk repo either from a local store or an S3 bucket

    Parameters
    ----------
    store: str
        The path to the icechunk store
    region: str
        The AWS region for S3 storage

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


class StreamflowReader(torch.nn.Module):
    """A class to read streamflow from a local zarr store or icechunk repo"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = read_ic(self.cfg.data_sources.streamflow, region=self.cfg.s3_region)
        # Index Lookup Dictionary
        self.divide_id_to_index = {divide_id: idx for idx, divide_id in enumerate(self.ds.divide_id.values)}

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """The forward function of the module for generating streamflow values

        Returns
        -------
        torch.Tensor
            streamflow predictions for the given timesteps and divides

        Raises
        ------
        IndexError
            The basin you're searching for is not in the sample
        """
        hydrofabric = kwargs["hydrofabric"]
        device = kwargs.get("device", "cpu")  # defaulting to a CPU tensor
        dtype = kwargs.get("dtype", torch.float32)  # defaulting to float32
        use_hourly = kwargs.get("use_hourly", False)
        valid_divide_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(hydrofabric.divide_ids):
            if divide_id in self.divide_id_to_index:
                valid_divide_indices.append(self.divide_id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                log.info(f"{divide_id} missing from the streamflow dataset")

        assert len(valid_divide_indices) != 0, "No valid divide IDs found in this batch. Throwing error"

        _ds = self.ds.isel(time=hydrofabric.dates.numerical_time_range, divide_id=valid_divide_indices)["Qr"]

        if use_hourly is False:
            _ds = _ds.interp(
                time=hydrofabric.dates.batch_hourly_time_range,
                method="nearest",
            )
        streamflow_data = (
            _ds.compute().values.astype(np.float32).T
        )  # Transposing to (num_timesteps, num_features)

        # Creating an output tensor where we're filling any missing data with minimum flow
        output = torch.full(
            (streamflow_data.shape[0], len(hydrofabric.divide_ids)),
            fill_value=0.001,
            device=device,
            dtype=dtype,
        )
        output[:, divide_idx_mask] = torch.tensor(streamflow_data, device=device, dtype=dtype)
        return output


class IcechunkUSGSReader:
    """An object to handle reads to the USGS Icechunk Store"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds = read_ic(self.cfg.data_sources.observations, region=self.cfg.s3_region)
        if self.cfg.data_sources.gages is None:
            raise ValueError("data_sources.gages must be set for IcechunkUSGSReader")
        self.gage_dict = read_gage_info(Path(self.cfg.data_sources.gages))

    def read_data(self, dates: Dates) -> xr.Dataset:
        """A function to read data from icechunk given specific dates

        Parameters
        ----------
        dates: Dates
            The Dates object

        Returns
        -------
        xr.Dataset
            The observations from the required gages for the requested timesteps
        """
        padded_gage_ids = [str(gage_id).zfill(8) for gage_id in self.gage_dict["STAID"]]
        ds_ = self.ds.sel(gage_id=padded_gage_ids).isel(time=dates.numerical_time_range)
        return ds_


class AttributesReader(torch.nn.Module):
    """A class to read attributes from a local zarr store or icechunk repo"""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.attributes_list = list(
            self.cfg.kan.input_var_names
        )  # Have to cast to list for this to work with xarray

        if cfg.geodataset == GeoDataset.LYNKER_HYDROFABRIC.value:
            self.ds = read_ic(self.cfg.data_sources.attributes, region=self.cfg.s3_region)
            # Index Lookup Dictionary
            self.divide_id_to_index = {
                divide_id: idx for idx, divide_id in enumerate(self.ds.divide_id.values)
            }
        elif cfg.geodataset == GeoDataset.MERIT.value:
            self.ds = xr.open_mfdataset(self.cfg.data_sources.attributes)
            self.divide_id_to_index = {COMID: idx for idx, COMID in enumerate(self.ds.COMID.values)}

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """The forward function of the module for generating attributes

        Returns
        -------
        torch.Tensor
            attributes for the given divides in the shape (n_attributes, n_divides)

        Raises
        ------
        IndexError
            The basin you're searching for is not in the sample
        """
