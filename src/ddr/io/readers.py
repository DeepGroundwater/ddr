"""A file to handle all reading from data sources"""

import logging
import warnings
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


def read_gage_info(gage_info_path: Path) -> dict[str, list]:
    """Reads gage information from a specified file.

    Parameters
    ----------
    gage_info_path : Path
        The path to the CSV file containing gage information.

    Returns
    -------
    dict[str, list]
        A dictionary containing gage information. Required keys: STAID, STANAME,
        DRAIN_SQKM, LAT_GAGE, LNG_GAGE. Optional keys (included when present in CSV):
        COMID, COMID_DRAIN_SQKM, ABS_DIFF, COMID_UNITAREA_SQKM.

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
    optional_columns = [
        "COMID",
        "COMID_DRAIN_SQKM",
        "ABS_DIFF",
        "COMID_UNITAREA_SQKM",
        "DA_VALID",
        "FLOW_SCALE",
    ]

    try:
        df = pd.read_csv(gage_info_path, delimiter=",", dtype={"STAID": str})

        if not set(expected_column_names).issubset(set(df.columns)):
            missing_headers = set(expected_column_names) - set(df.columns)
            if len(missing_headers) == 1 and "STANAME" in missing_headers:
                df["STANAME"] = df["STAID"]
            else:
                raise KeyError(f"The CSV file is missing the following headers: {list(missing_headers)}")

        df["STAID"] = df["STAID"].astype(str).str.zfill(8)

        out = {
            field: df[field].tolist() if field == "STANAME" else df[field].values.tolist()
            for field in expected_column_names
            if field in df.columns
        }

        for col in optional_columns:
            if col in df.columns:
                out[col] = df[col].values.tolist()

        return out
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {gage_info_path}") from e


def filter_gages_by_area_threshold(
    gage_ids: np.ndarray,
    gage_dict: dict[str, list],
    threshold: float,
) -> tuple[np.ndarray, int]:
    """Filter gage IDs by absolute drainage area difference.

    Parameters
    ----------
    gage_ids : np.ndarray
        Array of STAID strings
    gage_dict : dict
        Dict from read_gage_info() — must contain "STAID" and "ABS_DIFF"
    threshold : float
        Maximum absolute area difference in km²

    Returns
    -------
    tuple[np.ndarray, int]
        Filtered gage IDs and count of removed gages

    Raises
    ------
    KeyError
        If gage_dict doesn't contain "ABS_DIFF" key
    """
    if "ABS_DIFF" not in gage_dict:
        raise KeyError("gage_dict must contain 'ABS_DIFF' key for area threshold filtering")

    staid_to_abs_diff = {
        str(staid): abs_diff
        for staid, abs_diff in zip(gage_dict["STAID"], gage_dict["ABS_DIFF"], strict=False)
    }

    keep_mask = np.array([staid_to_abs_diff.get(gid, float("inf")) <= threshold for gid in gage_ids])
    filtered = gage_ids[keep_mask]
    n_removed = len(gage_ids) - len(filtered)
    return filtered, n_removed


def filter_gages_by_da_valid(
    gage_ids: np.ndarray,
    gage_dict: dict[str, list],
) -> tuple[np.ndarray, int]:
    """Filter gage IDs using pre-computed DA_VALID column.

    Parameters
    ----------
    gage_ids : np.ndarray
        Array of STAID strings
    gage_dict : dict
        Dict from read_gage_info() — must contain "STAID" and "DA_VALID"

    Returns
    -------
    tuple[np.ndarray, int]
        Filtered gage IDs and count of removed gages

    Raises
    ------
    KeyError
        If gage_dict doesn't contain "DA_VALID" key
    """
    if "DA_VALID" not in gage_dict:
        raise KeyError("gage_dict must contain 'DA_VALID' key for DA_VALID filtering")

    staid_to_valid = {
        str(staid): valid for staid, valid in zip(gage_dict["STAID"], gage_dict["DA_VALID"], strict=False)
    }

    keep_mask = np.array([staid_to_valid.get(gid, False) for gid in gage_ids])
    filtered = gage_ids[keep_mask]
    n_removed = len(gage_ids) - len(filtered)
    return filtered, n_removed


def compute_flow_scale_factor(
    drain_sqkm: float,
    comid_drain_sqkm: float,
    comid_unitarea_sqkm: float,
) -> float:
    """Compute the fraction of Q' to keep at a gage's catchment segment.

    When a gage sits partway through a catchment (not at the outlet), the modeled
    lateral inflow Q' is too large. This function computes a scaling factor [0, 1]
    that reduces Q' proportionally to the area mismatch.

    Parameters
    ----------
    drain_sqkm : float
        Gage drainage area (DRAIN_SQKM).
    comid_drain_sqkm : float
        Total drainage area of the COMID the gage is mapped to (COMID_DRAIN_SQKM).
    comid_unitarea_sqkm : float
        Local (unit) catchment area of that COMID (COMID_UNITAREA_SQKM).

    Returns
    -------
    float
        Scaling factor in [0, 1]. Returns 1.0 (no scaling) when the gage drains
        at least as much area as the COMID, or when inputs are degenerate.
    """
    import math

    if math.isnan(drain_sqkm) or math.isnan(comid_drain_sqkm) or math.isnan(comid_unitarea_sqkm):
        return 1.0
    if comid_unitarea_sqkm <= 0:
        return 1.0
    diff = drain_sqkm - comid_drain_sqkm
    if diff >= 0:
        return 1.0
    if abs(diff) >= comid_unitarea_sqkm:
        return 1.0
    return (comid_unitarea_sqkm - abs(diff)) / comid_unitarea_sqkm


def build_flow_scale_tensor(
    batch: list[str],
    gage_dict: dict[str, list],
    gage_compressed_indices: list[int],
    num_segments: int,
) -> torch.Tensor:
    """Build a per-segment flow scaling tensor for a batch of gages.

    Parameters
    ----------
    batch : list[str]
        STAID strings for gages in this batch (same order as gage_compressed_indices).
    gage_dict : dict[str, list]
        Dict from ``read_gage_info()`` — must contain ``STAID``.
        If ``COMID_DRAIN_SQKM`` or ``COMID_UNITAREA_SQKM`` are absent,
        returns an all-ones tensor (graceful skip).
    gage_compressed_indices : list[int]
        Compressed segment index for each gage in *batch*.
    num_segments : int
        Total number of segments in the compressed network.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` with 1.0 everywhere except gage segments
        that need scaling.
    """
    import math

    flow_scale = torch.ones(num_segments, dtype=torch.float32)

    staid_list = [str(s) for s in gage_dict["STAID"]]
    staid_to_idx = {s: i for i, s in enumerate(staid_list)}

    # Fast path: use pre-computed FLOW_SCALE from CSV when available
    if "FLOW_SCALE" in gage_dict:
        for gage_staid, seg_idx in zip(batch, gage_compressed_indices, strict=False):
            lookup_key = str(gage_staid).zfill(8)
            dict_idx = staid_to_idx.get(lookup_key)
            if dict_idx is None:
                continue
            val = gage_dict["FLOW_SCALE"][dict_idx]
            if isinstance(val, float) and math.isnan(val):
                continue  # keeps default 1.0
            flow_scale[seg_idx] = val
        return flow_scale

    # Fallback: compute from raw columns
    if "COMID_DRAIN_SQKM" not in gage_dict or "COMID_UNITAREA_SQKM" not in gage_dict:
        return flow_scale

    for gage_staid, seg_idx in zip(batch, gage_compressed_indices, strict=False):
        lookup_key = str(gage_staid).zfill(8)
        dict_idx = staid_to_idx.get(lookup_key)
        if dict_idx is None:
            continue
        factor = compute_flow_scale_factor(
            drain_sqkm=gage_dict["DRAIN_SQKM"][dict_idx],
            comid_drain_sqkm=gage_dict["COMID_DRAIN_SQKM"][dict_idx],
            comid_unitarea_sqkm=gage_dict["COMID_UNITAREA_SQKM"][dict_idx],
        )
        flow_scale[seg_idx] = factor

    return flow_scale


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
        routing_dataclass = kwargs["routing_dataclass"]
        device = kwargs.get("device", "cpu")  # defaulting to a CPU tensor
        dtype = kwargs.get("dtype", torch.float32)  # defaulting to float32
        use_hourly = kwargs.get("use_hourly", False)
        valid_divide_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(routing_dataclass.divide_ids):
            if divide_id in self.divide_id_to_index:
                valid_divide_indices.append(self.divide_id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                log.info(f"{divide_id} missing from the streamflow dataset")

        assert len(valid_divide_indices) != 0, "No valid divide IDs found in this batch. Throwing error"

        _ds = self.ds.isel(time=routing_dataclass.dates.numerical_time_range, divide_id=valid_divide_indices)[
            "Qr"
        ]

        if use_hourly is False:
            _ds = _ds.interp(
                time=routing_dataclass.dates.batch_hourly_time_range,
                method="nearest",
            )
        streamflow_data = (
            _ds.compute().values.astype(np.float32).T
        )  # Transposing to (num_timesteps, num_features)

        # Creating an output tensor where we're filling any missing data with minimum flow
        output = torch.full(
            (streamflow_data.shape[0], len(routing_dataclass.divide_ids)),
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


class ForcingsReader(torch.nn.Module):
    """A class to read meteorological forcings (P, PET, Temp) from an icechunk store."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        assert cfg.data_sources.forcings is not None, "data_sources.forcings must be set"
        assert cfg.cuda_lstm is not None, "cuda_lstm config required for ForcingsReader"
        self.ds = read_ic(cfg.data_sources.forcings, region=cfg.s3_region)
        self.forcing_var_names = list(cfg.cuda_lstm.forcing_var_names)
        # Index Lookup Dictionary
        self.divide_id_to_index = {divide_id: idx for idx, divide_id in enumerate(self.ds.divide_id.values)}
        # Compute time offset: forcings store may not start at 1980-01-01 (the Dates origin)
        first_time = pd.Timestamp(self.ds.time.values[0])
        origin = pd.Timestamp("1980-01-01")
        self._time_offset = (first_time - origin).days

        # Load or compute forcing statistics for z-score normalization
        from ddr.io.statistics import set_forcing_statistics

        forcing_stats = set_forcing_statistics(cfg, self.ds)
        self.forcing_means = torch.tensor(
            [forcing_stats[var]["mean"] for var in self.forcing_var_names],
            dtype=torch.float32,
        )
        self.forcing_stds = torch.tensor(
            [forcing_stats[var]["std"] for var in self.forcing_var_names],
            dtype=torch.float32,
        )

    def forward(self, **kwargs: Any) -> torch.Tensor:
        """Read forcing variables for the given routing dataclass.

        Returns
        -------
        torch.Tensor
            Forcing data with shape (T_daily, N, num_forcing_vars), dtype float32.
        """
        routing_dataclass = kwargs["routing_dataclass"]
        device = kwargs.get("device", "cpu")
        dtype = kwargs.get("dtype", torch.float32)

        valid_divide_indices = []
        divide_idx_mask: list[int] = []
        missing_count = 0

        for i, divide_id in enumerate(routing_dataclass.divide_ids):
            if divide_id in self.divide_id_to_index:
                valid_divide_indices.append(self.divide_id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                missing_count += 1

        if missing_count > 0:
            total = len(routing_dataclass.divide_ids)
            log.info(f"{missing_count}/{total} divide IDs missing from forcings (zero-filled)")

        assert len(valid_divide_indices) != 0, "No valid divide IDs found in forcings store"

        # Convert Dates numerical_time_range (days from 1980-01-01) to forcings store indices
        forcings_indices = routing_dataclass.dates.numerical_time_range - self._time_offset

        N = len(routing_dataclass.divide_ids)
        T = len(forcings_indices)
        num_vars = len(self.forcing_var_names)

        # Read each forcing variable and stack
        var_tensors = []
        for var_name in self.forcing_var_names:
            _ds = self.ds[var_name].isel(time=forcings_indices, divide_id=valid_divide_indices)
            data = _ds.compute().values.astype(np.float32).T  # (T, num_valid)
            # Fill NaN with per-basin temporal mean; if entire basin is NaN, fall back to 0.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                basin_means = np.nanmean(data, axis=0, keepdims=True)  # (1, num_valid)
            nan_mask = np.isnan(data)
            data = np.where(nan_mask, basin_means, data)
            data = np.nan_to_num(data, nan=0.0)
            var_tensor = torch.full((T, N), 0.0, dtype=dtype)
            var_tensor[:, divide_idx_mask] = torch.tensor(data, dtype=dtype)
            var_tensors.append(var_tensor)

        # Stack: [T, N, num_vars]
        output = torch.stack(var_tensors, dim=-1).to(device)
        assert output.shape == (T, N, num_vars)
        # Z-score normalize: (x - mean) / std
        output = (output - self.forcing_means.to(device)) / self.forcing_stds.to(device)
        return output


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
