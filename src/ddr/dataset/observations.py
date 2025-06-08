import logging
from pathlib import Path

import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr

from ddr.dataset.Dates import Dates

log = logging.getLogger(__name__)


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
        "COMID",
        "edge_intersection",
        "zone_edge_id",
        "zone_edge_uparea",
        "zone_edge_vs_gage_area_difference",
        "drainage_area_percent_error",
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


class IcechunkUSGSReader:
    """An object to handle reads to the USGS Icechunk Store"""

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs["cfg"]
        if "s3" in self.cfg.data_sources.observations:
            # Getting the bucket and prefix from an s3:// URI
            log.info(f"Reading Icechunk USGS Observations from {self.cfg.data_sources.observations}")
            bucket = self.cfg.data_sources.observations[5:].split("/")[0]
            prefix = self.cfg.data_sources.observations[5:].split("/")[1]
            storage_config = ic.s3_storage(
                bucket=bucket, prefix=prefix, region=self.cfg.s3_region, anonymous=True
            )
        else:
            # Assuming Local Icechunk Store
            log.info("Reading Icechunk USGS Observations from local disk")
            storage_config = ic.local_filesystem_storage(str(self.cfg.data_sources.observations))
        repo = ic.Repository.open(storage_config)
        session = repo.readonly_session("main")
        self.file_path = session.store
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
        ds = (
            xr.open_zarr(self.file_path, consolidated=False)
            .sel(gage_id=padded_gage_ids)
            .isel(time=dates.numerical_time_range)
        )
        return ds
