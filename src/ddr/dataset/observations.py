import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from tqdm import tqdm

from ddr.dataset.Dates import Dates

log = logging.getLogger(__name__)

def convert_ft3_s_to_m3_s(flow_rates_ft3_s: np.ndarray) -> np.ndarray:
    """
    Convert a 2D tensor of flow rates
    from cubic feet per second (ft³/s) to cubic meters per second (m³/s).
    """
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
    try:
        df = pd.read_csv(gage_info_path)
        df["STAID"] = [str(_id).zfill(8) for _id in df["STAID"].values]
        if "Unnamed: 0" in df.columns: 
            df = df.drop(["Unnamed: 0"], axis=1)
        if "index" in df.columns: 
            df = df.drop(["index"], axis=1)

        gages_dict = {
            field: df[field].values.tolist()
            for field in df.columns
        }
        return gages_dict
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {gage_info_path}") from e

class ZarrUSGSReader():
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = kwargs["cfg"]
        self.gage_dict = read_gage_info(Path(self.cfg.data_sources.training_basins))

    def read_data(self, dates: Dates) -> xr.Dataset:
        padded_gage_idx = [str(gage_idx).zfill(8) for gage_idx in self.gage_dict["STAID"]]
        y = np.zeros([len(padded_gage_idx), len(dates.daily_time_range)])
        root = zarr.open_group(Path(self.cfg.data_sources.observations), mode="r")
        for idx, gage_id in enumerate(
            tqdm(
                padded_gage_idx,
                desc="\rReading Zarr USGS observations",
                ncols=140,
                ascii=True,
            )
        ):
            try:
                data_obs = root[gage_id]
                y[idx, :] = data_obs[dates.numerical_time_range]
            except KeyError as e:
                log.error(f"Cannot find zarr store: {e}")
        _observations = convert_ft3_s_to_m3_s(y)
        ds = xr.Dataset(
            {"streamflow": (["gage_id", "time"], _observations)},
            coords={
                "gage_id": padded_gage_idx,
                "time": dates.daily_time_range,
            },
        )
        return ds
