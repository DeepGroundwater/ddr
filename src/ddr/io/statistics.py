import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ddr.validation.configs import Config

log = logging.getLogger(__name__)


def set_statistics(cfg: Config, ds: xr.Dataset) -> pd.DataFrame:
    """Creating the necessary statistics for normalizing attributes

    Parameters
    ----------
    cfg: Config
        The configuration object containing the path to the data sources.
    attributes: zarr.Group
        The zarr.Group object containing attributes.

    Returns
    -------
    pd.DataFrame: A DataFrame containing the statistics for normalizing attributes.
    """
    attributes_name = Path(cfg.data_sources.attributes).name  # gets the name of the attributes store
    statistics_path = Path(cfg.data_sources.statistics)
    statistics_path.mkdir(exist_ok=True)
    stats_file = statistics_path / f"{cfg.geodataset.value}_attribute_statistics_{attributes_name}.json"

    if stats_file.exists():
        # TODO improve the logic for saving/selecting statistics
        log.info(f"Reading Attribute Statistics from file: {stats_file.name}")
        # Read JSON file instead of CSV
        with open(stats_file) as f:
            json_ = json.load(f)
        df = pd.DataFrame(json_)
    else:
        log.info(f"Reading {cfg.geodataset.value} attributes to construct statistics")
        json_ = {}
        for attr in list(ds.data_vars.keys()):  # Iterating through all variables
            data = ds[attr].values
            json_[attr] = {
                "min": np.nanmin(data, axis=0),
                "max": np.nanmax(data, axis=0),
                "mean": np.nanmean(data, axis=0),
                "std": np.nanstd(data, axis=0),
                "p10": np.nanpercentile(data, 10, axis=0),
                "p90": np.nanpercentile(data, 90, axis=0),
            }
        df = pd.DataFrame(json_)
        # Save as JSON file instead of CSV
        with open(stats_file, "w") as f:
            json.dump(json_, f, indent=2)

    return df
