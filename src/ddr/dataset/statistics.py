import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ddr.validation.validate_configs import Config

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
    stats_file = statistics_path / f"attribute_statistics_{attributes_name}.json"

    if stats_file.exists():
        # TODO improve the logic for saving/selecting statistics
        log.info(f"Reading Attribute Statistics from file: {stats_file.name}")
        # Read JSON file instead of CSV
        with open(stats_file) as f:
            json_ = json.load(f)
        df = pd.DataFrame(json_)
    else:
        log.info("Reading CONUS hydrofabric to construct attribute statistics")
        json_ = {}
        for attr in list(ds.data_vars.keys()):  # Iterating through all variables
            data = ds[attr].values
            json_[attr] = {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "p10": np.percentile(data, 10, axis=0),
                "p90": np.percentile(data, 90, axis=0),
            }
        df = pd.DataFrame(json_)
        # Save as JSON file instead of CSV
        with open(stats_file, "w") as f:
            json.dump(json_, f, indent=2)

    return df
