import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

def set_statistics(cfg) -> pd.DataFrame:
    """Creating the necessary statistics for normalizing atributes

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
    attributes_name = "hydrofabric_v2.2"
    statistics_path = Path(cfg.data_sources.statistics)
    statistics_path.mkdir(exist_ok=True)
    stats_file = statistics_path / f"attribute_statistics_{attributes_name}.json"
    if stats_file.exists():
        log.info(f"Reading Attribute Statistics from file: {stats_file.name}")
        df = pd.read_csv(str(stats_file))
        # TODO improve the logic for saving/selecting statistics
    else:
        print("Reading CONUS hydrofabric to construct attribute statistics")
        gdf = gpd.read_file(cfg.data_sources.conus_hydrofabric, layer="divide-attributes")
        json_ = {}
        for attr in cfg.params.attributes:
            data = gdf[attr].values
            json_[attr] = {
                "min": np.min(data, axis=0),
                "max": np.max(data, axis=0),
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "p10": np.percentile(data, 10, axis=0),
                "p90": np.percentile(data, 90, axis=0),
            }
        df = pd.DataFrame(json_)
        df.to_csv(str(stats_file))
    return df
