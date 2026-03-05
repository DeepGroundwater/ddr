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


def set_streamflow_statistics(cfg: Config, ds: xr.Dataset) -> dict[str, dict[str, float]]:
    """Compute and cache per-basin mean/std of lateral inflow Q' for z-score normalization.

    Lazily computes statistics on first run, then reads from cached JSON.
    Statistics are computed over the full time range in the dataset to be
    stable across train/test splits.

    Parameters
    ----------
    cfg : Config
        Configuration object (uses cfg.data_sources.statistics for cache path
        and cfg.geodataset for filename prefix).
    ds : xr.Dataset
        The streamflow dataset containing 'Qr' variable with dims (time, divide_id).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of divide_id (string) -> {"mean": float, "std": float}.
    """
    store_name = Path(cfg.data_sources.streamflow).name
    statistics_path = Path(cfg.data_sources.statistics)
    statistics_path.mkdir(exist_ok=True)
    stats_file = statistics_path / f"{cfg.geodataset.value}_streamflow_statistics_{store_name}.json"

    if stats_file.exists():
        log.info(f"Reading streamflow statistics from file: {stats_file.name}")
        with open(stats_file) as f:
            cached: dict[str, dict[str, float]] = json.load(f)
            return cached

    log.info("Computing per-basin Q' statistics (first run, this may take a few minutes)...")
    divide_ids = ds.divide_id.values
    num_divides = len(divide_ids)
    chunk_size = 1000
    result: dict[str, dict[str, float]] = {}

    for start in range(0, num_divides, chunk_size):
        end = min(start + chunk_size, num_divides)
        chunk = ds["Qr"].isel(divide_id=slice(start, end)).values  # (time, chunk_size)
        chunk_ids = divide_ids[start:end]

        means = np.nanmean(chunk, axis=0)
        stds = np.nanstd(chunk, axis=0)

        for i, did in enumerate(chunk_ids):
            mean_val = float(means[i]) if not np.isnan(means[i]) else 1e-6
            std_val = float(stds[i]) if not np.isnan(stds[i]) and stds[i] > 0 else 1e-8
            result[str(did)] = {"mean": mean_val, "std": std_val}

        log.info(f"  Processed {end}/{num_divides} divide_ids")

    with open(stats_file, "w") as f:
        json.dump(result, f)
    log.info(f"Saved streamflow statistics to {stats_file.name}")

    return result
