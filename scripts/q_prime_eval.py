import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from ddr._version import __version__
from ddr.analysis import Metrics
from ddr.dataset import utils

daily_format: str = "%Y/%m/%d"
log = logging.getLogger(__name__)


def print_metrics_summary(metrics: Metrics, save_path: Path, valid_gauges: np.ndarray) -> None:
    """Print formatted metrics summary and save to file

    Parameters
    ----------
    metrics: Metrics
        The metrics object within DDR
    save_path: Path
        The path to save outputs to
    """

    # Calculate summary statistics (removing NaN values)
    def safe_percentile(arr, percentile):
        """Calculate percentile ignoring NaN values"""
        clean_arr = arr[~np.isnan(arr)]
        if len(clean_arr) == 0:
            return np.nan
        return np.percentile(clean_arr, percentile)

    def safe_mean(arr):
        """Calculate mean ignoring NaN values"""
        clean_arr = arr[~np.isnan(arr)]
        if len(clean_arr) == 0:
            return np.nan
        return np.mean(clean_arr)

    bias_stats = {
        "median": safe_percentile(metrics.bias, 50),
        "mean": safe_mean(metrics.bias),
        "q25": safe_percentile(metrics.bias, 25),
        "q75": safe_percentile(metrics.bias, 75),
    }

    flv_stats = {
        "median": safe_percentile(metrics.flv, 50),
        "mean": safe_mean(metrics.flv),
        "q25": safe_percentile(metrics.flv, 25),
        "q75": safe_percentile(metrics.flv, 75),
    }

    fhv_stats = {
        "median": safe_percentile(metrics.fhv, 50),
        "mean": safe_mean(metrics.fhv),
        "q25": safe_percentile(metrics.fhv, 25),
        "q75": safe_percentile(metrics.fhv, 75),
    }

    kge_stats = {
        "median": safe_percentile(metrics.kge, 50),
        "mean": safe_mean(metrics.kge),
        "q25": safe_percentile(metrics.kge, 25),
        "q75": safe_percentile(metrics.kge, 75),
    }

    nse_stats = {
        "median": safe_percentile(metrics.nse, 50),
        "mean": safe_mean(metrics.nse),
        "q25": safe_percentile(metrics.nse, 25),
        "q75": safe_percentile(metrics.nse, 75),
    }

    # Count valid gauges for each metric
    valid_counts = {
        "bias": int(np.sum(~np.isnan(metrics.bias))),
        "flv": int(np.sum(~np.isnan(metrics.flv))),
        "fhv": int(np.sum(~np.isnan(metrics.fhv))),
        "kge": int(np.sum(~np.isnan(metrics.kge))),
        "nse": int(np.sum(~np.isnan(metrics.nse))),
    }

    total_gauges = len(metrics.bias)

    # Print header
    print("\n" + "=" * 80)
    print(" " * 25 + "STREAMFLOW PREDICTION METRICS SUMMARY")
    print("=" * 80)
    print(f"Total Gauges Evaluated: {total_gauges}")
    print("-" * 80)

    # Print metrics table
    print(f"{'METRIC':<12} {'MEDIAN':<10} {'MEAN':<10} {'Q25':<10} {'Q75':<10} {'VALID':<8}")
    print("-" * 80)
    print(
        f"{'Bias':<12} {bias_stats['median']:>9.3f} {bias_stats['mean']:>9.3f} {bias_stats['q25']:>9.3f} {bias_stats['q75']:>9.3f} {valid_counts['bias']:>7d}"
    )
    print(
        f"{'FLV (%)':<12} {flv_stats['median']:>9.2f} {flv_stats['mean']:>9.2f} {flv_stats['q25']:>9.2f} {flv_stats['q75']:>9.2f} {valid_counts['flv']:>7d}"
    )
    print(
        f"{'FHV (%)':<12} {fhv_stats['median']:>9.2f} {fhv_stats['mean']:>9.2f} {fhv_stats['q25']:>9.2f} {fhv_stats['q75']:>9.2f} {valid_counts['fhv']:>7d}"
    )
    print(
        f"{'KGE':<12} {kge_stats['median']:>9.3f} {kge_stats['mean']:>9.3f} {kge_stats['q25']:>9.3f} {kge_stats['q75']:>9.3f} {valid_counts['kge']:>7d}"
    )
    print(
        f"{'NSE':<12} {nse_stats['median']:>9.3f} {nse_stats['mean']:>9.3f} {nse_stats['q25']:>9.3f} {nse_stats['q75']:>9.3f} {valid_counts['nse']:>7d}"
    )
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_data = {
        "timestamp": timestamp,
        "total_gauges": int(total_gauges),
        "evaluation_period": {
            "start": getattr(metrics, "eval_start", "N/A"),
            "end": getattr(metrics, "eval_end", "N/A"),
        },
        "metrics_summary": {
            "bias": {k: float(v) if not np.isnan(v) else None for k, v in bias_stats.items()},
            "flv_percent": {k: float(v) if not np.isnan(v) else None for k, v in flv_stats.items()},
            "fhv_percent": {k: float(v) if not np.isnan(v) else None for k, v in fhv_stats.items()},
            "kge": {k: float(v) if not np.isnan(v) else None for k, v in kge_stats.items()},
            "nse": {k: float(v) if not np.isnan(v) else None for k, v in nse_stats.items()},
        },
        "valid_gauge_counts": valid_counts,
    }

    json_path = save_path / f"metrics_summary_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    detailed_data = {
        "STAID": valid_gauges.tolist(),
        "bias": [float(x) if not np.isnan(x) else None for x in metrics.bias],
        "flv_percent": [float(x) if not np.isnan(x) else None for x in metrics.flv],
        "fhv_percent": [float(x) if not np.isnan(x) else None for x in metrics.fhv],
        "kge": [float(x) if not np.isnan(x) else None for x in metrics.kge],
        "nse": [float(x) if not np.isnan(x) else None for x in metrics.nse],
        "mae": [float(x) if not np.isnan(x) else None for x in metrics.mae],
        "rmse": [float(x) if not np.isnan(x) else None for x in metrics.rmse],
        "corr": [float(x) if not np.isnan(x) else None for x in metrics.corr],
        "pbias": [float(x) if not np.isnan(x) else None for x in metrics.pbias],
    }

    detailed_df = pd.DataFrame(detailed_data)
    csv_path = save_path / f"detailed_metrics_{timestamp}.csv"
    detailed_df.to_csv(csv_path, index=False)
    log.info(f"Metrics summary saved to: {json_path}")
    log.info(f"Detailed metrics saved to: {csv_path}")


def eval(
    cfg: DictConfig,
    streamflow: xr.Dataset,
    observations: xr.Dataset,
    gages_adjacency: zarr.Group,
    basins_df: pd.DataFrame,
) -> None:
    """Evaluated the summed Q` performance against USGS daily observations

    Parameters
    ----------
    cfg : DictConfig
        The config file
    streamflow : xr.Dataset
        The streamflow predictions
    observations : xr.Dataset
        USGS observations
    gages_adjacency : zarr.Group
        All of the gage subsets in COO form
    basins_df : pd.DataFrame
        All gauges to be used in the comparisons
    """
    gauges = [str(_id).zfill(8) for _id in basins_df["STAID"].values]
    valid_gauges = np.array(gauges)[np.isin(gauges, list(gages_adjacency.keys()))]
    log.info(f"{valid_gauges.shape[0]}/{len(gauges)} Gauges found in the hydrofabric")

    eval_daily_time_range = pd.date_range(
        datetime.strptime(cfg.eval.start_time, daily_format),
        datetime.strptime(cfg.eval.end_time, daily_format),
        freq="D",
        inclusive="both",
    )
    conus_divide_ids = streamflow.divide_id.values
    conus_time_range = streamflow.time.values
    time_indices = np.where(np.isin(conus_time_range, eval_daily_time_range))[0]
    preds = np.zeros([len(valid_gauges), len(eval_daily_time_range)], dtype=np.float32)
    target: np.ndarray = observations.sel(
        time=eval_daily_time_range, gage_id=valid_gauges
    ).streamflow.values.astype(np.float32)  # type: ignore
    for i, gauge in tqdm(
        enumerate(valid_gauges), total=len(valid_gauges), desc="Processing gauges", ncols=140
    ):
        basins: np.ndarray = np.array([f"cat-{_id}" for _id in gages_adjacency[gauge]["order"][:]])  # type: ignore
        divide_indices = np.where(np.isin(conus_divide_ids, basins))[0]
        qr = streamflow.isel(time=time_indices, divide_id=divide_indices)["Qr"].values.astype(np.float32)
        preds[i] = qr.sum(axis=0)
    metrics = Metrics(pred=preds, target=target)
    print_metrics_summary(metrics, cfg.params.save_path, valid_gauges)


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="evaluation_config",
)
def main(cfg: DictConfig) -> None:
    """Main function."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    start_time = time.perf_counter()
    try:
        print(f"Checking Summed Q` NSE for streamflow predictions from: {cfg.data_sources.streamflow}")
        streamflow = utils.read_ic(cfg.data_sources.streamflow, region=cfg.s3_region)
        observations = utils.read_ic(cfg.data_sources.observations, region=cfg.s3_region)
        gages_adjacency = zarr.open_group(cfg.data_sources.gages_adjacency)
        basins_df = pd.read_csv(cfg.data_sources.gages)
        eval(
            cfg=cfg,
            streamflow=streamflow,
            observations=observations,
            gages_adjacency=gages_adjacency,
            basins_df=basins_df,
        )

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    finally:
        print("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    os.environ["DDR_VERSION"] = __version__
    main()
