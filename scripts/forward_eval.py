"""A function which takes a trained model, then evaluates performance on a single, or many, basins"""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SequentialSampler

from ddr._version import __version__
from ddr.dataset import StreamflowReader as streamflow
from ddr.dataset import forward_eval_dataset
from ddr.dataset import utils as ds_utils
from ddr.nn import kan
from ddr.routing.torch_mc import dmc
from ddr.validation import Config, Metrics, plot_time_series, utils, validate_config

log = logging.getLogger(__name__)


def forward_eval(cfg: Config, flow: streamflow, routing_model: dmc, nn: kan):
    """Do model evaluation and get performance metrics."""
    dataset = forward_eval_dataset(cfg=cfg)

    if cfg.experiment.checkpoint:
        file_path = Path(cfg.experiment.checkpoint)
        device = torch.device(cfg.device)
        log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
        state = torch.load(file_path, map_location=device)
        state_dict = state["model_state_dict"]
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        nn.load_state_dict(state_dict)

    else:
        log.warning("Creating new spatial model for evaluation.")

    sampler = SequentialSampler(
        data_source=dataset,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.experiment.batch_size,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=False,  # Cannot drop last as it's needed for eval
    )

    # Initialize collection lists
    all_predictions = []
    all_observations = []
    all_dates = []
    all_gage_ids = []
    all_gage_names = []

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for i, hydrofabric in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)

            streamflow_predictions = flow(hydrofabric=hydrofabric, device=cfg.device, dtype=torch.float32)
            spatial_params = nn(inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device))
            dmc_kwargs = {
                "hydrofabric": hydrofabric,
                "spatial_parameters": spatial_params,
                "streamflow": streamflow_predictions,
            }
            dmc_output = routing_model(**dmc_kwargs)

            num_days = len(dmc_output["runoff"][0][13 : (-11 + cfg.params.tau)]) // 24
            daily_runoff = ds_utils.downsample(
                dmc_output["runoff"][:, 13 : (-11 + cfg.params.tau)],
                rho=num_days,
            )

            nan_mask = hydrofabric.observations.isnull().any(dim="time")
            np_nan_mask = nan_mask.streamflow.values

            filtered_ds = hydrofabric.observations.where(~nan_mask, drop=True)
            filtered_observations = torch.tensor(
                filtered_ds.streamflow.values, device=cfg.device, dtype=torch.float32
            )[:, 1:-1]  # Cutting off days to match with realigned timesteps

            filtered_predictions = daily_runoff[~np_nan_mask]

            gage_ids = dataset.obs_reader.gage_dict["STAID"]
            gage_names = dataset.obs_reader.gage_dict["STANAME"]
            indices = np.where(~np_nan_mask)[0]
            batch_dates = dataset.dates.batch_daily_time_range[1:-1]

            # Store data for this batch
            all_predictions.append(filtered_predictions.detach().cpu().numpy())
            all_observations.append(filtered_observations.detach().cpu().numpy())
            all_dates.append(batch_dates)

            if i == 0:
                all_gage_ids.extend([gage_ids[idx] for idx in indices])
                all_gage_names.extend([gage_names[idx] for idx in indices])

    # Concatenate results across all batches
    all_predictions = np.concatenate(all_predictions, axis=1)
    all_observations = np.concatenate(all_observations, axis=1)
    all_dates_flat = np.concatenate(all_dates)
    time_range = pd.DatetimeIndex(all_dates_flat).drop_duplicates().sort_values()

    # Calculate metrics
    metrics = Metrics(pred=all_predictions, target=all_observations)
    nse = metrics.nse
    rmse = metrics.rmse
    kge = metrics.kge

    utils.log_metrics(nse, rmse, kge)

    # Create time ranges
    date_time_format = "%Y/%m/%d"
    start_time = datetime.strptime(cfg.experiment.start_time, date_time_format).strftime("%Y-%m-%d")
    end_time = datetime.strptime(cfg.experiment.end_time, date_time_format).strftime("%Y-%m-%d")

    # Create xarray datasets following your trainer pattern
    pred_da = xr.DataArray(
        data=all_predictions,
        dims=["gage_ids", "time"],
        coords={"gage_ids": all_gage_ids, "time": time_range},
    )
    obs_da = xr.DataArray(
        data=all_observations,
        dims=["gage_ids", "time"],
        coords={"gage_ids": all_gage_ids, "time": time_range},
    )

    # Create daily dataset
    ds = xr.Dataset(
        data_vars={"predictions": pred_da, "observations": obs_da},
        attrs={
            "description": f"Predictions and obs for time period {start_time} - {end_time}",
            "version": __version__,
            "checkpoint": cfg.experiment.checkpoint,
        },
    )

    # Save daily data to zarr (following your trainer naming pattern)
    daily_zarr_path = cfg.params.save_path / f"{start_time}_{end_time}_validation"
    log.info(f"Saving daily evaluation results to: {daily_zarr_path}")
    ds.to_zarr(daily_zarr_path, mode="w")

    # Plot time series for each gauge
    plots_dir = cfg.params.save_path / "plots"
    plots_dir.mkdir(exist_ok=True)
    log.info(f"Creating time series plots for {len(all_gage_ids)} gauges")
    for i, gage_id in enumerate(all_gage_ids):
        # Get predictions and observations for this gauge
        pred_data = all_predictions[i].squeeze()
        obs_data = all_observations[i].squeeze()

        # Create time range for plotting (matching the data length)
        plotted_dates = (
            time_range if len(time_range) == len(pred_data) else dataset.dates.batch_daily_time_range[1:-1]
        )

        # Calculate NSE for this specific gauge
        gage_nse = nse[i] if not np.isnan(nse[i]) else 0.0

        plot_time_series(
            pred_data,
            obs_data,
            plotted_dates,
            gage_id,
            gage_id,  # Using gage_id for both gage_id and name
            metrics={"nse": gage_nse},
            path=plots_dir / f"gage_{gage_id}_evaluation_plot.png",
            warmup=cfg.experiment.warmup,
        )

    log.info(f"Evaluation complete. Daily results saved to {daily_zarr_path}")
    log.info(f"Dataset contains {len(all_gage_ids)} gauges and {len(time_range)} time steps")


@hydra.main(
    version_base="1.3",
    config_path="../config",
)
def main(cfg: DictConfig) -> None:
    """Main function."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    config = validate_config(cfg)
    start_time = time.perf_counter()
    try:
        nn = kan(
            input_var_names=config.kan.input_var_names,
            learnable_parameters=config.kan.learnable_parameters,
            hidden_size=config.kan.hidden_size,
            num_hidden_layers=config.kan.num_hidden_layers,
            grid=config.kan.grid,
            k=config.kan.k,
            seed=config.seed,
            device=config.device,
        )
        routing_model = dmc(cfg=config, device=cfg.device)
        flow = streamflow(config)
        forward_eval(cfg=config, flow=flow, routing_model=routing_model, nn=nn)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Evaluating DDR with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
