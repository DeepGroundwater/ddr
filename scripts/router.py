"""A function which takes a trained model, then runs forward simulation at catchment scale."""

import logging
import os
import time
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, SequentialSampler

from ddr._version import __version__
from ddr.dataset import StreamflowReader as streamflow
from ddr.dataset import TestDataset
from ddr.dataset import utils as ds_utils
from ddr.nn import kan
from ddr.routing.torch_mc import dmc
from ddr.validation import Config, validate_config

log = logging.getLogger(__name__)


def route_trained_model(cfg: Config, flow: streamflow, routing_model: dmc, nn: kan):
    """Route a trained model over a specific amount of defined catchments"""
    dataset = TestDataset(cfg=cfg)

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

    nn = nn.eval()
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

    # Create time ranges
    date_time_format = "%Y/%m/%d"
    start_time = datetime.strptime(cfg.experiment.start_time, date_time_format).strftime("%Y-%m-%d")
    end_time = datetime.strptime(cfg.experiment.end_time, date_time_format).strftime("%Y-%m-%d")

    if cfg.data_sources.target_catchments is not None:
        num_outputs = len(dataset.hydrofabric.outflow_idx)
        log.info(f"Routing for {num_outputs} target catchments")
    elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
        num_outputs = len(dataset.hydrofabric.outflow_idx)
        log.info(f"Routing for {num_outputs} gages")
    else:
        num_outputs = dataset.hydrofabric.adjacency_matrix.shape[0]
        log.info(f"Routing for {num_outputs} segments (all)")

    num_timesteps = len(dataset.dates.hourly_time_range)
    predictions = np.zeros((num_outputs, num_timesteps), dtype=np.float32)

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
            predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

    num_days = len(predictions[0][13 : (-11 + cfg.params.tau)]) // 24
    daily_runoff = ds_utils.downsample(
        torch.tensor(predictions[:, (13 + cfg.params.tau) : (-11 + cfg.params.tau)]),
        rho=num_days,
    ).numpy()
    time_range = dataset.dates.daily_time_range[1:-1]

    pred_da = xr.DataArray(
        data=daily_runoff,
        dims=["catchment_ids", "time"],
        coords={"catchment_ids": dataset.hydrofabric.divide_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "Streamflow"},
    )
    attrs = {
        "description": "Predictions and obs for time period",
        "start time": start_time,
        "end time": end_time,
        "version": __version__,
        "model": str(cfg.experiment.checkpoint) if cfg.experiment.checkpoint else "No Trained Model",
    }
    if cfg.data_sources.target_catchments is not None:
        attrs["target catchments"] = str(cfg.data_sources.target_catchments)
    elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
        attrs["evaluation basins file"] = str(cfg.data_sources.gages)
    else:
        attrs["large scale simulation"] = str(True)
    ds = xr.Dataset(
        data_vars={"predictions": pred_da},
        attrs=attrs,
    )
    ds.to_zarr(
        cfg.params.save_path / "chrout.zarr",
        mode="w",
    )

    log.info("Routing complete.")


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
        route_trained_model(cfg=config, flow=flow, routing_model=routing_model, nn=nn)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Forward simulation with DDR version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
