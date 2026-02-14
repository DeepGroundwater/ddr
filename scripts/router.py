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

from ddr import dmc, forcings_reader, kan, leakance_lstm, streamflow
from ddr._version import __version__
from ddr.scripts_utils import compute_daily_runoff, load_checkpoint
from ddr.validation import Config, validate_config

log = logging.getLogger(__name__)


def route_trained_model(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    leakance_nn: leakance_lstm | None = None,
    forcings_reader_nn: forcings_reader | None = None,
) -> None:
    """Route a trained model over a specific amount of defined catchments"""
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    if cfg.experiment.checkpoint:
        load_checkpoint(nn, cfg.experiment.checkpoint, torch.device(cfg.device), leakance_nn=leakance_nn)
    else:
        log.warning("Creating new spatial model for evaluation.")

    nn = nn.eval()
    if leakance_nn is not None:
        leakance_nn.cache_states = True
        leakance_nn = leakance_nn.eval()
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

    assert dataset.routing_dataclass is not None, "Routing dataclass not defined in dataset"
    assert dataset.routing_dataclass.adjacency_matrix is not None, (
        "Routing dataclass adjacency_matrix not defined"
    )

    if cfg.data_sources.target_catchments is not None:
        assert dataset.routing_dataclass.outflow_idx is not None, "Routing dataclass outflow_idx not defined"
        num_outputs = len(dataset.routing_dataclass.outflow_idx)
        log.info(f"Routing for {num_outputs} target catchments")
    elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
        assert dataset.routing_dataclass.outflow_idx is not None, "Routing dataclass outflow_idx not defined"
        num_outputs = len(dataset.routing_dataclass.outflow_idx)
        log.info(f"Routing for {num_outputs} gages")
    else:
        num_outputs = dataset.routing_dataclass.adjacency_matrix.shape[0]
        log.info(f"Routing for {num_outputs} segments (all)")

    num_timesteps = len(dataset.dates.hourly_time_range)
    predictions = np.zeros((num_outputs, num_timesteps), dtype=np.float32)
    zeta_sum_np: np.ndarray | None = None
    q_prime_sum_np: np.ndarray | None = None

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)

            streamflow_predictions = flow(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )
            spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
            dmc_kwargs = {
                "routing_dataclass": routing_dataclass,
                "spatial_parameters": spatial_params,
                "streamflow": streamflow_predictions,
                "carry_state": i > 0,
            }

            if leakance_nn is not None and forcings_reader_nn is not None:
                forcing_data = forcings_reader_nn(
                    routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                )
                leakance_params = leakance_nn(
                    forcings=forcing_data,
                    attributes=routing_dataclass.normalized_spatial_attributes.to(cfg.device),
                )
                dmc_kwargs["leakance_params"] = leakance_params

            dmc_output = routing_model(**dmc_kwargs)
            predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

            if "zeta_sum" in dmc_output:
                batch_zeta = dmc_output["zeta_sum"].cpu().numpy()
                batch_q_prime = dmc_output["q_prime_sum"].cpu().numpy()
                if zeta_sum_np is None:
                    zeta_sum_np = batch_zeta
                    q_prime_sum_np = batch_q_prime
                else:
                    zeta_sum_np += batch_zeta
                    q_prime_sum_np += batch_q_prime

    daily_runoff = compute_daily_runoff(torch.tensor(predictions), cfg.params.tau)
    time_range = dataset.dates.daily_time_range[1:-1]

    pred_da = xr.DataArray(
        data=daily_runoff,
        dims=["catchment_ids", "time"],
        coords={"catchment_ids": dataset.routing_dataclass.divide_ids, "time": time_range},
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
    if zeta_sum_np is not None:
        ds["zeta_sum"] = xr.DataArray(
            data=zeta_sum_np,
            dims=["catchment_ids"],
            coords={"catchment_ids": dataset.routing_dataclass.divide_ids},
            attrs={"units": "m3/s", "long_name": "Cumulative leakance (sum of zeta across timesteps)"},
        )
        ds["q_prime_sum"] = xr.DataArray(
            data=q_prime_sum_np,
            dims=["catchment_ids"],
            coords={"catchment_ids": dataset.routing_dataclass.divide_ids},
            attrs={
                "units": "m3/s",
                "long_name": "Cumulative lateral inflow (sum of q_prime across timesteps)",
            },
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
        leakance_nn = None
        forcings_reader_nn = None
        if config.params.use_leakance:
            leakance_nn = leakance_lstm(
                input_var_names=config.leakance_lstm.input_var_names,
                forcing_var_names=config.leakance_lstm.forcing_var_names,
                hidden_size=config.leakance_lstm.hidden_size,
                num_layers=config.leakance_lstm.num_layers,
                dropout=config.leakance_lstm.dropout,
                seed=config.seed,
                device=config.device,
            )
            forcings_reader_nn = forcings_reader(config)
        routing_model = dmc(cfg=config, device=cfg.device)
        flow = streamflow(config)
        route_trained_model(
            cfg=config,
            flow=flow,
            routing_model=routing_model,
            nn=nn,
            leakance_nn=leakance_nn,
            forcings_reader_nn=forcings_reader_nn,
        )

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
