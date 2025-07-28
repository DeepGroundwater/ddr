import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader, RandomSampler

from ddr._version import __version__
from ddr.dataset import StreamflowReader as streamflow
from ddr.dataset import train_dataset
from ddr.dataset import utils as ds_utils
from ddr.nn import kan
from ddr.routing.torch_mc import dmc
from ddr.validation import Config, Metrics, plot_time_series, utils, validate_config

log = logging.getLogger(__name__)


def train(cfg: Config, flow: streamflow, routing_model: dmc, nn: kan):
    """Do model training."""
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    dataset = train_dataset(cfg=cfg)

    if cfg.experiment.checkpoint:
        file_path = Path(cfg.experiment.checkpoint)
        device = torch.device(cfg.device)
        log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
        state = torch.load(file_path, map_location=device)
        state_dict = state["model_state_dict"]
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        nn.load_state_dict(state_dict)
        start_epoch = state["epoch"]
        start_mini_batch = (
            0 if state["mini_batch"] == 0 else state["mini_batch"] + 1
        )  # Start from the next mini-batch

        if start_epoch in cfg.experiment.learning_rate.keys():
            lr = cfg.experiment.learning_rate[start_epoch]
        else:
            key_list = list(cfg.experiment.learning_rate.keys())
            lr = cfg.experiment.learning_rate[key_list[0]]  # defaults to first LR
    else:
        log.info("Creating new spatial model")
        start_epoch = 1
        start_mini_batch = 0
        lr = cfg.experiment.learning_rate[start_epoch]

    optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    sampler = RandomSampler(
        data_source=dataset,
        generator=data_generator,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.experiment.batch_size,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    for epoch in range(start_epoch, cfg.experiment.epochs + 1):
        if epoch in cfg.experiment.learning_rate.keys():
            log.info(f"Setting learning rate: {cfg.experiment.learning_rate[epoch]}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.experiment.learning_rate[epoch]

        for i, hydrofabric in enumerate(dataloader, start=0):
            if i < start_mini_batch:
                log.info(f"Skipping mini-batch {i}. Resuming at {start_mini_batch}")
            else:
                start_mini_batch = 0
                routing_model.set_progress_info(epoch=epoch, mini_batch=i)

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

                loss = mse_loss(
                    input=filtered_predictions.transpose(0, 1)[cfg.experiment.warmup :].unsqueeze(2),
                    target=filtered_observations.transpose(0, 1)[cfg.experiment.warmup :].unsqueeze(2),
                )

                log.info("Running backpropagation")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                np_pred = filtered_predictions.detach().cpu().numpy()
                np_target = filtered_observations.detach().cpu().numpy()
                plotted_dates = dataset.dates.batch_daily_time_range[1:-1]  # type: ignore

                metrics = Metrics(pred=np_pred, target=np_target)
                _nse = metrics.nse
                nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
                rmse = metrics.rmse
                kge = metrics.kge
                utils.log_metrics(nse, rmse, kge, epoch=epoch, mini_batch=i)
                log.info(f"Loss: {loss.item()}")
                log.info(f"Median Mannings Roughness: {torch.median(routing_model.n.detach().cpu()).item()}")

                random_gage = -1  # TODO: scale out when we have more gauges
                plot_time_series(
                    filtered_predictions[-1].detach().cpu().numpy(),
                    filtered_observations[-1].cpu().numpy(),
                    plotted_dates,
                    hydrofabric.observations.gage_id.values[random_gage],
                    hydrofabric.observations.gage_id.values[random_gage],
                    metrics={"nse": nse[-1]},
                    path=cfg.params.save_path / f"plots/epoch_{epoch}_mb_{i}_validation_plot.png",
                    warmup=cfg.experiment.warmup,
                )

                utils.save_state(
                    epoch=epoch,
                    generator=data_generator,
                    mini_batch=i,
                    mlp=nn,
                    optimizer=optimizer,
                    name=cfg.name,
                    saved_model_path=cfg.params.save_path / "saved_models",
                )


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
        train(cfg=config, flow=flow, routing_model=routing_model, nn=nn)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    log.info(f"Training DDR with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
