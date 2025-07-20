import logging
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from ddr._version import __version__
from ddr.analysis.metrics import Metrics
from ddr.analysis.plots import plot_time_series
from ddr.analysis.utils import save_state
from ddr.dataset.streamflow import StreamflowReader as streamflow
from ddr.dataset.train_dataset import train_dataset
from ddr.dataset.utils import downsample
from ddr.nn.kan import kan
from ddr.routing.torch_mc import dmc

log = logging.getLogger(__name__)


def _set_seed(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.np_seed)
    random.seed(cfg.seed)


def train(cfg, flow, routing_model, nn):
    """Do model training."""
    dataset = train_dataset(cfg=cfg)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        shuffle=cfg.train.shuffle,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    if cfg.train.checkpoint:
        file_path = Path(cfg.train.checkpoint)
        log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
        state = torch.load(file_path, map_location=cfg.device)
        state_dict = state["model_state_dict"]

        nn.load_state_dict(state_dict)
        torch.set_rng_state(state["rng_state"])
        start_epoch = state["epoch"]
        # start_mini_batch = 0 if state["mini_batch"] == 0 else state["mini_batch"] + 1  # Start from the next mini-batch

        if torch.cuda.is_available() and "cuda_rng_state" in state:
            torch.cuda.set_rng_state(state["cuda_rng_state"])
        if start_epoch in cfg.train.learning_rate.keys():
            lr = cfg.train.learning_rate[start_epoch]
        else:
            key_list = list(cfg.train.learning_rate.keys())
            lr = cfg.train.learning_rate[key_list[-1]]
    else:
        log.info("Creating new spatial model")
        start_epoch = 1
        # start_mini_batch = 0
        lr = cfg.train.learning_rate[str(0)]

    optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        routing_model.epoch = epoch
        for i, hydrofabric in enumerate(dataloader, start=0):
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
            daily_runoff = downsample(
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
                input=filtered_predictions.transpose(0, 1)[cfg.train.warmup :].unsqueeze(2),
                target=filtered_observations.transpose(0, 1)[cfg.train.warmup :].unsqueeze(2),
            )

            log.info("Running backpropagation")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            np_pred = filtered_predictions.detach().cpu().numpy()
            np_target = filtered_observations.detach().cpu().numpy()
            plotted_dates = dataset.dates.batch_daily_time_range[1:-1]  # type: ignore

            metrics = Metrics(pred=np_pred, target=np_target)
            pred_nse = metrics.nse
            pred_nse_filtered = pred_nse[~np.isinf(pred_nse) & ~np.isnan(pred_nse)]
            median_nse = torch.tensor(pred_nse_filtered).median()

            random_gage = -1  # TODO: scale out when we have more gauges
            plot_time_series(
                filtered_predictions[-1].detach().cpu().numpy(),
                filtered_observations[-1].cpu().numpy(),
                plotted_dates,
                dataset.obs_reader.gage_dict["STAID"][random_gage],
                dataset.obs_reader.gage_dict["STANAME"][random_gage],
                metrics={"nse": pred_nse[-1]},
                path=cfg.params.save_path / f"plots/epoch_{epoch}_mb_{i}_validation_plot.png",
                warmup=cfg.train.warmup,
            )

            save_state(
                epoch=epoch,
                mini_batch=i,
                mlp=nn,
                optimizer=optimizer,
                name=cfg.name,
                saved_model_path=cfg.params.save_path / "saved_models",
            )

            log.info(f"Loss: {loss.item()}")
            log.info(f"Median NSE: {median_nse}")
            log.info(f"Median Mannings Roughness: {torch.median(routing_model.n.detach().cpu()).item()}")

        if epoch in cfg.train.learning_rate.keys():
            log.info(f"Updating learning rate: {cfg.train.learning_rate[epoch]}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = cfg.train.learning_rate[epoch]


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="training_config",
)
def main(cfg: DictConfig) -> None:
    """Main function."""
    _set_seed(cfg=cfg)
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)
    start_time = time.perf_counter()
    try:
        nn = kan(
            input_var_names=cfg.kan.input_var_names,
            learnable_parameters=cfg.kan.learnable_parameters,
            hidden_size=cfg.kan.hidden_size,
            num_hidden_layers=cfg.kan.num_hidden_layers,
            grid=cfg.kan.grid,
            k=cfg.kan.k,
            seed=cfg.seed,
            device=cfg.device,
        )
        routing_model = dmc(cfg=cfg, device=cfg.device)
        flow = streamflow(cfg)
        train(cfg=cfg, flow=flow, routing_model=routing_model, nn=nn)

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
