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

from ddr import ddr_functions, dmc, forcings_reader, kan, leakance_lstm, streamflow
from ddr._version import __version__
from ddr.scripts_utils import load_checkpoint, resolve_learning_rate
from ddr.validation import Config, Metrics, plot_time_series, utils, validate_config

log = logging.getLogger(__name__)


def train(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    leakance_nn: leakance_lstm | None = None,
    forcings_reader_nn: forcings_reader | None = None,
) -> None:
    """Do model training."""
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    lr = cfg.experiment.learning_rate[1]
    start_epoch = 1
    start_mini_batch = 0

    kan_optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    lstm_optimizer: torch.optim.Optimizer | None = None
    if leakance_nn is not None:
        lstm_optimizer = torch.optim.Adadelta(params=leakance_nn.parameters())

    if cfg.experiment.checkpoint:
        state = load_checkpoint(
            nn,
            cfg.experiment.checkpoint,
            torch.device(cfg.device),
            leakance_nn=leakance_nn,
            kan_optimizer=kan_optimizer,
            lstm_optimizer=lstm_optimizer,
        )
        start_epoch = state["epoch"]
        start_mini_batch = (
            0 if state["mini_batch"] == 0 else state["mini_batch"] + 1
        )  # Start from the next mini-batch
        lr = resolve_learning_rate(cfg.experiment.learning_rate, start_epoch)
        for param_group in kan_optimizer.param_groups:
            param_group["lr"] = lr
    else:
        log.info("Creating new spatial model")
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
            log.info(f"Setting KAN learning rate: {cfg.experiment.learning_rate[epoch]}")
            for param_group in kan_optimizer.param_groups:
                param_group["lr"] = cfg.experiment.learning_rate[epoch]

        for i, routing_dataclass in enumerate(dataloader, start=0):
            if i < start_mini_batch:
                log.info(f"Skipping mini-batch {i}. Resuming at {start_mini_batch}")
            else:
                start_mini_batch = 0
                routing_model.set_progress_info(epoch=epoch, mini_batch=i)
                kan_optimizer.zero_grad()
                if lstm_optimizer is not None:
                    lstm_optimizer.zero_grad()

                streamflow_predictions = flow(
                    routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                )
                spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
                dmc_kwargs = {
                    "routing_dataclass": routing_dataclass,
                    "spatial_parameters": spatial_params,
                    "streamflow": streamflow_predictions,
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

                num_days = len(dmc_output["runoff"][0][13 : (-11 + cfg.params.tau)]) // 24
                daily_runoff = ddr_functions.downsample(
                    dmc_output["runoff"][:, 13 : (-11 + cfg.params.tau)],
                    rho=num_days,
                )

                nan_mask = routing_dataclass.observations.isnull().any(dim="time")
                np_nan_mask = nan_mask.streamflow.values

                filtered_ds = routing_dataclass.observations.where(~nan_mask, drop=True)
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
                kan_optimizer.step()
                if lstm_optimizer is not None:
                    lstm_optimizer.step()

                np_pred = filtered_predictions.detach().cpu().numpy()
                np_target = filtered_observations.detach().cpu().numpy()
                plotted_dates = dataset.dates.batch_daily_time_range[1:-1]

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
                    routing_dataclass.observations.gage_id.values[random_gage],
                    routing_dataclass.observations.gage_id.values[random_gage],
                    metrics={"nse": nse[-1]},
                    path=cfg.params.save_path / f"plots/epoch_{epoch}_mb_{i}_validation_plot.png",
                    warmup=cfg.experiment.warmup,
                )

                utils.save_state(
                    epoch=epoch,
                    generator=data_generator,
                    mini_batch=i,
                    mlp=nn,
                    kan_optimizer=kan_optimizer,
                    name=cfg.name,
                    saved_model_path=cfg.params.save_path / "saved_models",
                    leakance_nn=leakance_nn,
                    lstm_optimizer=lstm_optimizer,
                )

                # Free batch-specific GPU tensors to prevent VRAM growth
                del streamflow_predictions, spatial_params, dmc_output, daily_runoff
                del loss, filtered_predictions, filtered_observations
                if leakance_nn is not None:
                    del forcing_data, leakance_params
                routing_model.clear_batch_state()


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
        train(
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
    log.info(f"Training DDR with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
