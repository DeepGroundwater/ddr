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

from ddr import ddr_functions, dmc, kan, streamflow
from ddr._version import __version__
from ddr.io.readers import ForcingsReader
from ddr.nn import TemporalPhiKAN
from ddr.scripts_utils import load_checkpoint, resolve_learning_rate
from ddr.validation import Config, Metrics, plot_time_series, utils, validate_config
from ddr.validation.enums import BiasLossFn
from ddr.validation.losses import huber_loss, kge_loss, mass_balance_loss

log = logging.getLogger(__name__)


def train(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    phi_kan: TemporalPhiKAN | None = None,
    q_prime_stats: dict[str, dict[str, float]] | None = None,
    forcings_reader: ForcingsReader | None = None,
) -> None:
    """Do model training."""
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    if cfg.experiment.checkpoint:
        state = load_checkpoint(nn, cfg.experiment.checkpoint, torch.device(cfg.device), phi_kan=phi_kan)
        start_epoch = state["epoch"]
        start_mini_batch = (
            0 if state["mini_batch"] == 0 else state["mini_batch"] + 1
        )  # Start from the next mini-batch
        lr = resolve_learning_rate(cfg.experiment.learning_rate, start_epoch)
    else:
        log.info("Creating new spatial model")
        start_epoch = 1
        start_mini_batch = 0
        lr = cfg.experiment.learning_rate[start_epoch]

    params_to_optimize = list(nn.parameters())
    if phi_kan is not None:
        params_to_optimize += list(phi_kan.parameters())
    optimizer = torch.optim.Adam(params=params_to_optimize, lr=lr)
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

        for i, routing_dataclass in enumerate(dataloader, start=0):
            if i < start_mini_batch:
                log.info(f"Skipping mini-batch {i}. Resuming at {start_mini_batch}")
            else:
                start_mini_batch = 0
                routing_model.set_progress_info(epoch=epoch, mini_batch=i)

                spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))

                if phi_kan is not None:
                    assert q_prime_stats is not None
                    # Get daily Q' for phi-KAN (24x less memory than hourly)
                    q_prime_daily = flow(
                        routing_dataclass=routing_dataclass,
                        device=cfg.device,
                        dtype=torch.float32,
                        use_hourly=True,
                    )
                    divide_ids = routing_dataclass.divide_ids
                    q_mean = torch.tensor(
                        [q_prime_stats.get(str(did), {}).get("mean", 1e-6) for did in divide_ids],
                        device=cfg.device,
                        dtype=torch.float32,
                    )
                    q_std = torch.tensor(
                        [q_prime_stats.get(str(did), {}).get("std", 1e-8) for did in divide_ids],
                        device=cfg.device,
                        dtype=torch.float32,
                    )
                    forcing_tensor = None
                    if forcings_reader is not None:
                        forcing_tensor = forcings_reader(
                            routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                        )
                    # Bias-correct at daily resolution
                    month = dataset.dates.batch_month_tensor_daily.to(cfg.device)
                    q_prime_corrected = phi_kan(
                        q_prime_daily,
                        month=month,
                        forcing=forcing_tensor,
                        q_prime_mean=q_mean,
                        q_prime_std=q_std,
                    )
                    # Interpolate corrected daily → hourly for MC routing
                    T_hourly = len(routing_dataclass.dates.batch_hourly_time_range)
                    streamflow_predictions = q_prime_corrected.repeat_interleave(24, dim=0)[:T_hourly]
                else:
                    streamflow_predictions = flow(
                        routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                    )

                dmc_kwargs = {
                    "routing_dataclass": routing_dataclass,
                    "spatial_parameters": spatial_params,
                    "streamflow": streamflow_predictions,
                }
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

                if phi_kan is not None:
                    pred_gt = filtered_predictions.transpose(0, 1)[cfg.experiment.warmup :]
                    obs_gt = filtered_observations.transpose(0, 1)[cfg.experiment.warmup :]
                    mb_loss = mass_balance_loss(pred_gt, obs_gt)
                    if cfg.bias.loss_fn == BiasLossFn.HUBER:
                        routing_loss = huber_loss(pred_gt, obs_gt)
                    elif cfg.bias.loss_fn == BiasLossFn.KGE:
                        routing_loss = kge_loss(pred_gt, obs_gt)
                    else:
                        routing_loss = mse_loss(pred_gt, obs_gt)
                    loss = cfg.bias.lambda_mass * mb_loss + (1 - cfg.bias.lambda_mass) * routing_loss
                else:
                    loss = huber_loss(
                        filtered_predictions.transpose(0, 1)[cfg.experiment.warmup :],
                        filtered_observations.transpose(0, 1)[cfg.experiment.warmup :],
                    )

                log.info("Running backpropagation")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                np_pred = filtered_predictions.detach().cpu().numpy()
                np_target = filtered_observations.detach().cpu().numpy()
                plotted_dates = dataset.dates.batch_daily_time_range[1:-1]

                metrics = Metrics(pred=np_pred, target=np_target)
                _nse = metrics.nse
                nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
                rmse = metrics.rmse
                kge_metric = metrics.kge
                utils.log_metrics(nse, rmse, kge_metric, epoch=epoch, mini_batch=i)
                if phi_kan is not None:
                    log.info(
                        f"Loss: {loss.item():.6f} (mass_balance: {mb_loss.item():.6f}, "
                        f"{cfg.bias.loss_fn.value}: {routing_loss.item():.6f})"
                    )
                else:
                    log.info(f"Loss: {loss.item()}")

                # Log parameter ranges for all learnable routing parameters
                param_map = {
                    "n": routing_model.n,
                    "q_spatial": routing_model.q_spatial,
                    "top_width": routing_model.top_width,
                    "side_slope": routing_model.side_slope,
                }
                for param_name in cfg.kan.learnable_parameters:
                    param_tensor = param_map.get(param_name)
                    if param_tensor is not None:
                        p = param_tensor.detach().cpu()
                        log.info(
                            f"{param_name}: min={p.min().item():.6f}, "
                            f"median={torch.median(p).item():.6f}, "
                            f"max={p.max().item():.6f}"
                        )

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
                    optimizer=optimizer,
                    name=cfg.name,
                    saved_model_path=cfg.params.save_path / "saved_models",
                    phi_kan=phi_kan,
                )

                # Free autograd graph from this mini-batch so the next
                # iteration doesn't OOM during the phi_kan forward pass.
                del dmc_output, daily_runoff, streamflow_predictions, loss
                del filtered_predictions, filtered_observations
                routing_model.routing_engine.q_prime = None
                routing_model.routing_engine._discharge_t = None
                torch.cuda.empty_cache()


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

        phi_kan = None
        q_prime_stats = None
        if config.bias.enabled:
            phi_kan = TemporalPhiKAN(
                cfg=config.bias,
                seed=config.seed,
                device=config.device,
            )

        routing_model = dmc(cfg=config, device=cfg.device)
        flow = streamflow(config)

        forcings_reader = None
        if config.bias.enabled:
            from ddr.io.statistics import set_streamflow_statistics
            from ddr.validation.enums import PhiInputs

            q_prime_stats = set_streamflow_statistics(config, flow.ds)
            if config.bias.phi_inputs == PhiInputs.FORCING:
                assert config.bias.forcing_var is not None
                forcings_reader = ForcingsReader(config, forcing_var_names=[config.bias.forcing_var])

        train(
            cfg=config,
            flow=flow,
            routing_model=routing_model,
            nn=nn,
            phi_kan=phi_kan,
            q_prime_stats=q_prime_stats,
            forcings_reader=forcings_reader,
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
