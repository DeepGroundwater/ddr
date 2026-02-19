import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler

from ddr import CudaLSTM, ddr_functions, dmc, forcings_reader, kan, streamflow
from ddr._version import __version__
from ddr.routing.utils import select_columns
from ddr.scripts_utils import load_checkpoint, resolve_learning_rate
from ddr.validation import Config, Metrics, hydrograph_loss, plot_time_series, utils, validate_config

log = logging.getLogger(__name__)


def train(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    lstm_nn: CudaLSTM,
    forcings_reader_nn: forcings_reader,
) -> None:
    """Do model training."""
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    start_epoch = 1
    start_mini_batch = 0

    lr = resolve_learning_rate(cfg.experiment.learning_rate, 1)
    kan_optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    lstm_optimizer = torch.optim.Adam(params=lstm_nn.parameters(), lr=lr)

    if cfg.experiment.checkpoint:
        state = load_checkpoint(
            nn,
            cfg.experiment.checkpoint,
            torch.device(cfg.device),
            lstm_nn=lstm_nn,
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
        for param_group in lstm_optimizer.param_groups:
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
        if epoch in cfg.experiment.learning_rate:
            lr = cfg.experiment.learning_rate[epoch]
            log.info(f"Setting learning rate: {lr}")
            for param_group in kan_optimizer.param_groups:
                param_group["lr"] = lr
            for param_group in lstm_optimizer.param_groups:
                param_group["lr"] = lr
        for i, routing_dataclass in enumerate(dataloader, start=0):
            if i < start_mini_batch:
                log.info(f"Skipping mini-batch {i}. Resuming at {start_mini_batch}")
            else:
                start_mini_batch = 0
                routing_model.set_progress_info(epoch=epoch, mini_batch=i)
                kan_optimizer.zero_grad()
                lstm_optimizer.zero_grad()

                streamflow_predictions = flow(
                    routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                )
                attr_names = routing_dataclass.attribute_names
                normalized_attrs = routing_dataclass.normalized_spatial_attributes.to(cfg.device)
                kan_attrs = select_columns(normalized_attrs, list(cfg.kan.input_var_names), attr_names)
                spatial_params = nn(inputs=kan_attrs)
                forcing_data = forcings_reader_nn(
                    routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                )
                lstm_attrs = select_columns(normalized_attrs, list(cfg.cuda_lstm.input_var_names), attr_names)
                lstm_params = lstm_nn(
                    forcings=forcing_data,
                    attributes=lstm_attrs,
                )
                dmc_kwargs = {
                    "routing_dataclass": routing_dataclass,
                    "spatial_parameters": spatial_params,
                    "streamflow": streamflow_predictions,
                    "lstm_params": lstm_params,
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

                pred = filtered_predictions[:, cfg.experiment.warmup :]
                target = filtered_observations[:, cfg.experiment.warmup :]
                loss_cfg = cfg.experiment.loss
                loss = hydrograph_loss(
                    pred=pred,
                    target=target,
                    peak_weight=loss_cfg.peak_weight,
                    baseflow_weight=loss_cfg.baseflow_weight,
                    timing_weight=loss_cfg.timing_weight,
                    peak_percentile=loss_cfg.peak_percentile,
                    baseflow_percentile=loss_cfg.baseflow_percentile,
                    eps=loss_cfg.eps,
                )

                with torch.no_grad():
                    from ddr.validation.losses import _regime_loss, _timing_loss

                    l_peak = _regime_loss(
                        pred, target, target, loss_cfg.peak_percentile, high=True, eps=loss_cfg.eps
                    )
                    l_base = _regime_loss(
                        pred, target, target, loss_cfg.baseflow_percentile, high=False, eps=loss_cfg.eps
                    )
                    l_timing = _timing_loss(pred, target, eps=loss_cfg.eps)
                    log.info(
                        f"Loss components: peak={l_peak.item():.4f}, "
                        f"base={l_base.item():.4f}, timing={l_timing.item():.4f}"
                    )

                log.info("Running backpropagation")

                loss.backward()
                kan_optimizer.step()
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

                n_vals = routing_model.n.detach().cpu()
                log.info(
                    f"Manning's n: median={n_vals.median().item():.4f}, "
                    f"mean={n_vals.mean().item():.4f}, "
                    f"min={n_vals.min().item():.4f}, max={n_vals.max().item():.4f}"
                )

                if "leakance_gate" in spatial_params:
                    gate_raw = spatial_params["leakance_gate"].detach().cpu()
                    gate_on = (gate_raw > 0.5).sum().item()
                    gate_total = gate_raw.numel()
                    log.info(
                        f"Leakance Gate: median={gate_raw.median().item():.4f}, "
                        f"mean={gate_raw.mean().item():.4f}, "
                        f"min={gate_raw.min().item():.4f}, max={gate_raw.max().item():.4f}, "
                        f"ON={gate_on}/{gate_total} ({100 * gate_on / gate_total:.1f}%)"
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
                    kan_optimizer=kan_optimizer,
                    name=cfg.name,
                    saved_model_path=cfg.params.save_path / "saved_models",
                    lstm_nn=lstm_nn,
                    lstm_optimizer=lstm_optimizer,
                )

                # Free batch-specific GPU tensors to prevent VRAM growth
                del streamflow_predictions, spatial_params, dmc_output, daily_runoff
                del loss, filtered_predictions, filtered_observations
                del forcing_data, lstm_params
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
            gate_parameters=config.kan.gate_parameters,
        )
        lstm_nn = CudaLSTM(
            input_var_names=config.cuda_lstm.input_var_names,
            forcing_var_names=config.cuda_lstm.forcing_var_names,
            learnable_parameters=config.cuda_lstm.learnable_parameters,
            hidden_size=config.cuda_lstm.hidden_size,
            num_layers=config.cuda_lstm.num_layers,
            dropout=config.cuda_lstm.dropout,
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
            lstm_nn=lstm_nn,
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
