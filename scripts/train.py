import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, RandomSampler

from ddr import ddr_functions, dmc, kan, streamflow
from ddr._version import __version__
from ddr.routing.utils import aggregate_neighbor_attributes, select_columns
from ddr.scripts_utils import load_checkpoint, resolve_learning_rate
from ddr.validation import Config, Metrics, plot_time_series, utils, validate_config

log = logging.getLogger(__name__)


def train(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
) -> None:
    """Do model training."""
    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    start_epoch = 1
    start_mini_batch = 0

    lr = resolve_learning_rate(cfg.experiment.learning_rate, 1)
    kan_optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)

    if cfg.experiment.checkpoint:
        state = load_checkpoint(
            nn,
            cfg.experiment.checkpoint,
            torch.device(cfg.device),
            kan_optimizer=kan_optimizer,
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

    # Cosine annealing with warm restarts (period doubles each restart)
    kan_scheduler = CosineAnnealingWarmRestarts(kan_optimizer, T_0=len(dataloader), T_mult=2, eta_min=1e-5)

    for epoch in range(start_epoch, cfg.experiment.epochs + 1):
        for i, routing_dataclass in enumerate(dataloader, start=0):
            if i < start_mini_batch:
                log.info(f"Skipping mini-batch {i}. Resuming at {start_mini_batch}")
            else:
                start_mini_batch = 0
                routing_model.set_progress_info(epoch=epoch, mini_batch=i)
                kan_optimizer.zero_grad()

                streamflow_predictions = flow(
                    routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
                )
                attr_names = routing_dataclass.attribute_names
                normalized_attrs = routing_dataclass.normalized_spatial_attributes.to(cfg.device)
                kan_attrs = select_columns(normalized_attrs, list(cfg.kan.input_var_names), attr_names)
                if cfg.kan.use_graph_context:
                    adjacency = routing_dataclass.adjacency_matrix.to(cfg.device)
                    neighbor_attrs = aggregate_neighbor_attributes(kan_attrs, adjacency)
                    kan_attrs = torch.cat([kan_attrs, neighbor_attrs], dim=1)
                spatial_params = nn(inputs=kan_attrs)

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

                pred = filtered_predictions[:, cfg.experiment.warmup :]
                target = filtered_observations[:, cfg.experiment.warmup :]
                loss = torch.nn.functional.mse_loss(pred, target)

                # --- Check forward pass output for NaN ---
                if torch.isnan(dmc_output["runoff"]).any():
                    nan_t = torch.isnan(dmc_output["runoff"]).any(dim=0).nonzero(as_tuple=True)[0][0].item()
                    log.error(f"NaN in routing output at timestep {nan_t} — skipping batch")
                    del streamflow_predictions, spatial_params, dmc_output, daily_runoff
                    del loss, filtered_predictions, filtered_observations
                    routing_model.clear_batch_state()
                    continue

                # --- Check loss for NaN ---
                if torch.isnan(loss):
                    log.error("NaN loss — skipping backward + optimizer step")
                    del streamflow_predictions, spatial_params, dmc_output, daily_runoff
                    del loss, filtered_predictions, filtered_observations
                    routing_model.clear_batch_state()
                    continue

                log.info("Running backpropagation")

                loss.backward()

                # --- NaN gradient guard ---
                kan_grad_norm = torch.nn.utils.clip_grad_norm_(
                    nn.parameters(), max_norm=cfg.experiment.grad_clip_norm
                )
                has_nan_grad = torch.isnan(kan_grad_norm)
                grad_msg = f"Grad norms: KAN={kan_grad_norm.item():.4g}"
                log.info(grad_msg)

                if has_nan_grad:
                    log.error(
                        "NaN in gradients after backward — skipping optimizer step. "
                        "Weights preserved from previous batch."
                    )
                    kan_optimizer.zero_grad()
                else:
                    kan_optimizer.step()
                    kan_scheduler.step()

                np_pred = filtered_predictions.detach().cpu().numpy()
                np_target = filtered_observations.detach().cpu().numpy()
                plotted_dates = dataset.dates.batch_daily_time_range[1:-1]

                metrics = Metrics(pred=np_pred, target=np_target)
                _nse = metrics.nse
                nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
                rmse = metrics.rmse
                kge = metrics.kge
                utils.log_metrics(nse, rmse, kge, epoch=epoch, mini_batch=i)
                log.info(f"Loss: {loss.item():.4f}")

                n_vals = routing_model.n.detach().cpu()
                log.info(
                    f"Manning's n: median={n_vals.median().item():.4f}, "
                    f"mean={n_vals.mean().item():.4f}, "
                    f"min={n_vals.min().item():.4f}, max={n_vals.max().item():.4f}"
                )

                if routing_model.routing_engine.x_storage is not None:
                    x_vals = routing_model.routing_engine.x_storage.detach().cpu()
                    log.info(
                        f"Muskingum X: median={x_vals.median().item():.4f}, "
                        f"mean={x_vals.mean().item():.4f}, "
                        f"min={x_vals.min().item():.4f}, max={x_vals.max().item():.4f}"
                    )

                if routing_model.routing_engine.use_reservoir:
                    pool_elev = routing_model.routing_engine._pool_elevation_t
                    res_mask = routing_model.routing_engine.reservoir_mask
                    if pool_elev is not None and res_mask is not None and res_mask.any():
                        p = pool_elev[res_mask].detach().cpu()
                        log.info(
                            f"Pool elevation: median={p.median():.2f}, range=[{p.min():.2f}, {p.max():.2f}]"
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
                )

                # Free batch-specific GPU tensors to prevent VRAM growth
                del streamflow_predictions, spatial_params, dmc_output, daily_runoff
                del loss, filtered_predictions, filtered_observations
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
            off_parameters=config.kan.off_parameters,
            use_graph_context=config.kan.use_graph_context,
        )
        routing_model = dmc(cfg=config, device=cfg.device)
        flow = streamflow(config)
        train(
            cfg=config,
            flow=flow,
            routing_model=routing_model,
            nn=nn,
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
