"""Train DDR, then immediately evaluate on the test period.

Combines scripts/train.py and the test loop into a single entrypoint.
Training uses the config as-is (e.g. 1981-1995, batch_size=64 gages, rho=90).
After training, the script auto-discovers the last checkpoint and evaluates on
the MERIT test period (1995-2010, batch_size=182 days, rho=None).

Usage:
    uv run python scripts/train_and_test.py --config-name=merit_training_config
"""

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

from ddr import dmc, kan, streamflow
from ddr._version import __version__
from ddr.scripts_utils import compute_daily_runoff, load_checkpoint
from ddr.validation import Config, Metrics, utils, validate_config
from scripts.train import train

log = logging.getLogger(__name__)


def _test(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
) -> None:
    """Do model evaluation and get performance metrics."""
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    if cfg.experiment.checkpoint:
        load_checkpoint(nn, cfg.experiment.checkpoint, torch.device(cfg.device))
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
        drop_last=False,
    )

    warmup = cfg.experiment.warmup
    assert dataset.routing_dataclass is not None, "Routing dataclass not defined in dataset"
    assert dataset.routing_dataclass.observations is not None, "Observations not defined in dataset"
    observations = dataset.routing_dataclass.observations.streamflow.values

    date_time_format = "%Y/%m/%d"
    start_time = datetime.strptime(cfg.experiment.start_time, date_time_format).strftime("%Y-%m-%d")
    end_time = datetime.strptime(cfg.experiment.end_time, date_time_format).strftime("%Y-%m-%d")
    all_gage_ids = dataset.routing_dataclass.observations.gage_id.values
    predictions = np.zeros([len(all_gage_ids), len(dataset.dates.hourly_time_range)])

    with torch.no_grad():
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
            dmc_output = routing_model(**dmc_kwargs)
            predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

    daily_runoff = compute_daily_runoff(torch.tensor(predictions), cfg.params.tau)
    daily_obs = observations[:, 1:-1]
    time_range = dataset.dates.daily_time_range[1:-1]

    pred_da = xr.DataArray(
        data=daily_runoff,
        dims=["gage_ids", "time"],
        coords={"gage_ids": all_gage_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "Streamflow"},
    )
    obs_da = xr.DataArray(
        data=daily_obs,
        dims=["gage_ids", "time"],
        coords={"gage_ids": all_gage_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "Observed Streamflow"},
    )
    ds = xr.Dataset(
        data_vars={"predictions": pred_da, "observations": obs_da},
        attrs={
            "description": "Predictions and obs for time period",
            "start time": start_time,
            "end time": end_time,
            "version": __version__,
            "evaluation basins file": str(cfg.data_sources.gages),
            "model": str(cfg.experiment.checkpoint) if cfg.experiment.checkpoint else "No Trained Model",
        },
    )
    ds.to_zarr(
        cfg.params.save_path / "model_test.zarr",
        mode="w",
    )
    metrics = Metrics(pred=ds.predictions.values[:, warmup:], target=ds.observations.values[:, warmup:])
    _nse = metrics.nse
    nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
    rmse = metrics.rmse
    kge = metrics.kge
    utils.log_metrics(nse, rmse, kge)
    log.info(
        "Test run complete. Please run examples/eval/evaluate.ipynb to generate performance plots / metrics"
    )


def _find_last_checkpoint(saved_models_dir: Path) -> Path:
    """Return the most recently modified .pt file in saved_models_dir."""
    pts = sorted(saved_models_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {saved_models_dir}")
    return pts[-1]


@hydra.main(version_base="1.3", config_path="../config")
def main(cfg: DictConfig) -> None:
    """Train, then evaluate."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)
    (cfg.params.save_path / "saved_models").mkdir(exist_ok=True)

    config = validate_config(cfg)

    start_time = time.perf_counter()
    try:
        # --- Phase 1: Train ---
        nn_model = kan(
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

        train(cfg=config, flow=flow, routing_model=routing_model, nn=nn_model)

        train_elapsed = time.perf_counter() - start_time
        log.info(f"Training complete in {train_elapsed / 60:.2f} minutes")

        # --- Phase 2: Test ---
        checkpoint = _find_last_checkpoint(config.params.save_path / "saved_models")
        log.info(f"Evaluating with checkpoint: {checkpoint.name}")

        test_config = config.model_copy(
            update={
                "mode": "testing",
                "experiment": config.experiment.model_copy(
                    update={
                        "start_time": "1995/10/01",
                        "end_time": "2010/09/30",
                        "batch_size": 182,
                        "rho": None,
                        "checkpoint": checkpoint,
                        "epochs": 1,
                    }
                ),
            }
        )

        nn_model = kan(
            input_var_names=test_config.kan.input_var_names,
            learnable_parameters=test_config.kan.learnable_parameters,
            hidden_size=test_config.kan.hidden_size,
            num_hidden_layers=test_config.kan.num_hidden_layers,
            grid=test_config.kan.grid,
            k=test_config.kan.k,
            seed=test_config.seed,
            device=test_config.device,
        )
        routing_model = dmc(cfg=test_config, device=test_config.device)
        flow = streamflow(test_config)

        _test(cfg=test_config, flow=flow, routing_model=routing_model, nn=nn_model)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")
        total_time = time.perf_counter() - start_time
        log.info(f"Total time: {total_time / 60:.2f} minutes")


if __name__ == "__main__":
    log.info(f"DDR train+test with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
