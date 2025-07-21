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
from torch.utils.data import DataLoader

from ddr._version import __version__
from ddr.analysis import Metrics, utils
from ddr.dataset import StreamflowReader as streamflow
from ddr.dataset import eval_dataset
from ddr.dataset import utils as ds_utils
from ddr.nn import kan
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


def evaluate(cfg, flow, routing_model, nn):
    """Do model evaluation and get performance metrics."""
    log.info("Starting evaluation...")

    dataset = eval_dataset(cfg=cfg)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.eval.batch_size,  # Use evaluation batch size
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=False,  # Don't drop the last batch for evaluation
    )

    if cfg.eval.checkpoint:
        file_path = Path(cfg.eval.checkpoint)
        log.info(f"Loading model from checkpoint for evaluation: {file_path.stem}")
        state = torch.load(file_path, map_location=cfg.device)
        state_dict = state["model_state_dict"]

        nn.load_state_dict(state_dict)
        nn.eval()

        routing_model.epoch = cfg.eval.epoch
    else:
        log.error("No model checkpoint provided for evaluation. Aborting.")
        return

    all_predictions = []
    all_observations = []
    all_dates = []
    all_gage_ids = []
    all_gage_names = []

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for i, hydrofabric in enumerate(dataloader, start=0):
            routing_model.mini_batch = i

            streamflow_predictions = flow(cfg=cfg, hydrofabric=hydrofabric)
            q_prime = streamflow_predictions["streamflow"] @ hydrofabric.transition_matrix
            spatial_params = nn(inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device))
            dmc_kwargs = {
                "hydrofabric": hydrofabric,
                "spatial_parameters": spatial_params,
                "streamflow": torch.tensor(q_prime, device=cfg.device, dtype=torch.float32),
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

            all_predictions.append(filtered_predictions.detach().cpu().numpy())
            all_observations.append(filtered_observations.detach().cpu().numpy())
            all_dates.extend(dataset.dates.batch_daily_time_range[1:-1])
            all_gage_ids.extend([gage_ids[i] for i in indices])
            all_gage_names.extend([gage_names[i] for i in indices])

    # Concatenate results across all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_observations = np.concatenate(all_observations, axis=0)

    metrics = Metrics(pred=all_predictions, target=all_observations)
    nse = metrics.nse
    rmse = metrics.rmse
    kge = metrics.kge

    utils.log_metrics(nse, rmse, kge)


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="evaluation_config",  # Create a separate evaluation config
)
def main(cfg: DictConfig) -> None:
    """Main function."""
    _set_seed(cfg=cfg)
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)

    try:
        start_time = time.perf_counter()
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
        evaluate(cfg=cfg, flow=flow, routing_model=routing_model, nn=nn)

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    finally:
        print("Cleaning up...")

        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Evaluating DDR with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
