import logging
import random
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

from ddr._version import __version__
from ddr.nn.kan import kan
from ddr.routing.dmc import dmc
from ddr.dataset.utils import downsample
from ddr.dataset.streamflow import StreamflowReader as streamflow
from ddr.dataset.train_dataset import train_dataset

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
    
    dataset = train_dataset(cfg=cfg)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=0,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )
    
    optimizer = torch.optim.Adam(params=nn.parameters(), lr=cfg.train.learning_rate[str(0)])
    
    for epoch in range(0, cfg.train.epochs + 1):
        routing_model.epoch = epoch
        for i, hydrofabric in enumerate(dataloader, start=0):
            routing_model.mini_batch = i
            
            streamflow_predictions = flow(cfg=cfg, hydrofabric=hydrofabric)
            q_prime = streamflow_predictions["streamflow"] @ hydrofabric.transition_matrix
            spatial_params = nn(
                inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device)
            )
            dmc_kwargs = {
                "hydrofabric": hydrofabric,
                "spatial_parameters": spatial_params,
                "streamflow": torch.tensor(q_prime, device=cfg.device, dtype=torch.float32)
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
            filtered_observations = torch.tensor(filtered_ds.streamflow.values, device=cfg.device, dtype=torch.float32)[
                :, 1:-1
            ]  # Cutting off days to match with realigned timesteps

            filtered_predictions = daily_runoff[~np_nan_mask]

            loss = mse_loss(
                input=filtered_predictions.transpose(0, 1)[cfg.train.warmup:].unsqueeze(2),
                target=filtered_observations.transpose(0, 1)[cfg.train.warmup:].unsqueeze(2),
            )

            log.info("Running backpropagation")

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Loss: {loss.item()}")
        
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
    _set_seed(cfg=cfg)
    try:
        start_time = time.perf_counter()
        nn = kan(
            input_var_names=cfg.kan.input_var_names,
            learnable_parameters=cfg.kan.learnable_parameters,
            hidden_size=cfg.kan.hidden_size,
            output_size=cfg.kan.output_size,
            num_hidden_layers=cfg.kan.num_hidden_layers,
            grid=cfg.kan.grid,
            k=cfg.kan.k,
            seed=cfg.seed
        )
        routing_model = dmc(
            cfg=cfg,
            device=cfg.device
        )
        flow = streamflow(cfg)
        train(
            cfg=cfg,
            flow=flow,
            routing_model=routing_model,
            nn=nn
        )
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    
    finally:
        print("Cleaning up...")
    
        total_time = time.perf_counter() - start_time
        log.info(
            f"Time Elapsed: {(total_time / 60):.6f} minutes"
        ) 
        
if __name__ == "__main__":
    print(f"Training DDR with version: {__version__}")
    main()
