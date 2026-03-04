"""A function which takes a trained model, then evaluates performance on a single, or many, basins"""

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

from ddr import CudaLSTM, dmc, forcings_reader, kan, streamflow
from ddr._version import __version__
from ddr.routing.utils import select_columns
from ddr.scripts_utils import compute_daily_runoff, load_checkpoint
from ddr.validation import Config, Metrics, utils, validate_config

log = logging.getLogger(__name__)


def test(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    lstm_nn: CudaLSTM,
    forcings_reader_nn: forcings_reader,
) -> None:
    """Do model evaluation and get performance metrics."""
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    if cfg.experiment.checkpoint:
        load_checkpoint(nn, cfg.experiment.checkpoint, torch.device(cfg.device), lstm_nn=lstm_nn)
    else:
        log.warning("Creating new spatial model for evaluation.")

    nn = nn.eval()
    lstm_nn.cache_states = True
    lstm_nn = lstm_nn.eval()
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

    warmup = cfg.experiment.warmup
    assert dataset.routing_dataclass is not None, "Routing dataclass not defined in dataset"
    assert dataset.routing_dataclass.observations is not None, "Observations not defined in dataset"
    observations = dataset.routing_dataclass.observations.streamflow.values

    # Create time ranges
    date_time_format = "%Y/%m/%d"
    start_time = datetime.strptime(cfg.experiment.start_time, date_time_format).strftime("%Y-%m-%d")
    end_time = datetime.strptime(cfg.experiment.end_time, date_time_format).strftime("%Y-%m-%d")
    all_gage_ids = dataset.routing_dataclass.observations.gage_id.values
    predictions = np.zeros([len(all_gage_ids), len(dataset.dates.hourly_time_range)])

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)

            streamflow_predictions = flow(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )
            attr_names = routing_dataclass.attribute_names
            normalized_attrs = routing_dataclass.normalized_spatial_attributes.to(cfg.device)
            kan_attrs = select_columns(normalized_attrs, list(cfg.kan.input_var_names), attr_names)
            spatial_params = nn(inputs=kan_attrs)
            lstm_batch_size = 1_000
            # Load forcings to CPU to avoid GPU OOM on full network
            forcing_data = forcings_reader_nn(
                routing_dataclass=routing_dataclass, device="cpu", dtype=torch.float32
            )
            all_attrs = select_columns(
                routing_dataclass.normalized_spatial_attributes,
                list(cfg.cuda_lstm.input_var_names),
                attr_names,
            )
            n_reaches = all_attrs.shape[0]

            # Save full hidden states from previous dataloader batch (None on first)
            full_hn = lstm_nn.hn
            full_cn = lstm_nn.cn
            new_full_hn: torch.Tensor | None = None
            new_full_cn: torch.Tensor | None = None
            batch_outputs: dict[str, list[torch.Tensor]] = {key: [] for key in lstm_nn.learnable_parameters}

            for s in range(0, n_reaches, lstm_batch_size):
                e = min(s + lstm_batch_size, n_reaches)
                bf = forcing_data[:, s:e, :].to(cfg.device)
                ba = all_attrs[s:e, :].to(cfg.device)

                # Slice cached hidden states for this reach batch
                if full_hn is not None:
                    assert full_cn is not None
                    lstm_nn.hn = full_hn[:, s:e, :].contiguous()
                    lstm_nn.cn = full_cn[:, s:e, :].contiguous()
                else:
                    lstm_nn.hn = None
                    lstm_nn.cn = None

                bout = lstm_nn(forcings=bf, attributes=ba)

                # Accumulate hidden states for next dataloader batch
                if lstm_nn.cache_states and lstm_nn.hn is not None:
                    if new_full_hn is None:
                        nl, _, hs = lstm_nn.hn.shape
                        new_full_hn = torch.zeros(nl, n_reaches, hs, device="cpu")
                        new_full_cn = torch.zeros(nl, n_reaches, hs, device="cpu")
                    assert lstm_nn.cn is not None
                    assert new_full_hn is not None and new_full_cn is not None
                    new_full_hn[:, s:e, :] = lstm_nn.hn.cpu()
                    new_full_cn[:, s:e, :] = lstm_nn.cn.cpu()

                for key in lstm_nn.learnable_parameters:
                    batch_outputs[key].append(bout[key])

                del bf, ba, bout
                torch.cuda.empty_cache()

            # Restore full hidden states for next dataloader batch
            if lstm_nn.cache_states:
                lstm_nn.hn = new_full_hn
                lstm_nn.cn = new_full_cn

            del forcing_data
            lstm_params = {k: torch.cat(v, dim=1) for k, v in batch_outputs.items()}
            del batch_outputs

            dmc_kwargs = {
                "routing_dataclass": routing_dataclass,
                "spatial_parameters": spatial_params,
                "streamflow": streamflow_predictions,
                "carry_state": i > 0,
                "lstm_params": lstm_params,
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
        test(
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
    print(f"Evaluating DDR with version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
