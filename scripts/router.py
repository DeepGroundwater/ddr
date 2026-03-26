"""A function which takes a trained model, then runs forward simulation at catchment scale."""

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
from ddr.validation import Config, validate_config
from ddr.validation.plots import plot_routing_hydrograph

log = logging.getLogger(__name__)


def print_routing_summary(
    ds: xr.Dataset,
    save_path: Path,
    runtime_seconds: float,
    plot_path: Path | None = None,
) -> str:
    """Print a human-readable summary of routing results to the terminal.

    Parameters
    ----------
    ds : xr.Dataset
        The routing output dataset containing a ``predictions`` variable.
    save_path : Path
        Directory where the zarr output was saved.
    runtime_seconds : float
        Wall-clock routing time in seconds.
    plot_path : Path | None, optional
        Path to the generated hydrograph PNG, if available.

    Returns
    -------
    str
        The formatted summary string (also printed to stdout).
    """
    pred = ds["predictions"]
    num_segments = pred.sizes["catchment_ids"]
    time_coords = pred.coords["time"].values
    time_start = str(np.datetime_as_string(time_coords[0], unit="D"))
    time_end = str(np.datetime_as_string(time_coords[-1], unit="D"))

    values = pred.values
    q_min = float(np.nanmin(values))
    q_mean = float(np.nanmean(values))
    q_max = float(np.nanmax(values))

    zarr_path = save_path / "chrout.zarr"

    lines = [
        "",
        "\u2550" * 46,
        "  DDR Routing Complete",
        "\u2550" * 46,
        f"  Segments routed:  {num_segments:,}",
        f"  Time range:       {time_start} \u2192 {time_end}",
        f"  Output:           {zarr_path}",
        "",
        "  Discharge summary (m\u00b3/s):",
        f"    Min:    {q_min:,.2f}",
        f"    Mean:   {q_mean:,.1f}",
        f"    Max:    {q_max:,.1f}",
        "",
        f"  Runtime: {runtime_seconds:.1f}s",
    ]
    if plot_path is not None:
        lines.append(f"  Plot:    {plot_path}")
    lines.append("\u2550" * 46)

    summary = "\n".join(lines)
    print(summary)
    return summary


def route_trained_model(cfg: Config, flow: streamflow, routing_model: dmc, nn: kan) -> xr.Dataset:
    """Route a trained model over a specific amount of defined catchments.

    Parameters
    ----------
    cfg : Config
        Validated DDR configuration.
    flow : streamflow
        Streamflow reader instance.
    routing_model : dmc
        Differentiable Muskingum-Cunge routing model.
    nn : kan
        KAN spatial parameter network.

    Returns
    -------
    xr.Dataset
        The routing output dataset written to zarr, containing a
        ``predictions`` DataArray with dims ``(catchment_ids, time)``.
    """
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
        drop_last=False,  # Cannot drop last as it's needed for eval
    )

    # Create time ranges
    date_time_format = "%Y/%m/%d"
    start_time = datetime.strptime(cfg.experiment.start_time, date_time_format).strftime("%Y-%m-%d")
    end_time = datetime.strptime(cfg.experiment.end_time, date_time_format).strftime("%Y-%m-%d")

    assert dataset.routing_dataclass is not None, "Routing dataclass not defined in dataset"
    assert dataset.routing_dataclass.outflow_idx is not None, "Routing dataclass output_idx not defined"
    assert dataset.routing_dataclass.adjacency_matrix is not None, (
        "Routing dataclass adjacency_matrix not defined"
    )

    if cfg.data_sources.target_catchments is not None:
        num_outputs = len(dataset.routing_dataclass.outflow_idx)
        log.info(f"Routing for {num_outputs} target catchments")
    elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
        num_outputs = len(dataset.routing_dataclass.outflow_idx)
        log.info(f"Routing for {num_outputs} gages")
    else:
        num_outputs = dataset.routing_dataclass.adjacency_matrix.shape[0]
        log.info(f"Routing for {num_outputs} segments (all)")

    num_timesteps = len(dataset.dates.hourly_time_range)
    predictions = np.zeros((num_outputs, num_timesteps), dtype=np.float32)

    with torch.no_grad():  # Disable gradient calculations during evaluation
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
    time_range = dataset.dates.daily_time_range[1:-1]

    pred_da = xr.DataArray(
        data=daily_runoff,
        dims=["catchment_ids", "time"],
        coords={"catchment_ids": dataset.routing_dataclass.divide_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "Streamflow"},
    )
    attrs = {
        "description": "Predictions and obs for time period",
        "start time": start_time,
        "end time": end_time,
        "version": __version__,
        "model": str(cfg.experiment.checkpoint) if cfg.experiment.checkpoint else "No Trained Model",
    }
    if cfg.data_sources.target_catchments is not None:
        attrs["target catchments"] = str(cfg.data_sources.target_catchments)
    elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
        attrs["evaluation basins file"] = str(cfg.data_sources.gages)
    else:
        attrs["large scale simulation"] = str(True)
    ds = xr.Dataset(
        data_vars={"predictions": pred_da},
        attrs=attrs,
    )
    ds.to_zarr(
        cfg.params.save_path / "chrout.zarr",
        mode="w",
    )

    log.info("Routing complete.")
    return ds


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
    ds: xr.Dataset | None = None
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
        ds = route_trained_model(cfg=config, flow=flow, routing_model=routing_model, nn=nn)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        total_time = time.perf_counter() - start_time

        if ds is not None:
            # Generate hydrograph plot
            plot_path = config.params.save_path / "routing_summary.png"
            try:
                plot_routing_hydrograph(
                    predictions=ds["predictions"],
                    path=plot_path,
                    target_catchments=config.data_sources.target_catchments,
                    gage_ids=(
                        config.data_sources.gages.split(",")
                        if isinstance(config.data_sources.gages, str)
                        and config.data_sources.target_catchments is None
                        else None
                    ),
                )
            except Exception:
                log.exception("Failed to generate routing hydrograph plot")
                plot_path = None

            # Print terminal summary
            print_routing_summary(
                ds=ds,
                save_path=config.params.save_path,
                runtime_seconds=total_time,
                plot_path=plot_path,
            )
        else:
            log.info("Cleaning up...")

        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    print(f"Forward simulation with DDR version: {__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
