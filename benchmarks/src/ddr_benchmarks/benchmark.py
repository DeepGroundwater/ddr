"""Benchmark script to compare DDR vs DiffRoute on same data.

This module provides a benchmark runner that evaluates both DDR and DiffRoute
routing models on identical input data, allowing for direct comparison of
performance metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import networkx as nx
import numpy as np
import torch
import xarray as xr

# DiffRoute imports
from diffroute import LTIRouter, RivTree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

# Reuse ALL DDR imports
from ddr import ddr_functions, dmc, kan, streamflow
from ddr._version import __version__
from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.io.readers import read_zarr
from ddr.validation import Config, Metrics, plot_box_fig, plot_cdf, utils
from ddr.validation.enums import GeoDataset

# Benchmark config validation
from ddr_benchmarks.configs import DiffRouteConfig, validate_benchmark_config

# Adapter functions
from ddr_benchmarks.diffroute_adapter import create_param_df

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = logging.getLogger(__name__)


# ============================================================================
# Reordering functions (pattern from tests/benchmarks/test_diffroute.py:28-58)
# ============================================================================


def reorder_to_diffroute(data: torch.Tensor, topo_order: np.ndarray, riv: RivTree) -> torch.Tensor:
    """Reorder data from DDR topological order to DiffRoute's DFS ordering.

    Args:
        data: Tensor with shape (batch, nodes, time) in DDR topological order
        topo_order: Array of node IDs in DDR topological order
        riv: RivTree with nodes_idx mapping node_id -> internal_index

    Returns
    -------
        Tensor reordered for DiffRoute input
    """
    topo_to_idx = {rid: i for i, rid in enumerate(topo_order)}
    reorder_idx = [topo_to_idx[rid] for rid in riv.nodes_idx.index]
    return data[:, reorder_idx, :]


def reorder_to_topo(data: torch.Tensor, topo_order: np.ndarray, riv: RivTree) -> torch.Tensor:
    """Reorder data from DiffRoute's DFS ordering back to DDR topological order.

    Args:
        data: Tensor with node dim at index 0 or 1, in DiffRoute's DFS order
        topo_order: Array of node IDs in DDR topological order
        riv: RivTree with nodes_idx mapping node_id -> internal_index

    Returns
    -------
        Tensor reordered to DDR topological order
    """
    reorder_idx = [int(riv.nodes_idx.loc[rid]) for rid in topo_order]
    if data.dim() == 2:
        return data[reorder_idx, :]
    return data[:, reorder_idx, :]


# ============================================================================
# Model runners
# ============================================================================


def run_ddr(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    routing_dataclass: RoutingDataclass,
) -> NDArray[np.floating[Any]]:
    """Run DDR model - extracted from scripts/test.py.

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        routing_model: DMC routing model
        nn: KAN neural network for spatial parameters
        routing_dataclass: RoutingDataclass with all routing data

    Returns
    -------
        Array of routed runoff predictions
    """
    streamflow_predictions = flow(routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32)
    spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
    dmc_output = routing_model(
        routing_dataclass=routing_dataclass,
        spatial_parameters=spatial_params,
        streamflow=streamflow_predictions,
    )
    return dmc_output["runoff"].cpu().numpy()


def run_diffroute_benchmark(
    cfg: Config,
    flow: streamflow,
    diffroute_cfg: DiffRouteConfig,
    dates: Dates,
    gage_ids: np.ndarray,
    num_hourly: int,
) -> NDArray[np.floating[Any]]:
    """Run DiffRoute routing per-gage using zarr subgroup graphs.

    Each gage gets its own connected NetworkX graph from its zarr subgroup,
    avoiding the disconnected-graph problem of the global adjacency matrix.

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        diffroute_cfg: DiffRoute-specific configuration
        dates: Dates object (will be reset to full range)
        gage_ids: Array of gage IDs to process
        num_hourly: Total number of hourly timesteps for output array

    Returns
    -------
        Array of DiffRoute predictions with shape (num_gages, num_hourly)
    """
    device = torch.device(cfg.device)
    output = np.zeros((len(gage_ids), num_hourly))

    # Open gages_adjacency zarr
    assert cfg.data_sources.gages_adjacency is not None, "gages_adjacency path required for DiffRoute"
    gages_adj = read_zarr(Path(cfg.data_sources.gages_adjacency))

    # Load CONUS order to map CONUS-level indices → COMIDs.
    # Gage subgroup indices_0/indices_1 are CONUS-level (into the full ~77K order),
    # not local to the subgroup's order array. See docs/engine/binsparse.md.
    conus_adj = read_zarr(Path(cfg.data_sources.conus_adjacency))
    conus_order = conus_adj["order"][:]

    # Reset dates to full range (DDR loop leaves it on last batch)
    dates.set_batch_time(dates.daily_time_range)

    # Shared stateless router
    dt = diffroute_cfg.dt
    k_val = diffroute_cfg.k if diffroute_cfg.k is not None else dt
    router = LTIRouter(max_delay=diffroute_cfg.max_delay, dt=dt).to(device)

    for gage_idx, gage_id in enumerate(tqdm(gage_ids, desc="DiffRoute per-gage")):
        if gage_id not in gages_adj:
            log.warning(f"Gage {gage_id} not found in gages_adjacency, skipping")
            continue

        gage_group = gages_adj[gage_id]

        # Build connected graph from this gage's subgroup.
        # indices_0/indices_1 are CONUS-level indices, so we use conus_order
        # to resolve them to COMIDs.
        row = gage_group["indices_0"][:]
        col = gage_group["indices_1"][:]
        order = gage_group["order"][:]
        gage_catchment = gage_group.attrs["gage_catchment"]

        G = nx.DiGraph()
        for node_id in order:
            G.add_node(int(node_id))
        for row_idx, col_idx in zip(row, col, strict=False):
            upstream_id = int(conus_order[col_idx])
            downstream_id = int(conus_order[row_idx])
            G.add_edge(upstream_id, downstream_id)

        # Build params and RivTree for this gage's subgraph
        k_arr = np.full(len(order), k_val)
        x_arr = np.full(len(order), diffroute_cfg.x)
        param_df = create_param_df(order, k=k_arr, x=x_arr, k_units="days")

        riv = RivTree(G, irf_fn=diffroute_cfg.irf_fn, param_df=param_df)
        riv = riv.to(device)

        # Create minimal RoutingDataclass for StreamflowReader
        gage_rd = RoutingDataclass(divide_ids=order, dates=dates)

        # Get lateral inflows for this gage's subgraph
        inflows = flow(routing_dataclass=gage_rd, device=device)  # (T, N)

        # Reshape for DiffRoute: (1, N, T)
        runoff = inflows.T.unsqueeze(0)

        # Reorder to DiffRoute order, route, reorder back
        runoff_reordered = reorder_to_diffroute(runoff, order, riv)
        runoff_reordered = runoff_reordered.to(device)

        discharge = router(runoff_reordered, riv)
        discharge_topo = reorder_to_topo(discharge.squeeze(0).cpu(), order, riv)

        # Extract gage node discharge
        gage_node_idx = np.where(order == int(gage_catchment))[0][0]
        output[gage_idx, :] = discharge_topo[gage_node_idx, :].numpy()

    return output


# ============================================================================
# Plotting utilities
# ============================================================================


def generate_comparison_plots(
    cfg: Config,
    ddr_metrics: Metrics,
    diffroute_metrics: Metrics,
) -> None:
    """Generate comparison plots for DDR vs DiffRoute.

    Args:
        cfg: Validated config object with save_path
        ddr_metrics: Metrics object for DDR predictions
        diffroute_metrics: Metrics object for DiffRoute predictions
    """
    plot_path = cfg.params.save_path / "plots"

    # CDF plot for NSE comparison
    fig, ax = plot_cdf(
        data_list=[ddr_metrics.nse, diffroute_metrics.nse],
        legend_labels=["DDR", "DiffRoute"],
        title="NSE Comparison: DDR vs DiffRoute",
        xlabel="NSE",
        ylabel="CDF",
        xlim=(-1, 1),
    )
    if fig is not None:
        fig.savefig(plot_path / "nse_cdf_comparison.png", dpi=300, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)

    # Box plot comparison
    fig = plot_box_fig(
        data=[[ddr_metrics.nse, diffroute_metrics.nse]],
        xlabel_list=["NSE"],
        legend_labels=["DDR", "DiffRoute"],
        color_list=["#4878D0", "#D65F5F"],
        title="NSE Distribution: DDR vs DiffRoute",
    )
    fig.savefig(plot_path / "nse_boxplot_comparison.png", dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)

    # KGE comparison
    fig, ax = plot_cdf(
        data_list=[ddr_metrics.kge, diffroute_metrics.kge],
        legend_labels=["DDR", "DiffRoute"],
        title="KGE Comparison: DDR vs DiffRoute",
        xlabel="KGE",
        ylabel="CDF",
        xlim=(-1, 1),
    )
    if fig is not None:
        fig.savefig(plot_path / "kge_cdf_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    log.info(f"Comparison plots saved to {plot_path}")


def save_results(
    cfg: Config,
    ddr_daily: NDArray[np.floating[Any]],
    diffroute_daily: NDArray[np.floating[Any]],
    daily_obs: NDArray[np.floating[Any]],
    gage_ids: NDArray[np.str_],
    dates: Dates,
) -> None:
    """Save benchmark results to zarr.

    Args:
        cfg: Validated config object with save_path
        ddr_daily: Daily DDR predictions
        diffroute_daily: Daily DiffRoute predictions
        daily_obs: Daily observations
        gage_ids: Array of gage IDs
        dates: Dates object with time range info
    """
    time_range = dates.daily_time_range[1:-1]

    ddr_da = xr.DataArray(
        data=ddr_daily,
        dims=["gage_ids", "time"],
        coords={"gage_ids": gage_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "DDR Streamflow Predictions"},
    )

    diffroute_da = xr.DataArray(
        data=diffroute_daily,
        dims=["gage_ids", "time"],
        coords={"gage_ids": gage_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "DiffRoute Streamflow Predictions"},
    )

    obs_da = xr.DataArray(
        data=daily_obs,
        dims=["gage_ids", "time"],
        coords={"gage_ids": gage_ids, "time": time_range},
        attrs={"units": "m3/s", "long_name": "Observed Streamflow"},
    )

    ds = xr.Dataset(
        data_vars={
            "ddr_predictions": ddr_da,
            "diffroute_predictions": diffroute_da,
            "observations": obs_da,
        },
        attrs={
            "description": "Benchmark comparison: DDR vs DiffRoute",
            "ddr_version": __version__,
            "model_checkpoint": str(cfg.experiment.checkpoint) if cfg.experiment.checkpoint else "None",
        },
    )

    output_path = cfg.params.save_path / "benchmark_results.zarr"
    ds.to_zarr(output_path, mode="w")
    log.info(f"Results saved to {output_path}")


# ============================================================================
# Main benchmark function (adapted from scripts/test.py:test())
# ============================================================================


def benchmark(
    cfg: Config,
    flow: streamflow,
    routing_model: dmc,
    nn: kan,
    diffroute_cfg: DiffRouteConfig,
) -> None:
    """Run benchmark comparison - adapted from scripts/test.py:test().

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        routing_model: DMC routing model
        nn: KAN neural network
        diffroute_cfg: DiffRoute-specific configuration
    """
    assert cfg.geodataset == GeoDataset.MERIT, (
        f"Benchmarking is currently only supported on MERIT, got '{cfg.geodataset}'"
    )

    # Set CUDA device for all operations
    if torch.cuda.is_available() and isinstance(cfg.device, int):
        torch.cuda.set_device(cfg.device)

    # === REUSE DDR DATA LOADING (same as test.py) ===
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)

    if cfg.experiment.checkpoint:
        file_path = Path(cfg.experiment.checkpoint)
        device = torch.device(cfg.device)
        log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
        state = torch.load(file_path, map_location=device)
        state_dict = state["model_state_dict"]
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(device)
        nn.load_state_dict(state_dict)
    else:
        log.warning("Creating new spatial model for evaluation.")

    nn = nn.eval()

    # TODO: uncomment dataloader after DiffRoute verification
    # sampler = SequentialSampler(data_source=dataset)
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=cfg.experiment.batch_size,
    #     num_workers=0,
    #     sampler=sampler,
    #     collate_fn=dataset.collate_fn,
    #     drop_last=False,
    # )

    warmup = cfg.experiment.warmup
    assert dataset.routing_dataclass is not None, "Routing dataclass not defined in dataset"
    assert dataset.routing_dataclass.observations is not None, "Observations not defined in dataset"
    observations = dataset.routing_dataclass.observations.streamflow.values

    all_gage_ids = dataset.routing_dataclass.observations.gage_id.values
    ddr_predictions = np.zeros([len(all_gage_ids), len(dataset.dates.hourly_time_range)])
    diffroute_predictions = np.zeros_like(ddr_predictions)
    diffroute_enabled = diffroute_cfg.enabled

    # === PHASE 1: DDR (same loop as test.py) ===
    # TODO: uncomment DDR loop after DiffRoute verification
    # log.info("Starting DDR evaluation...")
    # with torch.no_grad():
    #     for i, routing_dataclass in enumerate(tqdm(dataloader, desc="DDR batches"), start=0):
    #         routing_model.set_progress_info(epoch=0, mini_batch=i)
    #
    #         streamflow_predictions = flow(
    #             routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
    #         )
    #         spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
    #         dmc_output = routing_model(
    #             routing_dataclass=routing_dataclass,
    #             spatial_parameters=spatial_params,
    #             streamflow=streamflow_predictions,
    #         )
    #         ddr_predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

    # === PHASE 2: DiffRoute per-gage ===
    if diffroute_enabled:
        log.info("Starting DiffRoute per-gage evaluation...")
        with torch.no_grad():
            diffroute_predictions = run_diffroute_benchmark(
                cfg=cfg,
                flow=flow,
                diffroute_cfg=diffroute_cfg,
                dates=dataset.dates,
                gage_ids=all_gage_ids,
                num_hourly=len(dataset.dates.hourly_time_range),
            )

    # === EVALUATION (same as test.py) ===
    num_days = len(ddr_predictions[0][13 : (-11 + cfg.params.tau)]) // 24

    # TODO: uncomment DDR downsample after DiffRoute verification
    # ddr_daily = ddr_functions.downsample(
    #     torch.tensor(ddr_predictions[:, (13 + cfg.params.tau) : (-11 + cfg.params.tau)]),
    #     rho=num_days,
    # ).numpy()

    diffroute_daily = ddr_functions.downsample(
        torch.tensor(diffroute_predictions[:, (13 + cfg.params.tau) : (-11 + cfg.params.tau)]),
        rho=num_days,
    ).numpy()

    daily_obs = observations[:, 1:-1]

    # Compute metrics using DDR's Metrics class
    # TODO: uncomment DDR metrics after DiffRoute verification
    # log.info("=" * 50)
    # log.info("=== DDR Metrics ===")
    # ddr_metrics = Metrics(pred=ddr_daily[:, warmup:], target=daily_obs[:, warmup:])
    # _nse = ddr_metrics.nse
    # nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
    # utils.log_metrics(nse, ddr_metrics.rmse, ddr_metrics.kge)

    if diffroute_enabled:
        log.info("=" * 50)
        log.info("=== DiffRoute Metrics ===")
        diffroute_metrics = Metrics(pred=diffroute_daily[:, warmup:], target=daily_obs[:, warmup:])
        _nse = diffroute_metrics.nse
        nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
        utils.log_metrics(nse, diffroute_metrics.rmse, diffroute_metrics.kge)

        # TODO: uncomment comparison plots after DDR is re-enabled
        # generate_comparison_plots(cfg, ddr_metrics, diffroute_metrics)

    # TODO: uncomment save after DDR is re-enabled
    # save_results(cfg, ddr_daily, diffroute_daily, daily_obs, all_gage_ids, dataset.dates)

    log.info("=" * 50)
    log.info("Benchmark complete!")


@hydra.main(version_base="1.3", config_path="../../config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main function - adapted from scripts/test.py:main()."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)

    # Validate benchmark config (DDR + model-specific)
    benchmark_cfg = validate_benchmark_config(cfg)
    config = benchmark_cfg.ddr
    diffroute_cfg = benchmark_cfg.diffroute

    start_time = time.perf_counter()

    try:
        # Initialize DDR models (same as test.py)
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
        routing_model = dmc(cfg=config, device=config.device)
        flow = streamflow(config)

        benchmark(cfg=config, flow=flow, routing_model=routing_model, nn=nn, diffroute_cfg=diffroute_cfg)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")
        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    import os

    import ddr_benchmarks

    os.environ["BENCHMARKS_VERSION"] = ddr_benchmarks.__version__
    print(f"Running DDR Benchmark v{ddr_benchmarks.__version__} (DDR {__version__})")
    main()
