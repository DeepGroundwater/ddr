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
import numpy as np
import torch
import xarray as xr

# DiffRoute imports
from diffroute import LTIRouter, RivTree
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from scipy import sparse
from torch.utils.data import DataLoader, SequentialSampler

# Reuse ALL DDR imports
from ddr import ddr_functions, dmc, kan, streamflow
from ddr._version import __version__
from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.validation import Config, Metrics, plot_box_fig, plot_cdf, utils

# Benchmark config validation
from ddr_benchmarks.configs import DiffRouteConfig, validate_benchmark_config

# EXISTING adapter functions - COO -> NetworkX conversion
from ddr_benchmarks.diffroute_adapter import coo_to_networkx, create_param_df

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

    sampler = SequentialSampler(data_source=dataset)
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

    all_gage_ids = dataset.routing_dataclass.observations.gage_id.values
    ddr_predictions = np.zeros([len(all_gage_ids), len(dataset.dates.hourly_time_range)])
    diffroute_predictions = np.zeros_like(ddr_predictions)
    diffroute_enabled = diffroute_cfg.enabled

    # === BUILD DIFFROUTE NETWORK (once, before loop) ===
    if diffroute_enabled:
        log.info("Building DiffRoute network from adjacency matrix...")
        assert dataset.routing_dataclass.adjacency_matrix is not None
        adj_coo = dataset.routing_dataclass.adjacency_matrix.cpu().to_sparse_coo().coalesce()
        indices = adj_coo.indices().numpy()
        values = adj_coo.values().numpy()
        coo = sparse.coo_matrix((values, (indices[0], indices[1])), shape=tuple(adj_coo.shape))

        topo_order = dataset.routing_dataclass.divide_ids
        G = coo_to_networkx(coo, topo_order)

        dt = diffroute_cfg.dt
        k_val = diffroute_cfg.k if diffroute_cfg.k is not None else dt
        k_arr = np.full(len(topo_order), k_val)
        x_arr = np.full(len(topo_order), diffroute_cfg.x)
        param_df = create_param_df(topo_order, k=k_arr, x=x_arr, k_units="days")

        riv = RivTree(G, irf_fn=diffroute_cfg.irf_fn, param_df=param_df)
        router = LTIRouter(max_delay=diffroute_cfg.max_delay, dt=dt)
        riv = riv.to(cfg.device)
        router = router.to(cfg.device)

        # Map gage_catchment IDs to indices in divide_ids for gage extraction
        assert dataset.routing_dataclass.gage_catchment is not None
        divide_id_to_idx = {int(did): i for i, did in enumerate(topo_order)}
        gage_indices = [divide_id_to_idx[int(gc)] for gc in dataset.routing_dataclass.gage_catchment]
        log.info("DiffRoute network ready.")

    # === RUN BOTH MODELS ON SAME DATA (same loop as test.py) ===
    log.info("Starting benchmark evaluation...")
    with torch.no_grad():
        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)
            log.info(f"Processing batch {i + 1}")

            # Lateral inflows (shared by both models)
            streamflow_predictions = flow(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )

            # --- DDR (same as test.py) ---
            spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
            dmc_output = routing_model(
                routing_dataclass=routing_dataclass,
                spatial_parameters=spatial_params,
                streamflow=streamflow_predictions,
            )
            ddr_predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

            # --- DiffRoute (on same lateral inflows) ---
            if diffroute_enabled:
                runoff = streamflow_predictions.T.unsqueeze(0)  # (1, reaches, timesteps)
                runoff_reordered = reorder_to_diffroute(runoff, topo_order, riv)
                runoff_reordered = runoff_reordered.to(cfg.device)

                discharge = router(runoff_reordered, riv)  # (1, nodes, time)
                discharge_topo = reorder_to_topo(discharge.squeeze(0).cpu(), topo_order, riv)

                # Extract discharge at gage nodes
                diffroute_predictions[:, dataset.dates.hourly_indices] = discharge_topo[
                    gage_indices, :
                ].numpy()

    # === EVALUATION (same as test.py) ===
    num_days = len(ddr_predictions[0][13 : (-11 + cfg.params.tau)]) // 24

    ddr_daily = ddr_functions.downsample(
        torch.tensor(ddr_predictions[:, (13 + cfg.params.tau) : (-11 + cfg.params.tau)]),
        rho=num_days,
    ).numpy()

    diffroute_daily = ddr_functions.downsample(
        torch.tensor(diffroute_predictions[:, (13 + cfg.params.tau) : (-11 + cfg.params.tau)]),
        rho=num_days,
    ).numpy()

    daily_obs = observations[:, 1:-1]

    # Compute metrics using DDR's Metrics class
    log.info("=" * 50)
    log.info("=== DDR Metrics ===")
    ddr_metrics = Metrics(pred=ddr_daily[:, warmup:], target=daily_obs[:, warmup:])
    _nse = ddr_metrics.nse
    nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
    utils.log_metrics(nse, ddr_metrics.rmse, ddr_metrics.kge)

    if diffroute_enabled:
        log.info("=" * 50)
        log.info("=== DiffRoute Metrics ===")
        diffroute_metrics = Metrics(pred=diffroute_daily[:, warmup:], target=daily_obs[:, warmup:])
        _nse = diffroute_metrics.nse
        nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
        utils.log_metrics(nse, diffroute_metrics.rmse, diffroute_metrics.kge)

        # Generate comparison plots using DDR's plotting
        generate_comparison_plots(cfg, ddr_metrics, diffroute_metrics)

    # Save results to zarr
    save_results(cfg, ddr_daily, diffroute_daily, daily_obs, all_gage_ids, dataset.dates)

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
