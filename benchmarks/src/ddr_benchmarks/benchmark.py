"""Benchmark script to compare DDR vs DiffRoute on same data.

This module provides a benchmark runner that evaluates both DDR and DiffRoute
routing models on identical input data, allowing for direct comparison of
performance metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import xarray as xr

# DiffRoute imports
from diffroute import LTIRouter, RivTree
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

# Reuse ALL DDR imports
from ddr import CudaLSTM, ddr_functions, dmc, forcings_reader, kan, streamflow
from ddr._version import __version__
from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.io.readers import read_zarr
from ddr.scripts_utils import load_checkpoint
from ddr.validation import Config, Metrics, plot_box_fig, plot_cdf, plot_gauge_map, plot_time_series, utils
from ddr.validation.enums import GeoDataset

# Adapter functions
from ddr_benchmarks.diffroute_adapter import create_param_df

# Benchmark config validation
from ddr_benchmarks.validation import DiffRouteConfig

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
    lstm_nn: CudaLSTM | None = None,
    forcings_reader_nn: forcings_reader | None = None,
) -> NDArray[np.floating[Any]]:
    """Run DDR model - extracted from scripts/test.py.

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        routing_model: DMC routing model
        nn: KAN neural network for spatial parameters
        routing_dataclass: RoutingDataclass with all routing data
        lstm_nn: Optional leakance LSTM model
        forcings_reader_nn: Optional forcings reader for leakance LSTM inputs

    Returns
    -------
        Array of routed runoff predictions
    """
    streamflow_predictions = flow(routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32)
    spatial_params: torch.Tensor = nn(inputs=routing_dataclass.normalized_spatial_attributes)
    spatial_params = spatial_params.to(cfg.device)
    dmc_kwargs: dict[str, Any] = {
        "routing_dataclass": routing_dataclass,
        "spatial_parameters": spatial_params,
        "streamflow": streamflow_predictions,
    }
    if lstm_nn is not None and forcings_reader_nn is not None:
        assert routing_dataclass.normalized_spatial_attributes is not None
        forcing_data = forcings_reader_nn(
            routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
        )
        lstm_params = lstm_nn(
            forcings=forcing_data,
            attributes=routing_dataclass.normalized_spatial_attributes.to(cfg.device),
        )
        dmc_kwargs["lstm_params"] = lstm_params
    dmc_output = routing_model(**dmc_kwargs)
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
    Headwater gages (single-reach, zero edges) are left as NaN because
    DiffRoute's RivTree requires at least one edge.

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        diffroute_cfg: DiffRoute-specific configuration
        dates: Dates object (will be reset to full range)
        gage_ids: Array of gage IDs to process
        num_hourly: Total number of hourly timesteps for output array

    Returns
    -------
        Array of DiffRoute predictions with shape (num_gages, num_hourly).
        Rows for headwater/missing gages are NaN.
    """
    device = torch.device(cfg.device)
    output = np.full((len(gage_ids), num_hourly), np.nan)

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
    k_val = diffroute_cfg.k if diffroute_cfg.k is not None else 0.1042
    router = LTIRouter(max_delay=diffroute_cfg.max_delay, dt=dt).to(device)

    num_headwater = 0
    for gage_idx, gage_id in enumerate(tqdm(gage_ids, desc="DiffRoute per-gage", ncols=140, ascii=True)):
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

        # Skip headwater gages — DiffRoute requires at least one edge (stays NaN)
        if len(row) == 0:
            num_headwater += 1
            continue

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

    num_routed = int(np.isfinite(output[:, 0]).sum())
    if num_headwater > 0:
        log.info(
            f"DiffRoute skipped {num_headwater} headwater gages (zero edges). "
            f"Routed {num_routed}/{len(gage_ids)} gages."
        )

    return output


def load_summed_q_prime(
    summed_q_prime_path: str,
    gage_ids: np.ndarray,
    daily_obs: NDArray[np.floating[Any]],
    warmup: int,
) -> tuple[Metrics, NDArray[np.floating[Any]], NDArray[np.bool_]] | None:
    """Load pre-computed summed Q' predictions and compute metrics.

    Parameters
    ----------
    summed_q_prime_path : str
        Path to the summed Q' zarr store (from scripts/summed_q_prime.py)
    gage_ids : np.ndarray
        Benchmark gage IDs to align with
    daily_obs : np.ndarray
        Daily observations aligned with gage_ids, shape (num_gages, num_days)
    warmup : int
        Number of warmup days to skip

    Returns
    -------
    tuple[Metrics, np.ndarray, np.ndarray] or None
        Metrics, daily predictions for common gages, and boolean mask into gage_ids
        indicating which gages were matched. Returns None if loading fails.
    """
    try:
        ds = xr.open_zarr(summed_q_prime_path)
    except (FileNotFoundError, ValueError, KeyError):
        log.warning(f"Failed to open summed Q' store at {summed_q_prime_path}")
        return None

    sqp_gage_ids = ds.gage_ids.values
    sqp_preds = ds.predictions.values  # (num_sqp_gages, num_days)

    # Find common gages and align ordering
    common_mask = np.isin(gage_ids, sqp_gage_ids)
    if not common_mask.any():
        log.warning("No common gages between benchmark and summed Q' store")
        return None

    common_gages = gage_ids[common_mask]
    sqp_idx = [np.where(sqp_gage_ids == g)[0][0] for g in common_gages]
    bench_idx = np.where(common_mask)[0]

    # Use the shorter of the two time dimensions
    num_days = min(sqp_preds.shape[1], daily_obs.shape[1])
    sqp_aligned = sqp_preds[sqp_idx, :num_days]
    obs_aligned = daily_obs[bench_idx, :num_days]

    log.info(f"Summed Q': {len(common_gages)}/{len(gage_ids)} gages matched, {num_days} days")

    metrics = Metrics(pred=sqp_aligned[:, warmup:], target=obs_aligned[:, warmup:])
    return metrics, sqp_aligned, common_mask


# ============================================================================
# Plotting utilities
# ============================================================================


def generate_comparison_plots(
    cfg: Config,
    ddr_metrics: Metrics,
    diffroute_metrics: Metrics | None,
    sqp_metrics: Metrics | None = None,
    gage_ids: np.ndarray | None = None,
    ddr_daily: NDArray | None = None,
    diffroute_daily: NDArray | None = None,
    sqp_daily: NDArray | None = None,
    sqp_common_mask: NDArray[np.bool_] | None = None,
    daily_obs: NDArray | None = None,
    dates: Dates | None = None,
    model_labels: list[str] | None = None,
) -> None:
    """Generate comparison plots for DDR vs DiffRoute (and optionally summed Q').

    Args:
        cfg: Validated config object with save_path
        ddr_metrics: Metrics object for DDR predictions
        diffroute_metrics: Metrics object for DiffRoute predictions (routed gages only)
        sqp_metrics: Optional Metrics object for summed Q' baseline
        gage_ids: Array of gage IDs
        ddr_daily: Daily DDR predictions
        diffroute_daily: Daily DiffRoute predictions (NaN for headwater gages)
        sqp_daily: Daily summed Q' predictions (common gages only)
        sqp_common_mask: Boolean mask into gage_ids for summed Q' common gages
        daily_obs: Daily observations
        dates: Dates object with time range info
        model_labels: Descriptive labels for [DDR, DiffRoute, SummedQ'] models
    """
    import matplotlib.pyplot as plt

    plot_path = cfg.params.save_path / "plots"
    time_period = f"{cfg.experiment.start_time} - {cfg.experiment.end_time}"

    # Default labels if not provided
    if model_labels is None:
        model_labels = ["DDR", "DiffRoute", "$\\sum$ Q'"]
    ddr_label = model_labels[0]
    dr_label = model_labels[1] if len(model_labels) > 1 else "DiffRoute"
    sqp_label = model_labels[2] if len(model_labels) > 2 else "$\\sum$ Q'"

    # Build data lists for CDF plots
    nse_data = [ddr_metrics.nse]
    kge_data = [ddr_metrics.kge]
    labels = [ddr_label]
    colors = ["#4878D0", "#D65F5F", "#59A14F"]

    if diffroute_metrics is not None:
        nse_data.append(diffroute_metrics.nse)
        kge_data.append(diffroute_metrics.kge)
        labels.append(dr_label)

    if sqp_metrics is not None:
        nse_data.append(sqp_metrics.nse)
        kge_data.append(sqp_metrics.kge)
        labels.append(sqp_label)

    # CDF plot for NSE comparison (no reference line)
    fig, ax = plot_cdf(
        data_list=nse_data,
        legend_labels=labels,
        title=f"NSE Comparison ({time_period})",
        xlabel="NSE",
        ylabel="CDF",
        xlim=(-1, 1),
        reference_line=None,
    )
    if fig is not None:
        fig.savefig(plot_path / "nse_cdf_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # CDF plot for KGE comparison (no reference line)
    fig, ax = plot_cdf(
        data_list=kge_data,
        legend_labels=labels,
        title=f"KGE Comparison ({time_period})",
        xlabel="KGE",
        ylabel="CDF",
        xlim=(-1, 1),
        reference_line=None,
    )
    if fig is not None:
        fig.savefig(plot_path / "kge_cdf_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # === Multi-metric box plot (matching evaluate.ipynb cell 4) ===
    key_list = ["bias", "rmse", "fhv", "flv", "nse", "kge"]
    xlabel = [r"Bias ($m^3/s$)", "RMSE", "FHV", "FLV", "NSE", "KGE"]

    # Collect all metrics objects with their labels
    all_metrics = [(ddr_metrics, ddr_label)]
    if diffroute_metrics is not None:
        all_metrics.append((diffroute_metrics, dr_label))
    if sqp_metrics is not None:
        all_metrics.append((sqp_metrics, sqp_label))

    box_labels = [label for _, label in all_metrics]
    box_colors = colors[: len(all_metrics)]

    data_box = []
    for key in key_list:
        metric_data = []
        for m, _ in all_metrics:
            vals = getattr(m, key).copy()
            # Clip NSE/KGE to [-1, 1]
            if key in ("nse", "kge"):
                vals = np.clip(vals, -1, 1)
            vals = vals[~np.isnan(vals)]
            metric_data.append(vals)
        data_box.append(metric_data)

    fig = plot_box_fig(
        data=data_box,
        xlabel_list=xlabel,
        legend_labels=box_labels,
        color_list=box_colors,
        title=f"Benchmark Comparison ({time_period})",
        sharey=False,
        figsize=(20, 8),
        legend_font_size=14,
        xlabel_font_size=16,
        tick_font_size=12,
    )
    fig.set_facecolor("white")
    fig.savefig(plot_path / "metric_boxplot_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # === Gauge maps via plot_gauge_map() ===
    if gage_ids is not None and cfg.data_sources.gages is not None:
        try:
            gages_df = pd.read_csv(cfg.data_sources.gages, dtype={"STAID": str})
            gages_df["STAID"] = gages_df["STAID"].str.zfill(8)
            gages_df = gages_df.set_index("STAID")

            gage_str_ids = [str(g).zfill(8) for g in gage_ids]
            common_mask = gages_df.index.isin(gage_str_ids)
            selected_gages = gages_df.loc[common_mask].copy()
            selected_gages = selected_gages.reset_index()

            # Align metrics to selected gages order
            gage_id_to_idx = {str(g).zfill(8): i for i, g in enumerate(gage_ids)}
            reorder = [gage_id_to_idx[sid] for sid in selected_gages["STAID"]]

            selected_gages["ddr_NSE"] = np.clip(ddr_metrics.nse[reorder], 0, 1)
            plot_gauge_map(
                gages=selected_gages,
                metric_column="ddr_NSE",
                title=f"{ddr_label} NSE",
                colormap="plasma",
                figsize=(16, 8),
                point_size=30,
                path=plot_path / "gauge_map_ddr_NSE.png",
            )

            if diffroute_metrics is not None and diffroute_daily is not None:
                # Expand routed-only metrics back to full gage array (NaN for headwaters)
                dr_routed_mask = ~np.isnan(diffroute_daily[:, 0])
                dr_nse_full = np.full(len(gage_ids), np.nan)
                dr_nse_full[dr_routed_mask] = diffroute_metrics.nse

                # Filter gauge map to only gages DiffRoute actually routed
                dr_nse_reordered = dr_nse_full[reorder]
                plot_mask = ~np.isnan(dr_nse_reordered)
                dr_gages = selected_gages[plot_mask].copy()
                dr_gages["diffroute_NSE"] = np.clip(dr_nse_reordered[plot_mask], 0, 1)

                plot_gauge_map(
                    gages=dr_gages,
                    metric_column="diffroute_NSE",
                    title=f"{dr_label} NSE",
                    colormap="plasma",
                    figsize=(16, 8),
                    point_size=30,
                    path=plot_path / "gauge_map_diffroute_NSE.png",
                )

                # Difference map: DDR - DiffRoute (positive = DDR better)
                # Only include gages DiffRoute actually routed
                ddr_nse_reordered = ddr_metrics.nse[reorder]
                nse_diff = ddr_nse_reordered[plot_mask] - dr_nse_reordered[plot_mask]
                nse_diff = np.clip(nse_diff, -1, 1)
                dr_gages["nse_diff"] = nse_diff
                vlim = max(abs(nse_diff.min()), abs(nse_diff.max()), 0.1)
                plot_gauge_map(
                    gages=dr_gages,
                    metric_column="nse_diff",
                    title=f"NSE Difference ({ddr_label} - {dr_label})",
                    colormap="RdBu",
                    colorbar_label="ΔNSE (positive = DDR better)",
                    figsize=(16, 8),
                    point_size=30,
                    vmin=-vlim,
                    vmax=vlim,
                    path=plot_path / "gauge_map_nse_diff.png",
                )

            if sqp_metrics is not None and sqp_common_mask is not None:
                # Expand common-gages-only metrics to full gage array (NaN for unmatched)
                sqp_nse_full = np.full(len(gage_ids), np.nan)
                sqp_nse_full[sqp_common_mask] = sqp_metrics.nse

                sqp_nse_reordered = sqp_nse_full[reorder]
                sqp_plot_mask = ~np.isnan(sqp_nse_reordered)
                sqp_gages = selected_gages[sqp_plot_mask].copy()
                sqp_gages["sqp_NSE"] = np.clip(sqp_nse_reordered[sqp_plot_mask], 0, 1)
                plot_gauge_map(
                    gages=sqp_gages,
                    metric_column="sqp_NSE",
                    title=f"{sqp_label} NSE",
                    colormap="plasma",
                    figsize=(16, 8),
                    point_size=30,
                    path=plot_path / "gauge_map_sqp_NSE.png",
                )
        except (FileNotFoundError, KeyError, ValueError):
            log.warning("Failed to generate gauge maps", exc_info=True)

    # === Per-gage hydrographs ===
    if gage_ids is not None and ddr_daily is not None and daily_obs is not None and dates is not None:
        hydro_path = plot_path / "hydrographs"
        hydro_path.mkdir(exist_ok=True)
        time_range = dates.daily_time_range[1:-1]

        # Expand subset metrics to full gage arrays for hydrograph labeling
        dr_nse_full = None
        if diffroute_metrics is not None and diffroute_daily is not None:
            dr_routed_mask = ~np.isnan(diffroute_daily[:, 0])
            dr_nse_full = np.full(len(gage_ids), np.nan)
            dr_nse_full[dr_routed_mask] = diffroute_metrics.nse

        sqp_nse_full = None
        if sqp_metrics is not None and sqp_common_mask is not None:
            sqp_nse_full = np.full(len(gage_ids), np.nan)
            sqp_nse_full[sqp_common_mask] = sqp_metrics.nse

        for gage_idx, gage_id in enumerate(
            tqdm(gage_ids, desc="Generating hydrographs", ncols=140, ascii=True)
        ):
            additional = []
            if diffroute_daily is not None and dr_nse_full is not None:
                # Skip DiffRoute line for headwater gages (NaN predictions)
                if not np.isnan(diffroute_daily[gage_idx, 0]):
                    additional.append(
                        (
                            diffroute_daily[gage_idx],
                            dr_label,
                            {"nse": float(dr_nse_full[gage_idx])},
                        )
                    )
            if sqp_daily is not None and sqp_nse_full is not None:
                # Skip summed Q' line for gages not in the summed Q' store
                if not np.isnan(sqp_nse_full[gage_idx]):
                    # sqp_daily is indexed by common gages; find position in subset
                    sqp_sub_idx = int(sqp_common_mask[: gage_idx + 1].sum()) - 1  # type: ignore
                    additional.append(
                        (
                            sqp_daily[sqp_sub_idx],
                            sqp_label,
                            {"nse": float(sqp_nse_full[gage_idx])},
                        )
                    )

            plot_time_series(
                prediction=ddr_daily[gage_idx],
                observation=daily_obs[gage_idx],
                time_range=time_range,
                gage_id=str(gage_id),
                name=str(gage_id),
                metrics={"nse": float(ddr_metrics.nse[gage_idx])},
                path=hydro_path / f"{gage_id}.png",
                warmup=cfg.experiment.warmup,
                title=f"Benchmark Hydrograph - GAGE ID: {gage_id}",
                additional_predictions=additional if additional else None,
            )

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
    summed_q_prime_path: str | None = None,
    lstm_nn: CudaLSTM | None = None,
    forcings_reader_nn: forcings_reader | None = None,
) -> None:
    """Run benchmark comparison - adapted from scripts/test.py:test().

    Args:
        cfg: Validated config object
        flow: Streamflow reader
        routing_model: DMC routing model
        nn: KAN neural network
        diffroute_cfg: DiffRoute-specific configuration
        summed_q_prime_path: Optional path to summed Q' zarr store
        lstm_nn: Optional leakance LSTM model
        forcings_reader_nn: Optional forcings reader for leakance LSTM inputs
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
        load_checkpoint(nn, cfg.experiment.checkpoint, torch.device(cfg.device), lstm_nn=lstm_nn)
    else:
        log.warning("Creating new spatial model for evaluation.")

    nn = nn.eval()
    if lstm_nn is not None:
        lstm_nn.cache_states = True
        lstm_nn = lstm_nn.eval()

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
    diffroute_predictions = np.full_like(ddr_predictions, np.nan)
    diffroute_enabled = diffroute_cfg.enabled

    # === PHASE 1: DDR (same loop as test.py) ===
    log.info("Starting DDR evaluation...")
    with torch.no_grad():
        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=0, mini_batch=i)

            streamflow_predictions = flow(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )
            spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(cfg.device))
            dmc_kwargs: dict[str, Any] = {
                "routing_dataclass": routing_dataclass,
                "spatial_parameters": spatial_params,
                "streamflow": streamflow_predictions,
                "carry_state": i > 0,
            }

            if lstm_nn is not None and forcings_reader_nn is not None:
                lstm_batch_size = 1_000
                # Load forcings to CPU to avoid GPU OOM on full network
                forcing_data = forcings_reader_nn(
                    routing_dataclass=routing_dataclass, device="cpu", dtype=torch.float32
                )
                all_attrs = routing_dataclass.normalized_spatial_attributes
                n_reaches = all_attrs.shape[0]

                # Save full hidden states from previous dataloader batch (None on first)
                full_hn = lstm_nn.hn
                full_cn = lstm_nn.cn
                new_full_hn: torch.Tensor | None = None
                new_full_cn: torch.Tensor | None = None
                batch_outputs: dict[str, list[torch.Tensor]] = {
                    key: [] for key in lstm_nn.learnable_parameters
                }

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
                dmc_kwargs["lstm_params"] = lstm_params

            dmc_output = routing_model(**dmc_kwargs)
            ddr_predictions[:, dataset.dates.hourly_indices] = dmc_output["runoff"].cpu().numpy()

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

    # === Filter to gages DiffRoute actually routed (apples-to-apples) ===
    if diffroute_enabled:
        routed_mask = ~np.isnan(diffroute_predictions[:, 0])
        num_excluded = int((~routed_mask).sum())
        log.info(
            f"Filtering to {int(routed_mask.sum())}/{len(all_gage_ids)} gages "
            f"({num_excluded} headwater/skipped gages excluded for apples-to-apples comparison)"
        )
        all_gage_ids = all_gage_ids[routed_mask]
        observations = observations[routed_mask]
        ddr_predictions = ddr_predictions[routed_mask]
        diffroute_predictions = diffroute_predictions[routed_mask]

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

    diffroute_metrics = None
    if diffroute_enabled:
        log.info("=" * 50)
        log.info(f"=== DiffRoute Metrics ({len(all_gage_ids)} gages) ===")
        diffroute_metrics = Metrics(pred=diffroute_daily[:, warmup:], target=daily_obs[:, warmup:])
        _nse = diffroute_metrics.nse
        nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
        utils.log_metrics(nse, diffroute_metrics.rmse, diffroute_metrics.kge)

    # Optional summed Q' baseline
    sqp_metrics = None
    sqp_daily = None
    sqp_common_mask = None
    if summed_q_prime_path is not None:
        log.info("=" * 50)
        log.info("=== Summed Q' Metrics ===")
        result = load_summed_q_prime(summed_q_prime_path, all_gage_ids, daily_obs, warmup)
        if result is not None:
            sqp_metrics, sqp_daily, sqp_common_mask = result
            _nse = sqp_metrics.nse
            nse = _nse[~np.isinf(_nse) & ~np.isnan(_nse)]
            utils.log_metrics(nse, sqp_metrics.rmse, sqp_metrics.kge)

    # === MASS BALANCE CHECK ===
    log.info("=" * 50)
    log.info("=== Mass Balance Accumulation Comparison ===")

    if sqp_daily is not None and sqp_common_mask is not None:
        sqp_total = sqp_daily[:, warmup:].sum(axis=1)
        # sqp_daily is indexed by common gages only; subset ddr_daily to match
        ddr_total_common = ddr_daily[sqp_common_mask, warmup:].sum(axis=1)
        ddr_vs_sqp = np.abs(ddr_total_common - sqp_total) / np.where(sqp_total != 0, sqp_total, 1.0)
        log.info(
            f"DDR vs ΣQ' — Mean rel. error: {ddr_vs_sqp.mean():.4f}, Median: {np.median(ddr_vs_sqp):.4f}"
        )
        if diffroute_enabled:
            # Intersect DiffRoute routed gages with summed Q' common gages
            dr_routed = ~np.isnan(diffroute_daily[:, 0])
            both_mask = sqp_common_mask & dr_routed
            dr_total = diffroute_daily[both_mask, warmup:].sum(axis=1)
            # sqp_daily is indexed by common gages; subset to those also routed by DiffRoute
            sqp_both = dr_routed[sqp_common_mask]
            sqp_total_dr = sqp_total[sqp_both]
            dr_vs_sqp = np.abs(dr_total - sqp_total_dr) / np.where(sqp_total_dr != 0, sqp_total_dr, 1.0)
            log.info(
                f"DiffRoute vs ΣQ' — Mean rel. error: {dr_vs_sqp.mean():.4f}, Median: {np.median(dr_vs_sqp):.4f}"
            )

    # Build descriptive model labels
    model_labels = [f"DDR v{__version__}", f"DiffRoute ({diffroute_cfg.irf_fn})", "$\\sum$ Q'"]

    # Generate comparison plots
    generate_comparison_plots(
        cfg=cfg,
        ddr_metrics=ddr_metrics,
        diffroute_metrics=diffroute_metrics,
        sqp_metrics=sqp_metrics,
        gage_ids=all_gage_ids,
        ddr_daily=ddr_daily,
        diffroute_daily=diffroute_daily if diffroute_enabled else None,
        sqp_daily=sqp_daily,
        sqp_common_mask=sqp_common_mask,
        daily_obs=daily_obs,
        dates=dataset.dates,
        model_labels=model_labels,
    )

    # Save results to zarr
    save_results(cfg, ddr_daily, diffroute_daily, daily_obs, all_gage_ids, dataset.dates)

    log.info("=" * 50)
    log.info("Benchmark complete!")
