r"""Generate a geometry prediction dataset with temporal statistics.

Loads a trained KAN checkpoint and predicts channel parameters (n, p, q) for
**all** reaches in the attributes file (global MERIT).  For reaches with
streamflow data (CONUS), batches through a water year of real lateral inflows
to compute accumulated discharge and derive trapezoidal geometry statistics
(min/max/median/mean).  Non-CONUS reaches get KAN parameters and slope but
NaN geometry stats.

Usage
-----
::

    python scripts/geometry_predictor.py --config-name=merit_geometry_config

Override the water year on the command line::

    python scripts/geometry_predictor.py --config-name=merit_geometry_config \\
        experiment.start_time=2001/10/01 experiment.end_time=2002/09/30
"""

import logging
import os
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import xarray as xr
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ddr import kan, streamflow
from ddr._version import __version__
from ddr.geometry.statistics import compute_geometry_statistics
from ddr.routing.mmc import MuskingumCunge, compute_hotstart_discharge
from ddr.routing.utils import denormalize
from ddr.scripts_utils import load_checkpoint
from ddr.validation import Config, validate_config

log = logging.getLogger(__name__)


def _predict_kan_params(
    nn: kan,
    attrs_ds: xr.Dataset,
    input_var_names: list[str],
    means: torch.Tensor,
    stds: torch.Tensor,
    cfg: Config,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run KAN on all reaches in the attributes dataset.

    Returns
    -------
    all_comids : np.ndarray
        All COMIDs from the attributes file.
    n_vals, p_vals, q_vals : torch.Tensor
        Denormalized KAN parameters, shape ``(N_global,)``.
    """
    all_comids = attrs_ds["COMID"].values
    spatial_attrs = torch.tensor(
        attrs_ds[input_var_names].to_array(dim="variable").values,
        dtype=torch.float32,
        device=cfg.device,
    )

    # Fill NaNs with variable mean, then z-score normalize
    for r in range(spatial_attrs.shape[0]):
        row_mean = torch.nanmean(spatial_attrs[r])
        nan_mask = torch.isnan(spatial_attrs[r])
        spatial_attrs[r, nan_mask] = row_mean

    normalized = ((spatial_attrs - means) / stds).T  # (N_global, n_vars)

    nn = nn.eval()
    log_space = cfg.params.log_space_parameters

    # Batch KAN inference to avoid OOM on 2.9M reaches
    batch_size = 500_000
    n_all, p_all, q_all = [], [], []
    for start in range(0, normalized.shape[0], batch_size):
        chunk = normalized[start : start + batch_size].to(cfg.device)
        with torch.no_grad():
            raw = nn(inputs=chunk)

        n_all.append(denormalize(raw["n"], cfg.params.parameter_ranges["n"], "n" in log_space).cpu())
        q_all.append(
            denormalize(
                raw["q_spatial"], cfg.params.parameter_ranges["q_spatial"], "q_spatial" in log_space
            ).cpu()
        )

        if "p_spatial" in raw and "p_spatial" in cfg.params.parameter_ranges:
            p_all.append(
                denormalize(
                    raw["p_spatial"], cfg.params.parameter_ranges["p_spatial"], "p_spatial" in log_space
                ).cpu()
            )
        else:
            p_all.append(torch.full((chunk.shape[0],), cfg.params.defaults["p_spatial"]))

    n_vals = torch.cat(n_all)
    p_vals = torch.cat(p_all)
    q_vals = torch.cat(q_all)

    log.info(
        f"KAN params ({len(all_comids):,} reaches) — "
        f"n: [{n_vals.min():.4f}, {n_vals.max():.4f}], "
        f"p: [{p_vals.min():.2f}, {p_vals.max():.2f}], "
        f"q: [{q_vals.min():.4f}, {q_vals.max():.4f}]"
    )

    return all_comids, n_vals, p_vals, q_vals


def generate_geometry_dataset(
    cfg: Config,
    flow: streamflow,
    nn: kan,
) -> xr.Dataset:
    """Generate a geometry dataset for all reaches with attributes.

    KAN parameters (n, p, q) are predicted for every reach.  Geometry
    statistics are computed only for CONUS reaches that have streamflow
    data; non-CONUS reaches get NaN for geometry columns.

    Parameters
    ----------
    cfg : Config
        Validated DDR configuration.
    flow : streamflow
        StreamflowReader instance (already opened).
    nn : kan
        KAN neural network (checkpoint already loaded).

    Returns
    -------
    xr.Dataset
        Per-reach dataset with KAN parameters for all reaches and
        geometry statistics for CONUS reaches.
    """
    # 1. Build CONUS dataset — provides adjacency matrix, attributes, slope
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)
    assert dataset.routing_dataclass is not None
    rc = dataset.routing_dataclass

    conus_comids = rc.divide_ids
    n_conus = len(conus_comids)
    log.info(f"CONUS network: {n_conus:,} reaches")

    # 2. Run KAN on ALL reaches (global)
    all_comids, n_vals, p_vals, q_vals = _predict_kan_params(
        nn=nn,
        attrs_ds=dataset.attribute_ds,
        input_var_names=dataset.attributes_list,
        means=dataset.means,
        stds=dataset.stds,
        cfg=cfg,
    )
    n_global = len(all_comids)
    log.info(f"Global reaches: {n_global:,} ({n_conus:,} CONUS, {n_global - n_conus:,} non-CONUS)")

    # 3. Build slope for all reaches (NaN for non-CONUS)
    global_slope = np.full(n_global, np.nan, dtype=np.float32)
    global_id_to_idx = {int(c): i for i, c in enumerate(all_comids)}
    conus_global_indices = np.array([global_id_to_idx[int(c)] for c in conus_comids])
    conus_slope = torch.clamp(
        rc.slope.to(cfg.device).to(torch.float32),
        min=cfg.params.attribute_minimums["slope"],
    )
    global_slope[conus_global_indices] = conus_slope.cpu().numpy()

    # 4. Build pattern mapper for topological discharge accumulation (CONUS only)
    mc = MuskingumCunge(cfg, device=cfg.device)
    mc.network = rc.adjacency_matrix
    mapper, _, _ = mc.create_pattern_mapper()

    # 5. Batch through the water year, accumulate Q per day (CONUS only)
    #
    # Dates.set_date_range uses inclusive="left" for the hourly range, so a
    # chunk of N daily dates only produces (N-1)*24 hourly timesteps.  We
    # extend each chunk by 1 extra day to get full hourly coverage, then
    # only process the intended days.
    total_days = len(dataset.dates.daily_time_range)
    batch_days = cfg.experiment.batch_size
    q_lb = cfg.params.attribute_minimums["discharge"]

    daily_q = np.empty((total_days, n_conus), dtype=np.float32)
    day_idx = 0

    for batch_start in range(0, total_days, batch_days):
        batch_end = min(batch_start + batch_days, total_days)
        n_batch_days = batch_end - batch_start

        # +1 day so the hourly range covers the last intended day
        chunk_end = min(batch_end + 1, total_days)
        chunk = np.arange(batch_start, chunk_end)
        dataset.dates.set_date_range(chunk)

        q_prime = flow(routing_dataclass=rc, device=cfg.device, dtype=torch.float32)

        # Only process days whose first hour is within the hourly tensor
        processable_days = min(n_batch_days, q_prime.shape[0] // 24)
        for d in range(processable_days):
            q_prime_day = torch.clamp(q_prime[d * 24], min=q_lb)
            q_acc = compute_hotstart_discharge(q_prime_day, mapper, mc.discharge_lb, cfg.device)
            daily_q[day_idx] = q_acc.cpu().numpy()
            day_idx += 1

        log.info(f"Days {batch_start:>3d}–{batch_start + processable_days - 1:<3d} complete")

    log.info(f"Processed {day_idx} of {total_days} days")
    daily_q = daily_q[:day_idx]

    # 6. Compute geometry statistics for CONUS subset
    conus_n = n_vals[conus_global_indices]
    conus_p = p_vals[conus_global_indices]
    conus_q = q_vals[conus_global_indices]

    stats = compute_geometry_statistics(
        n=conus_n,
        p_spatial=conus_p,
        q_spatial=conus_q,
        slope=conus_slope.cpu(),
        daily_accumulated_discharge=daily_q,
        attribute_minimums=cfg.params.attribute_minimums,
    )

    # 7. Build output xr.Dataset — all reaches, NaN where no streamflow
    output = xr.Dataset(coords={"COMID": all_comids})

    output["n"] = ("COMID", n_vals.numpy())
    output["p_spatial"] = ("COMID", p_vals.numpy())
    output["q_spatial"] = ("COMID", q_vals.numpy())
    output["slope"] = ("COMID", global_slope)

    for key, conus_arr in stats.items():
        global_arr = np.full(n_global, np.nan, dtype=np.float32)
        global_arr[conus_global_indices] = conus_arr
        output[key] = ("COMID", global_arr)

    output.attrs["water_year"] = f"{cfg.experiment.start_time} to {cfg.experiment.end_time}"
    output.attrs["n_days"] = day_idx
    output.attrs["n_reaches_total"] = n_global
    output.attrs["n_reaches_conus"] = n_conus
    output.attrs["checkpoint"] = str(cfg.experiment.checkpoint) if cfg.experiment.checkpoint else ""
    output.attrs["version"] = __version__

    return output


@hydra.main(
    version_base="1.3",
    config_path="../config",
)
def main(cfg: DictConfig) -> None:
    """Generate geometry predictions for all reaches."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    config = validate_config(cfg)

    start_time = time.perf_counter()
    try:
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

        if config.experiment.checkpoint:
            load_checkpoint(nn_model, config.experiment.checkpoint, torch.device(config.device))
        else:
            log.warning("No checkpoint specified — using untrained KAN weights")

        flow = streamflow(config)
        ds = generate_geometry_dataset(cfg=config, flow=flow, nn=nn_model)

        output_path = config.params.save_path / "geometry_predictions.nc"
        ds.to_netcdf(output_path)

        total_time = time.perf_counter() - start_time
        print(
            f"\n{'═' * 46}\n"
            f"  Geometry Predictions Complete\n"
            f"{'═' * 46}\n"
            f"  Total:      {ds.attrs['n_reaches_total']:,}\n"
            f"  CONUS:      {ds.attrs['n_reaches_conus']:,}\n"
            f"  Days:       {ds.attrs['n_days']}\n"
            f"  Water year: {ds.attrs['water_year']}\n"
            f"  Output:     {output_path}\n"
            f"  Size:       {output_path.stat().st_size / 1e6:.1f} MB\n"
            f"  Runtime:    {total_time:.1f}s\n"
            f"{'═' * 46}"
        )

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")


if __name__ == "__main__":
    print(f"DDR Geometry Predictor v{__version__}")
    os.environ["DDR_VERSION"] = __version__
    main()
