"""Single-catchment DDR on the Juniata bundle — no Hydra, every knob visible.

train() / test() mirror scripts/train.py and scripts/test.py at batch_size=1.
The ``q_prime = flow(...)`` line is the seam for the future differentiable
runoff model: any module returning the same gradient-capable hourly
(num_timesteps, num_divides) m³/s tensor can replace the icechunk reader,
and gradients then flow end-to-end past the routing into runoff parameters.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ddr import dmc, kan, streamflow
from ddr.geodatazoo.base_geodataset import BaseGeoDataset
from ddr.geodatazoo.dataclasses import RoutingDataclass
from ddr.scripts_utils import resolve_learning_rate, tau_trim_and_downsample
from ddr.validation import Config, Metrics
from ddr.validation.enums import Mode

log = logging.getLogger(__name__)


def make_config(
    bundle_dir: Path,
    device: str = "cpu",
    epochs: int = 30,
    rho: int = 90,
    train_period: tuple[str, str] = ("1981/10/01", "1995/09/30"),
    test_period: tuple[str, str] = ("1995/10/01", "2010/09/30"),
) -> Config:
    """Build the same Pydantic Config the main scripts use, in plain code.

    Single gauge -> one mini-batch (one random rho-day window) per epoch, so
    ``epochs`` is the optimizer-step count. 30 steps demonstrates learning; it
    is not a converged model (see README).

    Parameters
    ----------
    bundle_dir : Path
        Root directory of the Juniata data bundle (e.g. examples/juniata/data).
    device : str, optional
        PyTorch device string, by default "cpu".
    epochs : int, optional
        Number of training steps, by default 30.
    rho : int, optional
        Days per training batch window, by default 90.
    train_period : tuple[str, str], optional
        (start, end) in ``YYYY/MM/DD`` format for training.
    test_period : tuple[str, str], optional
        (start, end) in ``YYYY/MM/DD`` format for evaluation (unused by
        make_config; test() accepts it directly).

    Returns
    -------
    Config
        Validated Pydantic config ready for train() / test().
    """
    bundle_dir = Path(bundle_dir)
    return Config(
        name="juniata-sample",
        geodataset="merit",
        mode=Mode.TRAINING,
        device=device,
        seed=42,
        np_seed=42,
        data_sources={
            # geospatial_fabric_gpkg is required by DataSources but only consumed
            # by ddr_bmi.py — not by Merit training/testing.  Point at the conus
            # adjacency zarr (an existing path) so Pydantic's Path coercion
            # succeeds without touching src/ddr/.
            "geospatial_fabric_gpkg": str(bundle_dir / "juniata_conus_adjacency.zarr"),
            "attributes": str(bundle_dir / "juniata_attributes.nc"),
            "conus_adjacency": bundle_dir / "juniata_conus_adjacency.zarr",
            "gages_adjacency": str(bundle_dir / "juniata_gages_adjacency.zarr"),
            "streamflow": str(bundle_dir / "juniata_qprime.ic"),
            "observations": str(bundle_dir / "juniata_obs.ic"),
            "gages": str(bundle_dir / "juniata_gage.csv"),
            # statistics dir is auto-created by set_statistics() on first run;
            # placing it inside the bundle keeps the example self-contained.
            "statistics": bundle_dir / "statistics",
        },
        params={"save_path": str(bundle_dir.parent / "runs"), "tau": 9},
        experiment={
            "batch_size": 1,
            "start_time": train_period[0],
            "end_time": train_period[1],
            "epochs": epochs,
            "rho": rho,
            "shuffle": True,
            "warmup": 5,
            # learning_rate keys must be int (ExperimentConfig field_validator)
            "learning_rate": {1: 1e-3, max(2, epochs // 2): 5e-4},
        },
        kan={
            "hidden_size": 21,
            "num_hidden_layers": 2,
            "grid": 50,
            "k": 2,
            "input_var_names": [
                "SoilGrids1km_clay",
                "aridity",
                "meanelevation",
                "meanP",
                "NDVI",
                "meanslope",
                "log10_uparea",
                "SoilGrids1km_sand",
                "ETPOT_Hargr",
                "Porosity",
            ],
            "learnable_parameters": ["n", "q_spatial", "p_spatial"],
        },
        # test_period is threaded through by test() below (Dates rebuild).
    )


def _models(cfg: Config) -> tuple[kan, dmc, streamflow]:
    """Instantiate the three model components from a Config.

    Parameters
    ----------
    cfg : Config
        Validated Pydantic config.

    Returns
    -------
    tuple[kan, dmc, streamflow]
        Neural network, routing model, streamflow reader.
    """
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
    return nn, dmc(cfg=cfg, device=cfg.device), streamflow(cfg)


def train(cfg: Config) -> Path:
    """Train the KAN on the single Juniata gauge; return last checkpoint path.

    Mirrors scripts/train.py at batch_size=1 — one random rho-day window per
    epoch. The collate_fn already calls dates.calculate_time_period() internally
    (BaseGeoDataset.collate_fn line 42), so we do not call it separately.

    Parameters
    ----------
    cfg : Config
        Training config (mode=TRAINING).

    Returns
    -------
    Path
        Path to the saved checkpoint .pt file.
    """
    torch.manual_seed(cfg.seed)
    # GeoDataset.get_dataset_class is the enum-factory used in scripts/train.py line 25
    dataset = cfg.geodataset.get_dataset_class(cfg=cfg)
    nn, routing_model, flow = _models(cfg)
    lr = resolve_learning_rate(cfg.experiment.learning_rate, 1)
    optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    ckpt_dir = Path(cfg.params.save_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.experiment.epochs + 1):
        # Mirror scripts/train.py: update LR when epoch key is present
        if epoch in cfg.experiment.learning_rate:
            new_lr = cfg.experiment.learning_rate[epoch]
            for g in optimizer.param_groups:
                g["lr"] = new_lr

        # collate_fn(list[str]) in training mode:
        #   1. calls dates.calculate_time_period() internally
        #   2. calls _collate_gages(np.array(batch))
        # Mirror: scripts/train.py uses DataLoader with collate_fn=dataset.collate_fn;
        # here we call it directly with all gage_ids (batch_size=1, one gage).
        batch = dataset.collate_fn(list(dataset.gage_ids))

        # q_prime seam: any hourly (T, N) m³/s tensor could replace this reader.
        # flow.forward() accepts device/dtype kwargs (defaults: cpu, float32).
        q_prime = flow(routing_dataclass=batch)
        params = nn(inputs=batch.normalized_spatial_attributes.to(cfg.device))
        routing_model.set_progress_info(epoch=epoch, mini_batch=0)
        out = routing_model(routing_dataclass=batch, spatial_parameters=params, streamflow=q_prime)

        # tau_trim_and_downsample: signed-at-zero tau convention (scripts_utils.py)
        daily = tau_trim_and_downsample(out["runoff"], cfg.params.tau)

        # Obs pairing: [:, :-2] — pooled day i ↔ obs day i; last 2 obs days have
        # no pooled window (scripts/train.py line 86 comment).
        obs = torch.tensor(
            batch.observations.streamflow.values,
            device=cfg.device,
            dtype=torch.float32,
        )[:, :-2]

        w = cfg.experiment.warmup
        daily_w = daily.transpose(0, 1)[w:].unsqueeze(2)  # (T-w, N, 1)
        obs_w = obs.transpose(0, 1)[w:].unsqueeze(2)  # (T-w, N, 1)

        # Per-timestep NaN guard: mask out any obs day that is NaN so those
        # timesteps do not contribute to the gradient.  This is safer than
        # scripts/train.py's gage-level drop (isnull().any(dim="time")) which
        # would silently zero the loss for the single gauge if it has any gap.
        mask = torch.isfinite(obs_w)
        loss = torch.nn.functional.l1_loss(daily_w[mask], obs_w[mask])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        # Free autograd graph to prevent OOM on repeated epochs (scripts/train.py line 155)
        del out, daily, q_prime, loss
        routing_model.routing_engine.q_prime = None
        routing_model.routing_engine._discharge_t = None

        log.info(f"epoch {epoch}: loss={loss_val:.4f}")

    ckpt = ckpt_dir / f"juniata_epoch_{cfg.experiment.epochs}.pt"
    torch.save({"model_state_dict": nn.state_dict(), "epoch": cfg.experiment.epochs}, ckpt)
    return ckpt


def _eval_dataset_and_batch(
    cfg: Config, test_period: tuple[str, str]
) -> tuple[Config, BaseGeoDataset, RoutingDataclass]:
    """Build the eval config, dataset, and pre-built batch for inference mode.

    In testing mode, Merit._init_inference() pre-builds a RoutingDataclass.
    collate_fn() in inference mode ignores its argument and returns the
    pre-built dataclass (BaseGeoDataset.collate_fn lines 44-49).

    Parameters
    ----------
    cfg : Config
        Training config to copy from.
    test_period : tuple[str, str]
        (start, end) in ``YYYY/MM/DD`` format.

    Returns
    -------
    tuple[Config, BaseGeoDataset, RoutingDataclass]
        Eval config, dataset, and pre-built routing dataclass.
    """
    eval_cfg = cfg.model_copy(deep=True)
    eval_cfg.mode = Mode.TESTING
    eval_cfg.experiment.start_time = test_period[0]
    eval_cfg.experiment.end_time = test_period[1]
    eval_cfg.experiment.rho = None
    dataset = eval_cfg.geodataset.get_dataset_class(cfg=eval_cfg)
    # Inference mode: collate_fn ignores the argument content but uses it to
    # set the date range via set_date_range(np.array(indices)).
    # Passing all indices covers the full evaluation period.
    # Mirror scripts/test.py: DataLoader iterates over len(dates.daily_time_range)
    # and accumulates hourly predictions in a pre-allocated array; here we do it
    # in a single call with the full index range.
    all_indices = list(range(len(dataset.dates.daily_time_range)))
    batch = dataset.collate_fn(all_indices)
    return eval_cfg, dataset, batch


def test(
    cfg: Config,
    checkpoint: Path,
    test_period: tuple[str, str] = ("1995/10/01", "2010/09/30"),
) -> xr.Dataset:
    """Evaluate a checkpoint over the full test period; return preds+obs+metrics.

    Parameters
    ----------
    cfg : Config
        Training config (make_config() output). Mode is overridden to TESTING.
    checkpoint : Path
        Path to a .pt checkpoint with "model_state_dict" key.
    test_period : tuple[str, str], optional
        (start, end) in ``YYYY/MM/DD`` format for evaluation.

    Returns
    -------
    xr.Dataset
        Variables: ``predictions`` (gage, time), ``observations`` (gage, time).
        Attrs: ``nse``, ``kge``, ``rmse`` (median across gages).
    """
    eval_cfg, dataset, batch = _eval_dataset_and_batch(cfg, test_period)
    nn, routing_model, flow = _models(eval_cfg)
    # Mirror scripts/test.py load_checkpoint usage (scripts_utils.py line 76-104)
    state = torch.load(checkpoint, map_location=eval_cfg.device, weights_only=False)
    nn.load_state_dict(state["model_state_dict"])
    nn.eval()

    with torch.no_grad():
        q_prime = flow(routing_dataclass=batch)
        params = nn(inputs=batch.normalized_spatial_attributes.to(eval_cfg.device))
        routing_model.set_progress_info(epoch=0, mini_batch=0)
        out = routing_model(routing_dataclass=batch, spatial_parameters=params, streamflow=q_prime)
        daily = tau_trim_and_downsample(out["runoff"], eval_cfg.params.tau).cpu().numpy()

    # [:, :-2] obs pairing (scripts/test.py line 77)
    obs = batch.observations.streamflow.values[:, :-2]
    # daily_time_range[:-2] matches the trimmed obs axis (scripts/test.py line 78)
    time = dataset.dates.daily_time_range[:-2]

    metrics = Metrics(pred=daily, target=obs)
    ds = xr.Dataset(
        {
            "predictions": (("gage", "time"), daily),
            "observations": (("gage", "time"), obs),
        },
        coords={"gage": ["01567000"], "time": time},
        attrs={
            "nse": float(np.nanmedian(metrics.nse)),
            "kge": float(np.nanmedian(metrics.kge)),
            "rmse": float(np.nanmedian(metrics.rmse)),
        },
    )
    return ds


def summed_qprime_baseline(
    cfg: Config,
    test_period: tuple[str, str] = ("1995/10/01", "2010/09/30"),
) -> xr.Dataset:
    """No-routing baseline: daily sum of all upstream lateral inflows.

    Uses tau=0 (day-aligned sum) to match summed_q_prime.py semantics — the
    brief explicitly requires tau=0 for the baseline.

    Parameters
    ----------
    cfg : Config
        Training config (make_config() output). Mode is overridden to TESTING.
    test_period : tuple[str, str], optional
        (start, end) in ``YYYY/MM/DD`` format for evaluation.

    Returns
    -------
    xr.Dataset
        Variable: ``predictions`` (gage, time).
        Attrs: ``nse``, ``kge`` (median across gages).
    """
    eval_cfg, dataset, batch = _eval_dataset_and_batch(cfg, test_period)
    flow = streamflow(eval_cfg)
    with torch.no_grad():
        q_prime = flow(routing_dataclass=batch)  # (T, N) hourly
        # Sum all upstream divides → single gage signal; transpose to (1, T)
        total = q_prime.sum(dim=1, keepdim=True).T  # (1, T)
        daily = tau_trim_and_downsample(total, tau=0).cpu().numpy()  # day-aligned

    time = dataset.dates.daily_time_range[:-2]
    obs = batch.observations.streamflow.values[:, :-2]
    metrics = Metrics(pred=daily, target=obs)
    return xr.Dataset(
        {"predictions": (("gage", "time"), daily)},
        coords={"gage": ["01567000"], "time": time},
        attrs={
            "nse": float(np.nanmedian(metrics.nse)),
            "kge": float(np.nanmedian(metrics.kge)),
        },
    )


def main() -> None:
    """CLI: train, test, baseline, print the comparison table."""
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=Path("examples/juniata/data"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    cfg = make_config(bundle_dir=a.bundle, device=a.device, epochs=a.epochs)
    ckpt = train(cfg)
    result = test(cfg, ckpt)
    baseline = summed_qprime_baseline(cfg)
    header = f"{'':>12} {'NSE':>8} {'KGE':>8}"
    routed_row = f"{'routed':>12} {result.attrs['nse']:8.3f} {result.attrs['kge']:8.3f}"
    label = "summed q'"
    baseline_row = f"{label:>12} {baseline.attrs['nse']:8.3f} {baseline.attrs['kge']:8.3f}"
    print(header)
    print(routed_row)
    print(baseline_row)


if __name__ == "__main__":
    main()
