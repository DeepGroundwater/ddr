"""TensorBoard logging wrapper with no-op fallback.

Provides a factory function ``create_tb_logger`` that returns either a real
TensorBoard logger (``TBLogger``) or a silent no-op (``_NoOpTBLogger``).
Calling code uses the logger unconditionally — zero branching required.

Example
-------
>>> tb = create_tb_logger(enabled=True, log_dir=Path("runs/tb"))
>>> tb.log_loss(0.42, global_step=100)
>>> tb.close()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ddr.validation.metrics import Metrics

log = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


class _NoOpTBLogger:
    """Silent no-op logger — all methods accept any arguments and do nothing."""

    def log_loss(self, loss: float, global_step: int) -> None:
        """No-op."""

    def log_learning_rate(self, lr: float, global_step: int) -> None:
        """No-op."""

    def log_grad_norm(self, grad_norm: float, global_step: int) -> None:
        """No-op."""

    def log_metrics(
        self,
        nse: np.ndarray,
        rmse: np.ndarray,
        kge: np.ndarray,
        global_step: int,
    ) -> None:
        """No-op."""

    def log_routing_params(
        self,
        n_vals: Any,
        global_step: int,
        x_vals: Any | None = None,
        pool_elev: Any | None = None,
    ) -> None:
        """No-op."""

    def log_benchmark_metrics(
        self,
        metrics: Metrics,
        model_name: str,
        global_step: int = 0,
    ) -> None:
        """No-op."""

    def close(self) -> None:
        """No-op."""


class TBLogger:
    """Thin wrapper around ``SummaryWriter`` with typed logging methods.

    Parameters
    ----------
    log_dir : Path
        Directory for TensorBoard event files.
    log_interval : int
        Only write scalars when ``global_step % log_interval == 0``.
    """

    def __init__(self, log_dir: Path, log_interval: int = 1) -> None:
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._log_interval = max(1, log_interval)

    def _should_log(self, global_step: int) -> bool:
        return global_step % self._log_interval == 0

    def log_loss(self, loss: float, global_step: int) -> None:
        """Log training loss."""
        if self._should_log(global_step):
            self._writer.add_scalar("train/loss", loss, global_step)

    def log_learning_rate(self, lr: float, global_step: int) -> None:
        """Log current learning rate."""
        if self._should_log(global_step):
            self._writer.add_scalar("train/learning_rate", lr, global_step)

    def log_grad_norm(self, grad_norm: float, global_step: int) -> None:
        """Log clipped gradient norm."""
        if self._should_log(global_step):
            self._writer.add_scalar("train/grad_norm", grad_norm, global_step)

    def log_metrics(
        self,
        nse: np.ndarray,
        rmse: np.ndarray,
        kge: np.ndarray,
        global_step: int,
    ) -> None:
        """Log per-step aggregate metrics (mean and median)."""
        if not self._should_log(global_step):
            return

        for name, vals in [("nse", nse), ("rmse", rmse), ("kge", kge)]:
            clean = vals[~np.isnan(vals) & ~np.isinf(vals)]
            if len(clean) > 0:
                self._writer.add_scalar(f"metrics/{name}_mean", float(np.mean(clean)), global_step)
                self._writer.add_scalar(f"metrics/{name}_median", float(np.median(clean)), global_step)

    def log_routing_params(
        self,
        n_vals: Any,
        global_step: int,
        x_vals: Any | None = None,
        pool_elev: Any | None = None,
    ) -> None:
        """Log routing parameter distributions."""
        if not self._should_log(global_step):
            return

        n = n_vals.float()
        self._writer.add_scalar("params/mannings_n_median", float(n.median()), global_step)
        self._writer.add_scalar("params/mannings_n_mean", float(n.mean()), global_step)
        self._writer.add_scalar("params/mannings_n_min", float(n.min()), global_step)
        self._writer.add_scalar("params/mannings_n_max", float(n.max()), global_step)

        if x_vals is not None:
            x = x_vals.float()
            self._writer.add_scalar("params/muskingum_x_median", float(x.median()), global_step)
            self._writer.add_scalar("params/muskingum_x_mean", float(x.mean()), global_step)
            self._writer.add_scalar("params/muskingum_x_min", float(x.min()), global_step)
            self._writer.add_scalar("params/muskingum_x_max", float(x.max()), global_step)

        if pool_elev is not None:
            p = pool_elev.float()
            self._writer.add_scalar("params/pool_elev_median", float(p.median()), global_step)
            self._writer.add_scalar("params/pool_elev_min", float(p.min()), global_step)
            self._writer.add_scalar("params/pool_elev_max", float(p.max()), global_step)

    def log_benchmark_metrics(
        self,
        metrics: Metrics,
        model_name: str,
        global_step: int = 0,
    ) -> None:
        """Log all benchmark metrics for a given model.

        Parameters
        ----------
        metrics : Metrics
            Computed benchmark metrics object.
        model_name : str
            Model identifier used as tag prefix, e.g. ``"ddr"``, ``"diffroute"``.
        global_step : int
            TensorBoard step (default 0 for one-shot benchmarks).
        """
        prefix = f"benchmark/{model_name}"
        for key in ("nse", "kge", "rmse", "bias", "fhv", "flv"):
            vals = getattr(metrics, key)
            clean = vals[~np.isnan(vals) & ~np.isinf(vals)]
            if len(clean) > 0:
                self._writer.add_scalar(f"{prefix}/{key}_mean", float(np.mean(clean)), global_step)
                self._writer.add_scalar(f"{prefix}/{key}_median", float(np.median(clean)), global_step)

    def close(self) -> None:
        """Flush and close the underlying ``SummaryWriter``."""
        self._writer.close()


def create_tb_logger(
    enabled: bool,
    log_dir: Path,
    log_interval: int = 1,
) -> TBLogger | _NoOpTBLogger:
    """Factory: return a real or no-op TensorBoard logger.

    Parameters
    ----------
    enabled : bool
        Whether TensorBoard logging was requested (``log_tensorboard`` config).
    log_dir : Path
        Directory for event files (created by ``SummaryWriter``).
    log_interval : int
        Write scalars every *N* steps (training only; benchmarks use 1).

    Returns
    -------
    TBLogger | _NoOpTBLogger
        A real logger when *enabled* and tensorboard is installed,
        otherwise a silent no-op.
    """
    if not enabled:
        return _NoOpTBLogger()

    if not _TB_AVAILABLE:
        log.warning(
            "log_tensorboard=True but tensorboard is not installed. "
            "Install with: uv sync --group tb. Falling back to no-op logger."
        )
        return _NoOpTBLogger()

    log_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"TensorBoard logging to {log_dir}")
    return TBLogger(log_dir=log_dir, log_interval=log_interval)
