"""Shared utilities extracted from DDR scripts for testability.

Functions here are used across scripts/test.py, scripts/train.py,
scripts/router.py, and scripts/summed_q_prime.py.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from ddr.io.functions import downsample

log = logging.getLogger(__name__)


def tau_trim_and_downsample(
    hourly_predictions: torch.Tensor,
    tau: int,
) -> torch.Tensor:
    """Slice hourly predictions on the signed-at-zero tau convention, pool to daily.

    tau is the number of hours the routed output is advanced before daily
    scoring (dMC-Juniata's sign; tau=0 is day-aligned): the slice is
    ``[tau : -(24 - tau)]`` and pooled day *i* aligns with calendar day *i* of
    the batch window. Ported from ddrs (tau-sweep findings §5i); the legacy
    ``[13+tau : -11+tau]`` convention with shipped tau=3 was signed −8 —
    mis-set in the wrong direction. Legacy pin: new tau=16 cuts the same hour
    window as legacy tau=3.

    Preserves autograd — used directly in the training loss path.

    Parameters
    ----------
    hourly_predictions : torch.Tensor
        Hourly discharge, shape (num_gages, num_hours).
    tau : int
        Hours of advance, ``0 <= tau < 24``. Default in config is 9 (measured
        optimum for the flagship Q' sources; per-source optima differ — the
        aorc2f-lumped store measured ≈ −8, outside the runtime range).

    Returns
    -------
    torch.Tensor
        Daily discharge, shape (num_gages, num_days).
    """
    if not 0 <= tau < 24:
        raise ValueError(f"tau must be in [0, 24), got {tau}")
    sliced = hourly_predictions[:, tau : -(24 - tau)]
    num_days = sliced.shape[1] // 24
    return downsample(sliced, rho=num_days)


def compute_daily_runoff(
    hourly_predictions: torch.Tensor,
    tau: int,
) -> np.ndarray:
    """Numpy wrapper around :func:`tau_trim_and_downsample` for eval scripts.

    Parameters
    ----------
    hourly_predictions : torch.Tensor
        Hourly discharge, shape (num_gages, num_hours).
    tau : int
        Hours of advance, ``0 <= tau < 24``.

    Returns
    -------
    np.ndarray
        Daily discharge, shape (num_gages, num_days).
    """
    return tau_trim_and_downsample(hourly_predictions, tau).numpy()


def load_checkpoint(
    nn: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
) -> dict:
    """Load DDR checkpoint, apply state_dict to model. Returns full state dict.

    Parameters
    ----------
    nn : torch.nn.Module
        The neural network to load weights into.
    checkpoint_path : str | Path
        Path to the .pt checkpoint file.
    device : str | torch.device
        Device to map tensors to.

    Returns
    -------
    dict
        The full checkpoint state dict (contains epoch, mini_batch, etc.).
    """
    file_path = Path(checkpoint_path)
    log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
    state: dict = torch.load(file_path, map_location=device)
    state_dict = state["model_state_dict"]
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(device)
    nn.load_state_dict(state_dict)
    return state


def resolve_learning_rate(
    learning_rate_schedule: dict[int, float],
    epoch: int,
) -> float:
    """Resolve LR for epoch from schedule dict. Falls back to first entry.

    Parameters
    ----------
    learning_rate_schedule : dict[int, float]
        Mapping of epoch number → learning rate.
    epoch : int
        Current epoch.

    Returns
    -------
    float
        Learning rate for the given epoch.
    """
    if epoch in learning_rate_schedule:
        return float(learning_rate_schedule[epoch])
    key_list = list(learning_rate_schedule.keys())
    return float(learning_rate_schedule[key_list[0]])


def safe_percentile(arr: np.ndarray, percentile: float) -> float:
    """Percentile ignoring NaN values. Returns np.nan if all NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array (may contain NaN).
    percentile : float
        Percentile to compute (0-100).

    Returns
    -------
    float
        Computed percentile or np.nan.
    """
    clean_arr = arr[~np.isnan(arr)]
    if len(clean_arr) == 0:
        return float(np.nan)
    return float(np.percentile(clean_arr, percentile))


def safe_mean(arr: np.ndarray) -> float:
    """Mean ignoring NaN values. Returns np.nan if all NaN.

    Parameters
    ----------
    arr : np.ndarray
        Input array (may contain NaN).

    Returns
    -------
    float
        Computed mean or np.nan.
    """
    clean_arr = arr[~np.isnan(arr)]
    if len(clean_arr) == 0:
        return float(np.nan)
    return float(np.mean(clean_arr))
