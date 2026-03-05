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


def compute_daily_runoff(
    hourly_predictions: torch.Tensor,
    tau: int,
) -> np.ndarray:
    """Slice hourly predictions with tau-dependent boundary trimming, downsample to daily.

    The slicing removes boundary artifacts from timezone/routing alignment:
    - Start index: 13 + tau (skip spin-up + timezone offset)
    - End index: -11 + tau (trim trailing edge)

    Parameters
    ----------
    hourly_predictions : torch.Tensor
        Hourly discharge, shape (num_gages, num_hours).
    tau : int
        Routing time step adjustment (typically 3).

    Returns
    -------
    np.ndarray
        Daily discharge, shape (num_gages, num_days).
    """
    sliced = hourly_predictions[:, (13 + tau) : (-11 + tau)]
    num_days = sliced.shape[1] // 24
    return downsample(sliced, rho=num_days).numpy()


def load_checkpoint(
    nn: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
    phi_kan: torch.nn.Module | None = None,
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
    phi_kan : torch.nn.Module | None
        Optional phi-KAN module to load weights into.

    Returns
    -------
    dict
        The full checkpoint state dict (contains epoch, mini_batch, etc.).
    """
    file_path = Path(checkpoint_path)
    log.info(f"Loading spatial_nn from checkpoint: {file_path.stem}")
    state: dict = torch.load(file_path, map_location=device)
    state_dict = state["model_state_dict"]
    # Filter out bounds_head keys from old checkpoints (bounds_head was removed)
    state_dict = {k: v.to(device) for k, v in state_dict.items() if not k.startswith("bounds_head")}
    nn.load_state_dict(state_dict)

    if phi_kan is not None and "phi_kan_state_dict" in state:
        phi_state = state["phi_kan_state_dict"]
        for key in phi_state:
            phi_state[key] = phi_state[key].to(device)
        phi_kan.load_state_dict(phi_state)
        log.info("Loaded phi_kan weights from checkpoint")
    elif phi_kan is not None:
        log.warning("No phi_kan_state_dict in checkpoint, using random init")

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
