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
    lstm_nn: torch.nn.Module | None = None,
    kan_optimizer: torch.optim.Optimizer | None = None,
    lstm_optimizer: torch.optim.Optimizer | None = None,
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
    lstm_nn : torch.nn.Module | None, optional
        The CudaLSTM model to load weights into, by default None.
    kan_optimizer : torch.optim.Optimizer | None, optional
        The KAN optimizer to restore state into, by default None.
    lstm_optimizer : torch.optim.Optimizer | None, optional
        The LSTM optimizer to restore state into, by default None.

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

    if lstm_nn is not None and "lstm_nn_state_dict" in state:
        log.info("Loading lstm_nn from checkpoint")
        lstm_state = state["lstm_nn_state_dict"]
        for key in lstm_state.keys():
            lstm_state[key] = lstm_state[key].to(device)
        lstm_nn.load_state_dict(lstm_state)

    if kan_optimizer is not None:
        if "kan_optimizer_state_dict" in state:
            kan_optimizer.load_state_dict(state["kan_optimizer_state_dict"])
        elif "optimizer_state_dict" in state:
            # Old checkpoints used a single Adam over KAN+LSTM params combined.
            # Param count won't match the new KAN-only optimizer, so skip.
            log.warning(
                "Old checkpoint has combined optimizer_state_dict (KAN+LSTM). "
                "Skipping optimizer restore — both optimizers will start fresh."
            )

    if lstm_optimizer is not None and "lstm_optimizer_state_dict" in state:
        lstm_optimizer.load_state_dict(state["lstm_optimizer_state_dict"])

    return state


def resolve_learning_rate(
    learning_rate_schedule: dict[int, float],
    epoch: int,
) -> float:
    """Resolve the learning rate for a given epoch from the schedule.

    Finds the largest epoch key <= the current epoch. Falls back to the
    smallest key if no key matches.

    Parameters
    ----------
    learning_rate_schedule : dict[int, float]
        Mapping of epoch numbers to learning rates.
    epoch : int
        Current epoch number.

    Returns
    -------
    float
        The learning rate to use.
    """
    applicable = {k: v for k, v in learning_rate_schedule.items() if k <= epoch}
    if applicable:
        return float(applicable[max(applicable)])
    return float(learning_rate_schedule[min(learning_rate_schedule)])


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
