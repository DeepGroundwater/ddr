import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _cpu_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """Copy optimizer state dict to CPU.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose state to copy.

    Returns
    -------
    dict[str, Any]
        Optimizer state dict with all tensors on CPU.
    """
    cpu_state: dict[str, Any] = {}
    for key, value in optimizer.state_dict().items():
        if key == "state":
            cpu_state[key] = {}
            for param_key, param_value in value.items():
                cpu_state[key][param_key] = {}
                for sub_key, sub_value in param_value.items():
                    if torch.is_tensor(sub_value):
                        cpu_state[key][param_key][sub_key] = sub_value.cpu()
                    else:
                        cpu_state[key][param_key][sub_key] = sub_value
        elif key == "param_groups":
            cpu_state[key] = []
            for param_group in value:
                cpu_param_group = {}
                for param_key, param_value in param_group.items():
                    cpu_param_group[param_key] = param_value
                cpu_state[key].append(cpu_param_group)
        else:
            cpu_state[key] = value
    return cpu_state


def save_state(
    epoch: int,
    generator: torch.Generator,
    mini_batch: int,
    mlp: nn.Module,
    kan_optimizer: torch.optim.Optimizer,
    name: str,
    saved_model_path: Path,
    lstm_nn: nn.Module | None = None,
    lstm_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save model state

    Parameters
    ----------
    epoch : int
        The epoch number
    mini_batch : int
        The mini batch number
    mlp : nn.Module
        The MLP model
    kan_optimizer : torch.optim.Optimizer
        The KAN optimizer (Adam)
    name: str
        The name of the file we're saving
    lstm_nn : nn.Module | None, optional
        The CudaLSTM model, by default None
    lstm_optimizer : torch.optim.Optimizer | None, optional
        The LSTM optimizer (Adadelta), by default None
    """
    mlp_state_dict = {key: value.cpu() for key, value in mlp.state_dict().items()}

    state = {
        "model_state_dict": mlp_state_dict,
        "kan_optimizer_state_dict": _cpu_optimizer_state(kan_optimizer),
        "rng_state": torch.get_rng_state(),
        "data_generator_state": generator.get_state(),
    }
    if lstm_optimizer is not None:
        state["lstm_optimizer_state_dict"] = _cpu_optimizer_state(lstm_optimizer)
    if lstm_nn is not None:
        state["leakance_nn_state_dict"] = {key: value.cpu() for key, value in lstm_nn.state_dict().items()}
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    if mini_batch == -1:
        state["epoch"] = epoch + 1
        state["mini_batch"] = 0
    else:
        state["epoch"] = epoch
        state["mini_batch"] = mini_batch

    torch.save(
        state,
        saved_model_path / f"_{name}_epoch_{state['epoch']}_mb_{state['mini_batch']}.pt",
    )


def log_metrics(
    nse: np.ndarray,
    rmse: np.ndarray,
    kge: np.ndarray,
    epoch: int | None = None,
    mini_batch: int | None = None,
) -> None:
    """
    Logs evaluation metrics in a formatted and readable way.

    Parameters
    ----------
    nse : np.ndarray
        NumPy array of Nash-Sutcliffe Efficiency values.
    rmse : np.ndarray
        NumPy array of Root Mean Squared Error values.
    kge : np.ndarray
        NumPy array of Kling-Gupta Efficiency values.
    epoch : int, optional
        Epoch number for header display.
    mini_batch : int, optional
        Mini batch number for header display.
    """
    if epoch is not None and mini_batch is not None:
        log.info("----------------------------------------")
        log.info(f"Epoch: {epoch:<9} | Mini Batch: {mini_batch:<8} ")
    log.info("----------------------------------------")
    log.info(f"{'Metric':<10} | {'Mean':>12} | {'Median':>12}")
    log.info("----------------------------------------")
    log.info(f"{'NSE':<10} | {np.nanmean(nse):12.4f} | {np.nanmedian(nse[~np.isinf(nse)]):12.4f}")
    log.info(f"{'RMSE':<10} | {np.nanmean(rmse):12.4f} | {np.nanmedian(rmse):12.4f}")
    log.info(f"{'KGE':<10} | {np.nanmean(kge):12.4f} | {np.nanmedian(kge):12.4f}")
    log.info("----------------------------------------")
