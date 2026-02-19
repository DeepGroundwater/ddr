"""Multi-component hydrograph loss targeting peak amplitude, baseflow, and timing.

Replaces the single NNSE loss (Song et al. 2025, Eq. 10) with three complementary
components that provide stronger gradients to specific physical parameters:
- Peak amplitude (top percentile) → Manning's n + leakance
- Baseflow (bottom percentile) → leakance gate
- Timing (temporal gradients) → Manning's n (wave celerity)

Each component is per-gage normalized by the variance of observations in that
flow regime, so large basins don't dominate.
"""

import torch
from torch import Tensor


def hydrograph_loss(
    pred: Tensor,
    target: Tensor,
    peak_weight: float = 1.0,
    baseflow_weight: float = 1.0,
    timing_weight: float = 0.5,
    peak_percentile: float = 0.98,
    baseflow_percentile: float = 0.30,
    eps: float = 0.1,
) -> Tensor:
    """Multi-component hydrograph loss.

    Parameters
    ----------
    pred : Tensor [N, T]
        Predicted discharge (after warmup slicing).
    target : Tensor [N, T]
        Observed discharge (after warmup slicing). Used for both masking and error.
    peak_weight : float
        Weight for peak amplitude component.
    baseflow_weight : float
        Weight for baseflow component.
    timing_weight : float
        Weight for temporal gradient component.
    peak_percentile : float
        Percentile threshold (0–1) above which timesteps are "peaks".
    baseflow_percentile : float
        Percentile threshold (0–1) below which timesteps are "baseflow".
    eps : float
        Stabilization constant added to variance denominators.

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    if peak_weight > 0:
        loss = loss + peak_weight * _regime_loss(pred, target, target, peak_percentile, high=True, eps=eps)

    if baseflow_weight > 0:
        loss = loss + baseflow_weight * _regime_loss(
            pred, target, target, baseflow_percentile, high=False, eps=eps
        )

    if timing_weight > 0:
        loss = loss + timing_weight * _timing_loss(pred, target, eps=eps)

    return loss


def _regime_loss(
    pred: Tensor,
    target: Tensor,
    obs_for_quantile: Tensor,
    percentile: float,
    high: bool,
    eps: float,
) -> Tensor:
    """Compute per-gage normalized MSE on a flow regime (peak or baseflow).

    Parameters
    ----------
    pred : Tensor [N, T]
    target : Tensor [N, T]
    obs_for_quantile : Tensor [N, T]
        Observations used to compute the percentile threshold (no gradient).
    percentile : float
        Quantile threshold (0–1).
    high : bool
        If True, select timesteps >= threshold (peaks).
        If False, select timesteps <= threshold (baseflow).
    eps : float
        Stabilization constant.

    Returns
    -------
    Tensor
        Scalar loss averaged across gages.
    """
    obs_detached = obs_for_quantile.detach()

    # Per-gage quantile thresholds [N] — no gradient through mask
    thresholds = torch.quantile(obs_detached, percentile, dim=1, keepdim=True)  # [N, 1]

    # Boolean mask [N, T]
    if high:
        mask = obs_detached >= thresholds
    else:
        mask = obs_detached <= thresholds

    mask_float = mask.float()
    count = mask_float.sum(dim=1)  # [N]
    valid = count > 0

    # Per-gage masked variance of observations (population variance)
    masked_target = target.detach() * mask_float
    masked_mean = masked_target.sum(dim=1) / count.clamp(min=1)  # [N]
    masked_var = ((target.detach() - masked_mean.unsqueeze(1)) ** 2 * mask_float).sum(dim=1) / count.clamp(
        min=1
    )  # [N]

    # Per-gage masked MSE, normalized by regime variance
    sq_err = (pred - target) ** 2 * mask_float  # [N, T]
    per_gage_mse = sq_err.sum(dim=1) / count.clamp(min=1)  # [N]
    per_gage_loss = per_gage_mse / (masked_var + eps)  # [N]

    if not valid.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return per_gage_loss[valid].mean()


def _timing_loss(pred: Tensor, target: Tensor, eps: float) -> Tensor:
    """Compute per-gage normalized MSE on temporal gradients (finite differences).

    Temporal gradients capture the rising/falling limb alignment. Manning's n
    controls wave celerity, which directly determines peak timing. This loss
    provides a direct gradient signal for timing errors.

    Parameters
    ----------
    pred : Tensor [N, T]
    target : Tensor [N, T]
    eps : float
        Stabilization constant.

    Returns
    -------
    Tensor
        Scalar loss averaged across gages.
    """
    dpred = pred[:, 1:] - pred[:, :-1]
    dtarget = target[:, 1:] - target[:, :-1]

    # Per-gage variance of observed gradients
    dvar = dtarget.detach().var(dim=1, correction=0) + eps  # [N]

    return ((dpred - dtarget) ** 2 / dvar.unsqueeze(1)).mean()
