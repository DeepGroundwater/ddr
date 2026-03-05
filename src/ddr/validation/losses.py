"""Differentiable loss functions for KAN bias correction training.

mass_balance_loss: gives φ-KAN direct gradients (bypasses MC routing).
kge_loss: Kling-Gupta Efficiency loss that flows through MC routing.
huber_loss: robust regression loss (quadratic near zero, linear for outliers).
"""

import torch


def mass_balance_loss(
    q_corrected: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mass balance (ρ) loss — gives φ direct gradients bypassing MC.

    MC conserves mass, so total volume at gauge equals total injected volume
    upstream. ρ_g = AUC_pred_g / AUC_obs_g. Loss = mean((ρ - 1)²).

    Parameters
    ----------
    q_corrected : (G, T)
        Bias-corrected discharge at gauge locations.
    target : (G, T)
        Observed discharge at gauge locations.
    eps : float
        Small constant to prevent division by zero.

    Returns
    -------
    loss : scalar tensor
    """
    auc_pred = q_corrected.sum(dim=1)  # (G,)
    auc_obs = target.sum(dim=1)  # (G,)
    rho = auc_pred / (auc_obs + eps)
    return ((rho - 1.0) ** 2).mean()


def kge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Differentiable KGE loss.

    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    Loss = mean(sqrt((r-1)² + (α-1)² + (β-1)² + eps))

    The eps inside the sqrt prevents NaN gradients at the ideal point.
    The eps in denominators prevents division by zero for constant series.

    Parameters
    ----------
    pred : (G, T)
        Predicted discharge at gauge locations.
    target : (G, T)
        Observed discharge at gauge locations.
    eps : float
        Stabilization constant.

    Returns
    -------
    loss : scalar tensor
    """
    mu_pred = pred.mean(dim=1)  # (G,)
    mu_obs = target.mean(dim=1)  # (G,)
    sigma_pred = pred.std(dim=1, correction=0)  # (G,)
    sigma_obs = target.std(dim=1, correction=0)  # (G,)

    # Correlation
    pred_anom = pred - mu_pred.unsqueeze(1)
    obs_anom = target - mu_obs.unsqueeze(1)
    cov = (pred_anom * obs_anom).mean(dim=1)  # (G,)
    r = cov / (sigma_pred * sigma_obs + eps)  # (G,)

    # Variability ratio
    alpha = sigma_pred / (sigma_obs + eps)  # (G,)

    # Bias ratio
    beta = mu_pred / (mu_obs + eps)  # (G,)

    # Euclidean distance from ideal (1, 1, 1)
    ed = torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + eps)  # (G,)
    return ed.mean()


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber loss averaged across gauges and time.

    Quadratic for errors < delta, linear for errors >= delta.
    Provides stronger raw-magnitude gradients than KGE while being
    robust to outlier flood events.

    Parameters
    ----------
    pred : (G, T)
        Predicted discharge at gauge locations.
    target : (G, T)
        Observed discharge at gauge locations.
    delta : float
        Threshold where loss transitions from quadratic to linear.

    Returns
    -------
    loss : scalar tensor
    """
    return torch.nn.functional.huber_loss(pred, target, delta=delta)
