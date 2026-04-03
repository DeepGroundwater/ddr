"""Compute per-reach temporal statistics from daily accumulated discharge.

Pure computation — no I/O, no Config, no data loading.  Takes pre-computed
KAN parameters and daily accumulated discharge arrays, runs
:func:`compute_trapezoidal_geometry` for each day, and returns per-reach
min/max/median/mean for depth, top_width, bottom_width, side_slope,
hydraulic_radius, and discharge.
"""

from __future__ import annotations

import numpy as np
import torch

from ddr.geometry.trapezoidal import compute_trapezoidal_geometry

_GEOMETRY_VARS = ("depth", "top_width", "bottom_width", "side_slope", "hydraulic_radius")


def compute_geometry_statistics(
    n: torch.Tensor,
    p_spatial: torch.Tensor,
    q_spatial: torch.Tensor,
    slope: torch.Tensor,
    daily_accumulated_discharge: np.ndarray,
    attribute_minimums: dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-reach temporal statistics for channel geometry and discharge.

    Parameters
    ----------
    n : torch.Tensor
        Manning's roughness per reach, shape ``(N,)``.
    p_spatial : torch.Tensor
        Leopold & Maddock width coefficient per reach, shape ``(N,)``.
    q_spatial : torch.Tensor
        Leopold & Maddock width-depth exponent per reach, shape ``(N,)``.
    slope : torch.Tensor
        Channel bed slope per reach, shape ``(N,)``.  Should already be
        clamped to a physical minimum.
    daily_accumulated_discharge : np.ndarray
        Accumulated discharge for each day, shape ``(n_days, N)`` in m³/s.
    attribute_minimums : dict, optional
        Physical lower bounds.  Keys ``"depth"`` and ``"bottom_width"`` are
        forwarded to :func:`compute_trapezoidal_geometry`.

    Returns
    -------
    dict[str, np.ndarray]
        Keys are ``{var}_{min,max,median,mean}`` for each variable in
        ``(depth, top_width, bottom_width, side_slope, hydraulic_radius,
        discharge)``, each of shape ``(N,)``.
    """
    if attribute_minimums is None:
        attribute_minimums = {}
    depth_lb = attribute_minimums.get("depth", 0.01)
    bw_lb = attribute_minimums.get("bottom_width", 0.01)

    n_days, n_reaches = daily_accumulated_discharge.shape
    daily_arrays = {var: np.empty((n_days, n_reaches), dtype=np.float32) for var in _GEOMETRY_VARS}

    for day in range(n_days):
        q_day = torch.tensor(daily_accumulated_discharge[day], dtype=torch.float32)
        geo = compute_trapezoidal_geometry(
            n=n,
            p_spatial=p_spatial,
            q_spatial=q_spatial,
            discharge=q_day,
            slope=slope,
            depth_lb=depth_lb,
            bottom_width_lb=bw_lb,
        )
        for var in _GEOMETRY_VARS:
            daily_arrays[var][day] = geo[var].numpy()

    result: dict[str, np.ndarray] = {}
    for var_name, arr in [*daily_arrays.items(), ("discharge", daily_accumulated_discharge)]:
        result[f"{var_name}_min"] = np.nanmin(arr, axis=0).astype(np.float32)
        result[f"{var_name}_max"] = np.nanmax(arr, axis=0).astype(np.float32)
        result[f"{var_name}_median"] = np.nanmedian(arr, axis=0).astype(np.float32)
        result[f"{var_name}_mean"] = np.nanmean(arr, axis=0).astype(np.float32)

    return result
