"""Pure trapezoidal channel geometry computation.

Computes cross-section properties from Manning's n, Leopold & Maddock power-law
parameters (p, q), discharge Q, and channel slope. Extracted from the core
routing engine (``routing/mmc.py``) so that geometry can be computed without
the full Muskingum-Cunge routing machinery.
"""

from __future__ import annotations

import torch


def compute_trapezoidal_geometry(
    n: torch.Tensor,
    p_spatial: torch.Tensor,
    q_spatial: torch.Tensor,
    discharge: torch.Tensor,
    slope: torch.Tensor,
    depth_lb: float = 0.01,
    bottom_width_lb: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Compute trapezoidal channel geometry from learned parameters.

    Given Manning's roughness (n), Leopold & Maddock width coefficient (p) and
    exponent (q), a representative discharge Q, and channel bed slope, derives
    the full trapezoidal cross-section geometry by inverting Manning's equation
    for flow depth.

    Parameters
    ----------
    n : torch.Tensor
        Manning's roughness coefficient (m^{-1/3} s). Shape ``(N,)``.
    p_spatial : torch.Tensor
        Leopold & Maddock width coefficient. Shape ``(N,)``.
    q_spatial : torch.Tensor
        Leopold & Maddock width-depth exponent (0 = rectangular, 1 = triangular).
        Shape ``(N,)``.
    discharge : torch.Tensor
        Representative discharge per reach (m^3/s). Shape ``(N,)``.
    slope : torch.Tensor
        Channel bed slope (m/m, dimensionless). Shape ``(N,)``.
    depth_lb : float
        Lower bound for computed depth (m). Default 0.01.
    bottom_width_lb : float
        Lower bound for computed bottom width (m). Default 0.01.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with keys:

        - ``depth`` (m)
        - ``top_width`` (m)
        - ``bottom_width`` (m)
        - ``side_slope`` (H:V ratio)
        - ``cross_sectional_area`` (m^2)
        - ``wetted_perimeter`` (m)
        - ``hydraulic_radius`` (m)
        - ``velocity`` (m/s)
    """
    q_eps = q_spatial + 1e-6

    # Invert Manning's equation for trapezoidal depth
    numerator = discharge * n * (q_eps + 1)
    denominator = p_spatial * torch.pow(slope, 0.5)
    depth = torch.clamp(
        torch.pow(
            torch.div(numerator, denominator + 1e-8),
            torch.div(3.0, 5.0 + 3.0 * q_eps),
        ),
        min=depth_lb,
    )

    # Leopold & Maddock power law: top_width = p * depth^q
    top_width = p_spatial * torch.pow(depth, q_eps)

    # Side slope (z:1 horizontal:vertical)
    side_slope = torch.clamp(top_width * q_eps / (2 * depth), min=0.5, max=50.0)

    # Bottom width of trapezoid
    bottom_width = torch.clamp(
        top_width - (2 * side_slope * depth),
        min=bottom_width_lb,
    )

    # Cross-sectional area
    area = (top_width + bottom_width) * depth / 2

    # Wetted perimeter
    wetted_perimeter = bottom_width + 2 * depth * torch.sqrt(1 + side_slope**2)

    # Hydraulic radius
    hydraulic_radius = area / wetted_perimeter

    # Manning's velocity
    velocity = torch.div(1, n) * torch.pow(hydraulic_radius, (2 / 3)) * torch.pow(slope, (1 / 2))

    return {
        "depth": depth,
        "top_width": top_width,
        "bottom_width": bottom_width,
        "side_slope": side_slope,
        "cross_sectional_area": area,
        "wetted_perimeter": wetted_perimeter,
        "hydraulic_radius": hydraulic_radius,
        "velocity": velocity,
    }
