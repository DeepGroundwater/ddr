"""Trapezoid-exact kinematic celerity: c = v·β, β = 5/3 − (4/3)·A·√(1+z²)/(T·P).

Ported from ddrs (celerity_beta gates). β limits: → 5/3 as b/y → ∞ (wide
rectangular), → 4/3 as b → 0 at fixed z. β is NON-monotone in b/y and reaches
≈1.07 for narrow sections — it is not bounded below by 4/3.
"""

from typing import Any
from unittest.mock import patch

import torch

from ddr.geometry.trapezoidal import compute_trapezoidal_geometry
from ddr.routing.mmc import MuskingumCunge
from tests.routing.test_utils import (
    create_mock_config,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)


def _beta(bottom_width: torch.Tensor, depth: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Reference β for a trapezoid parameterized by (b, y, z)."""
    top_width = bottom_width + 2.0 * z * depth
    area = (bottom_width + top_width) * depth / 2.0
    wetted_perimeter = bottom_width + 2.0 * depth * torch.sqrt(1.0 + z**2)
    return 5.0 / 3.0 - (4.0 / 3.0) * area * torch.sqrt(1.0 + z**2) / (top_width * wetted_perimeter)


class TestBetaLimits:
    def test_wide_rectangular_limit(self) -> None:
        # b/y → ∞, z → 0: β → 5/3
        beta = _beta(torch.tensor(1e6), torch.tensor(1.0), torch.tensor(0.0))
        assert torch.isclose(beta, torch.tensor(5.0 / 3.0), atol=1e-4)

    def test_triangular_limit(self) -> None:
        # b → 0 at fixed z: A = z·y², P = 2y√(1+z²), T = 2zy
        # → β = 5/3 − (4/3)·(z·y²·√(1+z²))/(2zy·2y√(1+z²)) = 5/3 − 1/3 = 4/3
        beta = _beta(torch.tensor(1e-8), torch.tensor(2.0), torch.tensor(1.5))
        assert torch.isclose(beta, torch.tensor(4.0 / 3.0), atol=1e-4)

    def test_narrow_section_below_four_thirds(self) -> None:
        # β is non-monotone in b/y; near b/y ≈ 2 at z = 0 it dips well below 4/3
        beta = _beta(torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.0))
        assert beta < 4.0 / 3.0

    def test_beta_matches_finite_difference_dQ_dA(self) -> None:
        # c = dQ/dA must equal v·β. Manning: v = n⁻¹ R^(2/3) √S, Q = v·A.
        # Perturb depth; dQ/dA = (dQ/dy)/(dA/dy).
        n, s = 0.03, 1e-3
        b, z = torch.tensor(5.0, dtype=torch.float64), torch.tensor(1.2, dtype=torch.float64)

        def q_and_a(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            tw = b + 2.0 * z * y
            area = (b + tw) * y / 2.0
            wp = b + 2.0 * y * torch.sqrt(1.0 + z**2)
            v = (1.0 / n) * (area / wp) ** (2.0 / 3.0) * s**0.5
            return v * area, area

        y0 = torch.tensor(1.7, dtype=torch.float64)
        eps = 1e-6
        q_hi, a_hi = q_and_a(y0 + eps)
        q_lo, a_lo = q_and_a(y0 - eps)
        dq_da_fd = (q_hi - q_lo) / (a_hi - a_lo)

        q0, a0 = q_and_a(y0)
        v0 = q0 / a0
        beta = _beta(b, y0, z)
        assert torch.isclose(v0 * beta, dq_da_fd, rtol=1e-6)


class TestRoutingUsesBeta:
    def test_route_timestep_celerity_is_not_five_thirds(self) -> None:
        """The routed celerity must reflect β, not the hardcoded 5/3."""
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        captured: dict[str, Any] = {}
        original = MuskingumCunge.calculate_muskingum_coefficients

        def spy(self: Any, length: Any, celerity: Any, x: Any) -> Any:
            captured["celerity"] = celerity
            # discharge as it was when this celerity was computed
            captured["discharge"] = self._discharge_t.clone()
            return original(self, length, celerity, x)

        with patch.object(MuskingumCunge, "calculate_muskingum_coefficients", spy):
            mc.forward()

        # Recompute the Manning velocity at the SAME discharge the captured
        # call used (forward() advances _discharge_t after each step)
        geom = compute_trapezoidal_geometry(
            n=mc.n,
            p_spatial=mc.p_spatial,
            q_spatial=mc.q_spatial,
            discharge=captured["discharge"],
            slope=mc.slope,
            depth_lb=float(mc.depth_lb),
            bottom_width_lb=float(mc.bottom_width_lb),
        )
        # β < 5/3 strictly for any finite trapezoid: celerity must be < v·5/3
        v = torch.clamp(geom["velocity"], min=mc.velocity_lb, max=torch.tensor(15.0))
        assert (captured["celerity"] < v * 5.0 / 3.0).all()
        assert (captured["celerity"] > v * 1.0).all()  # β > 1 on physical channels
