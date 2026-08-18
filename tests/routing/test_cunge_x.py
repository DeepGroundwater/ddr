"""Cunge-derived Muskingum X: X = clamp(0.5·(1 − Q/(T·S·c·L)), 0, 0.5).

Matches the scheme's numerical diffusion D_num = c·L·(0.5−X) to the channel's
physical diffusivity D_phys = Q/(2·T·S). Replaces the static per-reach x
(MERIT constant 0.3 / Lynker zarr). Fixture warning from ddrs: short reaches
saturate the clamp at 0 and pass vacuously — use ≥5000 m where X must be
interior.
"""

from typing import Any
from unittest.mock import patch

import torch

from ddr.routing.mmc import MuskingumCunge
from tests.routing.test_utils import (
    create_mock_config,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)


def _cunge_x(
    q: torch.Tensor, t: torch.Tensor, s: torch.Tensor, c: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    return torch.clamp(0.5 * (1.0 - q / (t * s * c * length)), min=0.0, max=0.5)


class TestCungeXFormula:
    def test_diffusion_matching_interior(self) -> None:
        # Interior X: D_num = c·L·(0.5−X) must equal D_phys = Q/(2·T·S)
        q, t, s, c, length = (
            torch.tensor(50.0),
            torch.tensor(30.0),
            torch.tensor(1e-3),
            torch.tensor(1.5),
            torch.tensor(5000.0),
        )
        x = _cunge_x(q, t, s, c, length)
        assert 0.0 < x < 0.5
        d_num = c * length * (0.5 - x)
        d_phys = q / (2.0 * t * s)
        assert torch.isclose(d_num, d_phys, rtol=1e-6)

    def test_clamps_to_zero_for_diffusive_channels(self) -> None:
        # Large Q over a short flat reach → raw X negative → clamp at 0
        x = _cunge_x(
            torch.tensor(500.0),
            torch.tensor(10.0),
            torch.tensor(1e-3),
            torch.tensor(1.0),
            torch.tensor(500.0),
        )
        assert x == 0.0

    def test_never_exceeds_half(self) -> None:
        x = _cunge_x(
            torch.tensor(1e-4),
            torch.tensor(50.0),
            torch.tensor(0.01),
            torch.tensor(3.0),
            torch.tensor(10000.0),
        )
        assert x <= 0.5


class TestRoutingUsesCungeX:
    def _spy_forward(self) -> dict[str, Any]:
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        # Long reaches so X is interior, not clamp-saturated (ddrs gotcha)
        hydrofabric.length = torch.full((4,), 5000.0)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        captured: dict[str, Any] = {}
        original = MuskingumCunge.calculate_muskingum_coefficients

        def spy(self: Any, length: Any, celerity: Any, x: Any) -> Any:
            captured["x"] = x
            captured["coeffs"] = original(self, length, celerity, x)
            return captured["coeffs"]

        with patch.object(MuskingumCunge, "calculate_muskingum_coefficients", spy):
            mc.forward()
        return captured

    def test_x_is_dynamic_and_bounded(self) -> None:
        captured = self._spy_forward()
        x = captured["x"]
        assert x.shape == (4,)  # per-reach, computed per timestep
        assert (x >= 0.0).all() and (x <= 0.5).all()

    def test_coefficient_identity(self) -> None:
        # c1 + c2 + c3 = 1 exactly, for ANY (K, X) — mass conservation
        captured = self._spy_forward()
        c1, c2, c3, _c4 = captured["coeffs"]
        assert torch.allclose(c1 + c2 + c3, torch.ones_like(c1), atol=1e-6)

    def test_gradients_flow_through_x(self) -> None:
        # X depends on discharge and geometry; the KAN parameters must receive
        # gradients through the X path (ddrs verified this branch falsifiable).
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        hydrofabric.length = torch.full((4,), 5000.0)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        for v in spatial_params.values():
            v.requires_grad_(True)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        output = mc.forward()
        output.sum().backward()
        for name, v in spatial_params.items():
            assert v.grad is not None, f"no gradient reached spatial parameter {name!r}"
            assert torch.isfinite(v.grad).all()
