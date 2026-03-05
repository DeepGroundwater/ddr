"""Tests for ddr.nn.temporal_phi_kan — temporal φ-KAN bias correction."""

import math

import torch

from ddr.nn.temporal_phi_kan import TemporalPhiKAN
from ddr.validation.configs import BiasCorrection
from ddr.validation.enums import PhiInputs


class TestTemporalPhiKAN:
    """Tests for the TemporalPhiKAN module."""

    def _make_phi_kan(self, phi_inputs: PhiInputs = PhiInputs.MONTHLY) -> TemporalPhiKAN:
        cfg = BiasCorrection(phi_inputs=phi_inputs)
        return TemporalPhiKAN(cfg=cfg, seed=42, device="cpu")

    def test_monthly_output_shape(self) -> None:
        model = self._make_phi_kan(PhiInputs.MONTHLY)
        T, N = 24, 10
        q_prime = torch.rand(T, N)
        month = torch.full((T,), 6.0)
        output = model(q_prime, month=month)

        assert output.shape == (T, N)

    def test_static_output_shape(self) -> None:
        model = self._make_phi_kan(PhiInputs.STATIC)
        T, N = 24, 10
        q_prime = torch.rand(T, N)
        output = model(q_prime)

        assert output.shape == (T, N)

    def test_forcing_output_shape(self) -> None:
        cfg = BiasCorrection(phi_inputs=PhiInputs.FORCING, forcing_var="precip")
        model = TemporalPhiKAN(cfg=cfg, seed=42, device="cpu")
        T, N = 24, 10
        q_prime = torch.rand(T, N)
        forcing = torch.rand(T, N)
        output = model(q_prime, forcing=forcing)

        assert output.shape == (T, N)

    def test_random_output_shape(self) -> None:
        model = self._make_phi_kan(PhiInputs.RANDOM)
        T, N = 24, 10
        q_prime = torch.rand(T, N)
        output = model(q_prime)

        assert output.shape == (T, N)

    def test_non_negative_output(self) -> None:
        model = self._make_phi_kan(PhiInputs.MONTHLY)
        T, N = 48, 5
        q_prime = torch.rand(T, N) * 0.01  # very small values
        month = torch.full((T,), 1.0)
        output = model(q_prime, month=month)

        assert (output >= 1e-6).all(), "Output has values below clamp threshold"

    def test_gradient_flow(self) -> None:
        model = self._make_phi_kan(PhiInputs.MONTHLY)
        T, N = 12, 5
        q_prime = torch.rand(T, N, requires_grad=True)
        month = torch.full((T,), 3.0)
        output = model(q_prime, month=month)

        loss = output.sum()
        loss.backward()

        has_grad = False
        for p in model.parameters():
            if p.grad is not None:
                has_grad = True
                break
        assert has_grad, "No parameter received gradients"

    def test_deterministic_eval(self) -> None:
        model = self._make_phi_kan(PhiInputs.MONTHLY)
        model.eval()
        T, N = 12, 5
        q_prime = torch.rand(T, N)
        month = torch.full((T,), 7.0)

        with torch.no_grad():
            out1 = model(q_prime, month=month)
            out2 = model(q_prime, month=month)

        assert torch.equal(out1, out2)

    def test_with_grid_bounds(self) -> None:
        model = self._make_phi_kan(PhiInputs.MONTHLY)
        T, N = 12, 5
        q_prime = torch.rand(T, N) * 100
        month = torch.full((T,), 6.0)
        grid_bounds = torch.tensor([[0.0, 100.0]] * N)

        output = model(q_prime, month=month, grid_bounds=grid_bounds)
        assert output.shape == (T, N)

    def test_without_grid_bounds(self) -> None:
        model = self._make_phi_kan(PhiInputs.STATIC)
        T, N = 12, 5
        q_prime = torch.rand(T, N)

        output = model(q_prime, grid_bounds=None)
        assert output.shape == (T, N)

    def test_monotonicity(self) -> None:
        """Output should increase when Q' increases (other inputs fixed)."""
        model = self._make_phi_kan(PhiInputs.STATIC)
        model.eval()
        N = 10
        q_low = torch.full((1, N), 0.2)
        q_high = torch.full((1, N), 0.8)
        with torch.no_grad():
            out_low = model(q_low)
            out_high = model(q_high)
        assert (out_high >= out_low).all(), (
            f"Expected monotonic increase: out_low={out_low}, out_high={out_high}"
        )

    def test_sin_cos_encoding(self) -> None:
        """Verify sin/cos month encoding produces expected values."""
        # January: month=1 → sin(2π/12) ≈ 0.5, cos(2π/12) ≈ 0.866
        month = torch.tensor([1.0])
        two_pi_month = 2.0 * math.pi * month / 12.0
        sin_val = torch.sin(two_pi_month)
        cos_val = torch.cos(two_pi_month)

        assert abs(sin_val.item() - 0.5) < 0.01
        assert abs(cos_val.item() - 0.866) < 0.01

        # April: month=4 → sin(2π*4/12) ≈ 0.866, cos ≈ -0.5
        month = torch.tensor([4.0])
        two_pi_month = 2.0 * math.pi * month / 12.0
        assert abs(torch.sin(two_pi_month).item() - 0.866) < 0.01
        assert abs(torch.cos(two_pi_month).item() - (-0.5)) < 0.01
