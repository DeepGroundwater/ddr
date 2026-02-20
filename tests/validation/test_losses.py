"""Tests for the multi-component hydrograph loss (overall, peak, baseflow, timing)."""

import torch

from ddr.validation.losses import (
    _overall_loss,
    _regime_loss,
    _timing_loss,
    hydrograph_loss,
)


class TestBasicMath:
    """Fundamental correctness of hydrograph_loss."""

    def test_perfect_prediction_is_zero(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        loss = hydrograph_loss(obs, obs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_returns_scalar(self) -> None:
        obs = torch.rand(5, 50)
        pred = obs + 0.1
        loss = hydrograph_loss(pred, obs)
        assert loss.dim() == 0

    def test_positive_for_imperfect(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        pred = obs + 1.0
        loss = hydrograph_loss(pred, obs)
        assert loss.item() > 0


class TestPeakAmplitude:
    """Peak component targets high-flow timesteps only."""

    def test_low_flow_error_does_not_affect_peak(self) -> None:
        """Error only in the low-flow portion should not change peak loss."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])

        # Perfect prediction everywhere
        pred_perfect = obs.clone()
        l_peak_perfect = _regime_loss(pred_perfect, obs, obs, 0.98, high=True, eps=0.1)

        # Error only in low flows (obs <= P30 ≈ 3.7)
        pred_low_err = obs.clone()
        pred_low_err[0, 0] += 5.0  # obs=1 is low flow
        pred_low_err[0, 1] += 5.0  # obs=2 is low flow
        l_peak_low_err = _regime_loss(pred_low_err, obs, obs, 0.98, high=True, eps=0.1)

        assert torch.isclose(l_peak_perfect, l_peak_low_err, atol=1e-6)

    def test_high_flow_error_affects_peak(self) -> None:
        """Error at the peak should produce nonzero peak loss."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, -1] += 10.0  # error at the peak
        l_peak = _regime_loss(pred, obs, obs, 0.98, high=True, eps=0.1)
        assert l_peak.item() > 0


class TestBaseflow:
    """Baseflow component targets low-flow timesteps only."""

    def test_high_flow_error_does_not_affect_baseflow(self) -> None:
        """Error only at peaks should not change baseflow loss."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])

        pred_perfect = obs.clone()
        l_base_perfect = _regime_loss(pred_perfect, obs, obs, 0.30, high=False, eps=0.1)

        # Error only at the peak
        pred_peak_err = obs.clone()
        pred_peak_err[0, -1] += 50.0
        l_base_peak_err = _regime_loss(pred_peak_err, obs, obs, 0.30, high=False, eps=0.1)

        assert torch.isclose(l_base_perfect, l_base_peak_err, atol=1e-6)

    def test_low_flow_error_affects_baseflow(self) -> None:
        """Error in low flows should produce nonzero baseflow loss."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, 0] += 2.0  # error at baseflow
        l_base = _regime_loss(pred, obs, obs, 0.30, high=False, eps=0.1)
        assert l_base.item() > 0


class TestOverall:
    """Overall MSE component covers all timesteps with strong un-shrunk gradients."""

    def test_perfect_prediction_is_zero(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        assert torch.isclose(_overall_loss(obs, obs, eps=0.1), torch.tensor(0.0), atol=1e-6)

    def test_matches_per_gage_mse(self) -> None:
        """Should reproduce per-gage MSE averaged across gages."""
        pred = torch.tensor([[3.0, 4.0, 5.0]])
        target = torch.tensor([[1.0, 2.0, 3.0]])
        expected = ((pred - target) ** 2).mean(dim=1).mean().item()
        loss = _overall_loss(pred, target, eps=0.1)
        assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)

    def test_covers_midrange(self) -> None:
        """Error only in mid-range flows should produce nonzero overall loss."""
        obs = torch.tensor([[1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0, 100.0]])
        pred = obs.clone()
        pred[0, 4] += 3.0  # error at mid-range (obs=7)
        assert _overall_loss(pred, obs, eps=0.1).item() > 0


class TestTiming:
    """Timing component targets temporal gradient alignment."""

    def test_shifted_hydrograph_nonzero(self) -> None:
        """A shifted hydrograph has different gradients → nonzero timing loss."""
        obs = torch.tensor([[0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0]])
        # Shift right by 1 step
        pred = torch.tensor([[0.0, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0]])
        l_timing = _timing_loss(pred, obs, eps=0.1)
        assert l_timing.item() > 0

    def test_perfect_timing_is_zero(self) -> None:
        """Identical hydrographs → zero timing loss."""
        obs = torch.tensor([[0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0]])
        l_timing = _timing_loss(obs, obs, eps=0.1)
        assert torch.isclose(l_timing, torch.tensor(0.0), atol=1e-6)

    def test_scale_invariant_via_normalization(self) -> None:
        """Proportional scaling should give similar timing loss due to normalization."""
        obs_small = torch.tensor([[0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0]])
        obs_large = obs_small * 100.0

        # Same relative shift
        pred_small = torch.tensor([[0.0, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0]])
        pred_large = pred_small * 100.0

        l_small = _timing_loss(pred_small, obs_small, eps=0.1)
        l_large = _timing_loss(pred_large, obs_large, eps=0.1)

        # Should be close (not exact due to eps)
        assert torch.isclose(l_small, l_large, rtol=0.1)


class TestRegimeLossCap:
    """Per-gage regime loss is capped at 10 to prevent variance-collapse blowup."""

    def test_cap_prevents_extreme_values(self) -> None:
        """When regime variance ≈ 0 (identical peak values), loss should be capped."""
        # 1 gage, 100 timesteps. All peaks have value=50 (var=0 in peak regime).
        obs = torch.cat([torch.linspace(1.0, 10.0, 99), torch.tensor([50.0])]).unsqueeze(0)
        pred = obs.clone()
        pred[0, -1] += 100.0  # huge error at the only peak timestep

        loss = _regime_loss(pred, obs, obs, 0.98, high=True, eps=0.1)
        # Without cap this would be 100^2 / 0.1 = 100_000. With cap it's 10.
        assert loss.item() <= 10.0

    def test_cap_does_not_affect_normal_values(self) -> None:
        """Normal loss values (< 10) should pass through unchanged."""
        obs = torch.linspace(1.0, 100.0, 100).unsqueeze(0)
        pred = obs.clone()
        pred[0, -1] += 1.0  # small error at peak
        loss = _regime_loss(pred, obs, obs, 0.98, high=True, eps=0.1)
        # Small error on high-variance regime → loss << 10
        assert loss.item() < 10.0
        assert loss.item() > 0.0


class TestPerGageNormalization:
    """Regime components are per-gage normalized; overall MSE is not."""

    def test_regime_loss_scale_invariant(self) -> None:
        """Regime loss normalizes by regime variance → proportional errors give similar loss."""
        small = torch.linspace(1.0, 10.0, 100).unsqueeze(0)
        large = small * 100.0

        small_pred = small * 1.1
        large_pred = large * 1.1

        loss_small = _regime_loss(small_pred, small, small, 0.30, high=False, eps=1e-6)
        loss_large = _regime_loss(large_pred, large, large, 0.30, high=False, eps=1e-6)

        assert torch.isclose(loss_small, loss_large, rtol=0.05)

    def test_overall_mse_scales_with_magnitude(self) -> None:
        """Overall MSE intentionally scales with basin magnitude (strong gradients)."""
        small = torch.linspace(1.0, 10.0, 100).unsqueeze(0)
        large = small * 100.0

        small_pred = small + 0.5
        large_pred = large + 50.0

        loss_small = _overall_loss(small_pred, small, eps=0.1)
        loss_large = _overall_loss(large_pred, large, eps=0.1)

        # Large basin error (50.0) is 100x small basin error (0.5) → MSE scales by 100^2
        assert loss_large > loss_small * 100


class TestGradients:
    """Verify gradient flow through the loss."""

    def test_gradient_flows(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        pred = (obs + 0.5).clone().detach().requires_grad_(True)
        loss = hydrograph_loss(pred, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_gradient_zero_at_perfect(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        pred = obs.clone().detach().requires_grad_(True)
        loss = hydrograph_loss(pred, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.allclose(pred.grad, torch.zeros_like(pred.grad), atol=1e-5)


class TestEdgeCases:
    """Edge cases: constant obs, single gage, short series."""

    def test_constant_obs_finite(self) -> None:
        """Constant observations (zero variance) should not produce inf."""
        obs = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]])
        pred = obs + 0.1
        loss = hydrograph_loss(pred, obs)
        assert torch.isfinite(loss)

    def test_single_gage(self) -> None:
        obs = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]])
        pred = obs + 0.5
        loss = hydrograph_loss(pred, obs)
        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_short_time_series(self) -> None:
        """Minimum viable series length (2 timesteps for timing diff)."""
        obs = torch.tensor([[1.0, 10.0]])
        pred = torch.tensor([[2.0, 9.0]])
        loss = hydrograph_loss(pred, obs)
        assert torch.isfinite(loss)


class TestWeights:
    """Zero weight should disable a component."""

    _ALL_OFF = {"overall_weight": 0.0, "peak_weight": 0.0, "baseflow_weight": 0.0, "timing_weight": 0.0}

    def test_all_zero_is_zero(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        pred = obs + 1.0
        loss = hydrograph_loss(pred, obs, **self._ALL_OFF)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_zero_overall_weight(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        pred = obs + 1.0

        loss_with = hydrograph_loss(pred, obs, **{**self._ALL_OFF, "overall_weight": 1.0})
        loss_without = hydrograph_loss(pred, obs, **self._ALL_OFF)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)

    def test_zero_peak_weight(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, -1] += 50.0  # peak error only

        loss_with = hydrograph_loss(pred, obs, **{**self._ALL_OFF, "peak_weight": 1.0})
        loss_without = hydrograph_loss(pred, obs, **self._ALL_OFF)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)

    def test_zero_baseflow_weight(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, 0] += 2.0  # baseflow error only

        loss_with = hydrograph_loss(pred, obs, **{**self._ALL_OFF, "baseflow_weight": 1.0})
        loss_without = hydrograph_loss(pred, obs, **self._ALL_OFF)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)

    def test_zero_timing_weight(self) -> None:
        obs = torch.tensor([[0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0]])
        pred = torch.tensor([[0.0, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0]])  # shifted

        loss_with = hydrograph_loss(pred, obs, **{**self._ALL_OFF, "timing_weight": 1.0})
        loss_without = hydrograph_loss(pred, obs, **self._ALL_OFF)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)
