"""Tests for the multi-component hydrograph loss (peak, baseflow, timing)."""

import torch

from ddr.validation.losses import _regime_loss, _timing_loss, hydrograph_loss


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


class TestPerGageNormalization:
    """Proportional errors → similar loss regardless of basin magnitude."""

    def test_equal_relative_error_similar_loss(self) -> None:
        """With small eps, proportional scaling gives nearly identical loss."""
        small = torch.linspace(1.0, 10.0, 100).unsqueeze(0)
        small_pred = small + 0.5

        # Large basin: same shape scaled 100x, error scaled proportionally
        large = small * 100.0
        large_pred = large + 50.0

        # Use small eps so it doesn't dominate the variance denominator
        loss_small = hydrograph_loss(small_pred, small, eps=1e-6)
        loss_large = hydrograph_loss(large_pred, large, eps=1e-6)

        # Should be very close when eps is negligible
        assert torch.isclose(loss_small, loss_large, rtol=0.05)


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

    def test_zero_peak_weight(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, -1] += 50.0  # peak error only

        loss_with = hydrograph_loss(pred, obs, peak_weight=1.0, baseflow_weight=0.0, timing_weight=0.0)
        loss_without = hydrograph_loss(pred, obs, peak_weight=0.0, baseflow_weight=0.0, timing_weight=0.0)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)

    def test_zero_baseflow_weight(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]])
        pred = obs.clone()
        pred[0, 0] += 2.0  # baseflow error only

        loss_with = hydrograph_loss(pred, obs, peak_weight=0.0, baseflow_weight=1.0, timing_weight=0.0)
        loss_without = hydrograph_loss(pred, obs, peak_weight=0.0, baseflow_weight=0.0, timing_weight=0.0)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)

    def test_zero_timing_weight(self) -> None:
        obs = torch.tensor([[0.0, 1.0, 5.0, 10.0, 5.0, 1.0, 0.0]])
        pred = torch.tensor([[0.0, 0.0, 1.0, 5.0, 10.0, 5.0, 1.0]])  # shifted

        loss_with = hydrograph_loss(pred, obs, peak_weight=0.0, baseflow_weight=0.0, timing_weight=1.0)
        loss_without = hydrograph_loss(pred, obs, peak_weight=0.0, baseflow_weight=0.0, timing_weight=0.0)

        assert loss_with.item() > 0
        assert torch.isclose(loss_without, torch.tensor(0.0), atol=1e-6)
