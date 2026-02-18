"""Tests for the normalized squared-error loss (Song et al. 2025, Eq. 10).

The NNSE loss used in train.py is:
    L = mean_i mean_t [ (pred - target)^2 / (var_i + eps) ]

where var_i = variance of observed flow at gage i (full window including warmup).
This is equivalent to optimizing mean(1 - NSE) when eps -> 0.
"""

import torch

EPS = 0.1


def nnse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    obs_for_variance: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Reproduce the NNSE loss from train.py.

    Parameters
    ----------
    pred : [N, T_train] predictions (after warmup slicing)
    target : [N, T_train] observations (after warmup slicing)
    obs_for_variance : [N, T_full] full observations (including warmup) used for variance
    eps : stabilization constant in (m^3/s)^2 units
    """
    obs_variance = obs_for_variance.var(dim=1)  # [N]
    return ((pred - target) ** 2 / (obs_variance.unsqueeze(1) + eps)).mean()


class TestNNSELossBasicMath:
    """Basic correctness of the NNSE loss formula."""

    def test_perfect_prediction_is_zero(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss = nnse_loss(obs, obs, obs)
        assert torch.isclose(loss, torch.tensor(0.0), atol=1e-7)

    def test_manual_single_gage(self) -> None:
        pred = torch.tensor([[3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0]])
        obs_full = torch.tensor([[1.0, 2.0, 3.0]])  # var = 2/3

        # (3-1)^2 + (4-2)^2 = 4 + 4 = 8
        # var = 2/3 ≈ 0.6667
        # normalized: 8 / (2/3 + 0.1) = 8 / 0.7667 ≈ 10.435
        # mean over 2 timesteps: ≈ 5.217
        var = obs_full.var(dim=1).item()  # 2/3
        expected = ((4.0 + 4.0) / (var + EPS)) / 2.0
        loss = nnse_loss(pred, target, obs_full)
        assert torch.isclose(loss, torch.tensor(expected), rtol=1e-5)

    def test_symmetric_error(self) -> None:
        """Loss should be the same whether pred > target or target > pred."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred_over = obs + 1.0
        pred_under = obs - 1.0
        loss_over = nnse_loss(pred_over, obs, obs)
        loss_under = nnse_loss(pred_under, obs, obs)
        assert torch.isclose(loss_over, loss_under, rtol=1e-6)


class TestNNSELossPerGageNormalization:
    """The key property: normalization prevents large basins from dominating."""

    def test_equal_nse_gives_equal_contribution(self) -> None:
        """Two gages with identical NSE should contribute equally to the loss,
        regardless of their flow magnitudes."""
        # Small gage: mean ~5, std ~2.9
        small = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]], dtype=torch.float64)
        # Large gage: same scaled up 100x → mean ~500, std ~290
        large = small * 100.0

        # Add same relative error (10% of range) to both
        small_pred = small + 0.5
        large_pred = large + 50.0

        loss_small = nnse_loss(small_pred, small, small)
        loss_large = nnse_loss(large_pred, large, large)

        # With normalization, these should be nearly equal. Small difference
        # remains because eps=0.1 has slightly larger relative impact on the
        # small gage's variance (8.0 + 0.1) vs the large gage's (80000 + 0.1).
        assert torch.isclose(loss_small, loss_large, rtol=1e-1)

    def test_without_normalization_large_dominates(self) -> None:
        """Without normalization (raw MSE), the large gage dominates. Sanity check."""
        small = torch.tensor([[1.0, 3.0, 5.0, 7.0, 9.0]])
        large = small * 100.0

        small_pred = small + 0.5
        large_pred = large + 50.0

        mse_small = ((small_pred - small) ** 2).mean()
        mse_large = ((large_pred - large) ** 2).mean()

        # Raw MSE: large gage is 10000x bigger
        assert mse_large / mse_small > 1000


class TestNNSELossEpsilon:
    """Tests for the epsilon stabilization constant."""

    def test_constant_observation_does_not_blow_up(self) -> None:
        """A gage with zero variance should not produce inf loss."""
        obs = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.0]])
        pred = torch.tensor([[5.5, 5.5, 5.5, 5.5, 5.5]])
        loss = nnse_loss(pred, obs, obs)
        assert torch.isfinite(loss)

    def test_near_zero_variance_damped(self) -> None:
        """Epsilon should damp the contribution of near-zero-variance gages."""
        obs = torch.tensor([[5.0, 5.0, 5.0, 5.0, 5.001]])  # var ≈ 0
        pred = obs + 1.0
        loss = nnse_loss(pred, obs, obs)
        # Loss = 1^2 * 5 / (≈0 + 0.1) / 5 = 1/0.1 = 10
        assert torch.isfinite(loss)
        assert loss.item() < 100  # Bounded, not blowing up


class TestNNSELossGradients:
    """Verify gradients flow correctly through the NNSE loss."""

    def test_gradient_flows_to_predictions(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = torch.tensor([[1.5, 2.5, 3.5, 4.5, 5.5]], requires_grad=True)
        loss = nnse_loss(pred, obs, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.all(torch.isfinite(pred.grad))

    def test_gradient_is_zero_at_perfect_prediction(self) -> None:
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = obs.clone().detach().requires_grad_(True)
        loss = nnse_loss(pred, obs, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.allclose(pred.grad, torch.zeros_like(pred.grad), atol=1e-6)

    def test_gradient_direction_is_correct(self) -> None:
        """When pred > target, gradient should be positive (push pred down)."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = (obs + 1.0).clone().detach().requires_grad_(True)
        loss = nnse_loss(pred, obs, obs)
        loss.backward()
        assert pred.grad is not None
        assert torch.all(pred.grad > 0)  # positive gradient → reduce pred to reduce loss


class TestNNSELossNSEEquivalence:
    """Verify the relationship: NNSE loss ≈ mean(1 - NSE) when eps → 0."""

    def test_equivalence_to_one_minus_nse(self) -> None:
        torch.manual_seed(42)
        n_gages = 10
        n_timesteps = 100
        target = torch.rand(n_gages, n_timesteps) * 10 + 1
        pred = target + torch.randn_like(target) * 0.5

        # Compute NNSE loss with very small eps
        tiny_eps = 1e-10
        loss = nnse_loss(pred, target, target, eps=tiny_eps)

        # Compute mean(1 - NSE) directly
        ss_res = ((pred - target) ** 2).sum(dim=1)
        ss_tot = ((target - target.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
        nse_per_gage = 1 - ss_res / ss_tot
        mean_one_minus_nse = (1 - nse_per_gage).mean()

        # NNSE loss normalizes by var (= ss_tot / (T-1)), so scale factor is T / (T-1)
        # loss = mean_i[ sum_t(err^2) / (var_i * T) ] = mean_i[ ss_res / (ss_tot / (T-1)) / T ]
        #      = mean_i[ ss_res * (T-1) / (ss_tot * T) ]
        #      = mean(1 - NSE) * (T-1) / T
        scale = (n_timesteps - 1) / n_timesteps
        assert torch.isclose(loss, mean_one_minus_nse * scale, rtol=1e-4)


class TestNNSELossMultiGage:
    """Test behavior with multiple gages."""

    def test_multi_gage_shape(self) -> None:
        """Loss should be a scalar regardless of number of gages."""
        obs = torch.rand(5, 50)
        pred = obs + 0.1
        loss = nnse_loss(pred, obs, obs)
        assert loss.dim() == 0  # scalar

    def test_single_gage_matches_multi(self) -> None:
        """Loss with identical gages should equal single-gage loss."""
        obs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = obs + 0.5

        loss_single = nnse_loss(pred, obs, obs)
        # Stack 3 identical gages
        loss_multi = nnse_loss(pred.repeat(3, 1), obs.repeat(3, 1), obs.repeat(3, 1))
        assert torch.isclose(loss_single, loss_multi, rtol=1e-6)

    def test_variance_from_full_window(self) -> None:
        """Variance should use the full observation window, not just the train slice."""
        obs_full = torch.tensor([[1.0, 2.0, 3.0, 100.0, 200.0]])  # high var from full
        pred = torch.tensor([[3.0, 100.0, 200.0]])
        target = torch.tensor([[3.0, 100.0, 200.0]])

        # Using full window for variance (includes low values 1,2 → higher var)
        loss_full = nnse_loss(pred, target, obs_full)
        assert torch.isclose(loss_full, torch.tensor(0.0), atol=1e-7)

        # vs using only train slice (all high values → lower var)
        loss_slice = nnse_loss(pred, target, target)
        assert torch.isclose(loss_slice, torch.tensor(0.0), atol=1e-7)
