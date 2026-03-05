"""Tests for ddr.validation.losses — differentiable loss functions."""

import torch

from ddr.validation.losses import kge_loss, mass_balance_loss


class TestMassBalanceLoss:
    """Test mass balance (ρ) loss."""

    def test_perfect_volume_match(self) -> None:
        """ρ=1 for identical series → loss=0."""
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = mass_balance_loss(pred, target)
        assert loss.item() < 1e-10

    def test_volume_mismatch(self) -> None:
        """ρ≠1 → loss>0."""
        pred = torch.tensor([[2.0, 4.0, 6.0, 8.0]])  # 2x target volume
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        loss = mass_balance_loss(pred, target)
        assert loss.item() > 0

    def test_batch_dimension(self) -> None:
        """Loss averages over multiple gauges."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 6.0]])  # G=2
        target = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        loss = mass_balance_loss(pred, target)
        assert loss.item() < 1e-10

    def test_gradient_is_finite(self) -> None:
        pred = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = torch.tensor([[2.0, 3.0, 4.0]])
        loss = mass_balance_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_gradient_does_not_flow_through_mc(self) -> None:
        """mass_balance_loss on pre-MC q_corrected must not grad MC params."""
        mc_layer = torch.nn.Linear(4, 4, bias=False)
        q_corrected = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        _mc_output = mc_layer(q_corrected)  # MC processes but loss ignores
        loss = mass_balance_loss(q_corrected, target)
        loss.backward()
        assert mc_layer.weight.grad is None, "MC should not receive gradients from mass_balance_loss"
        assert q_corrected.grad is not None, "q_corrected should receive gradients"


class TestKgeLoss:
    """Test KGE loss."""

    def test_perfect_prediction(self) -> None:
        """Identical pred and target → loss ≈ eps (near zero)."""
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss = kge_loss(pred, target)
        assert loss.item() < 0.01

    def test_poor_prediction(self) -> None:
        """Uncorrelated series → loss > 0."""
        pred = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss = kge_loss(pred, target)
        assert loss.item() > 0.1

    def test_gradient_is_finite(self) -> None:
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]], requires_grad=True)
        target = torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])
        loss = kge_loss(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_constant_prediction_no_nan(self) -> None:
        """Constant prediction (σ=0) should not produce NaN."""
        pred = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0]], requires_grad=True)
        target = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        loss = kge_loss(pred, target)
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_batch_dimension(self) -> None:
        """Loss averages over multiple gauges."""
        pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # G=2
        target = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        loss = kge_loss(pred, target)
        assert loss.item() < 0.01
