"""Integration tests for φ-KAN bias correction pipeline."""

import torch

from ddr.nn.kan import kan
from ddr.nn.temporal_phi_kan import TemporalPhiKAN
from ddr.validation.configs import BiasCorrection
from ddr.validation.enums import PhiInputs
from ddr.validation.losses import kge_loss, mass_balance_loss


class TestEndToEndSynthetic:
    """End-to-end synthetic data through φ-KAN."""

    def test_output_shape_and_non_negative(self) -> None:
        cfg = BiasCorrection(phi_inputs=PhiInputs.MONTHLY)
        phi_kan = TemporalPhiKAN(cfg=cfg, seed=42, device="cpu")
        T, N = 48, 10
        q_prime = torch.rand(T, N)
        month = torch.full((T,), 6.0)
        output = phi_kan(q_prime, month=month)

        assert output.shape == (T, N)
        assert (output >= 1e-6).all(), "Output should be non-negative"

    def test_with_grid_bounds_from_spatial_kan(self) -> None:
        """Full pipeline: Spatial KAN → grid_bounds → φ-KAN."""
        bias_cfg = BiasCorrection(phi_inputs=PhiInputs.STATIC)
        spatial_kan = kan(
            input_var_names=["attr1", "attr2", "attr3"],
            learnable_parameters=["n", "q_spatial"],
            hidden_size=7,
            num_hidden_layers=1,
            grid=3,
            k=3,
            seed=42,
            device="cpu",
            output_grid_bounds=True,
            bias_cfg=bias_cfg,
        )
        phi_kan = TemporalPhiKAN(cfg=bias_cfg, seed=42, device="cpu")

        N = 5
        inputs = torch.rand(N, 3)
        spatial_params = spatial_kan(inputs=inputs)

        assert "grid_bounds" in spatial_params
        grid_bounds = spatial_params.pop("grid_bounds")
        assert grid_bounds.shape == (N, 2)
        assert (grid_bounds[:, 1] > grid_bounds[:, 0]).all(), "max should be > min"

        T = 24
        q_prime = torch.rand(T, N)
        output = phi_kan(q_prime, grid_bounds=grid_bounds)
        assert output.shape == (T, N)


class TestGradientFlow:
    """Verify gradients flow correctly through the combined loss."""

    def test_combined_loss_grads_reach_both_networks(self) -> None:
        """Both φ-KAN and Spatial KAN should receive gradients from combined loss."""
        bias_cfg = BiasCorrection(phi_inputs=PhiInputs.STATIC)
        spatial_kan = kan(
            input_var_names=["a", "b", "c"],
            learnable_parameters=["n", "q_spatial"],
            hidden_size=7,
            num_hidden_layers=1,
            grid=3,
            k=3,
            seed=42,
            device="cpu",
            output_grid_bounds=True,
            bias_cfg=bias_cfg,
        )
        phi_kan = TemporalPhiKAN(cfg=bias_cfg, seed=42, device="cpu")

        N = 5
        inputs = torch.rand(N, 3)
        spatial_params = spatial_kan(inputs=inputs)
        grid_bounds = spatial_params.pop("grid_bounds")

        T = 12
        q_prime = torch.rand(T, N, requires_grad=True)
        q_corrected = phi_kan(q_prime, grid_bounds=grid_bounds)

        # Simulate MC routing as a simple linear transform (preserves gradient flow)
        mc_layer = torch.nn.Linear(N, N, bias=False)
        q_routed = mc_layer(q_corrected)

        target = torch.rand(T, N)

        # Combined loss
        mb = mass_balance_loss(q_corrected, target)
        kge = kge_loss(q_routed, target)
        loss = 0.5 * mb + 0.5 * kge
        loss.backward()

        # φ-KAN should have gradients (from both losses)
        phi_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in phi_kan.parameters())
        assert phi_has_grad, "φ-KAN should receive gradients from combined loss"

        # Spatial KAN should have gradients (from kge_loss through MC)
        spatial_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in spatial_kan.parameters()
        )
        assert spatial_has_grad, "Spatial KAN should receive gradients from kge_loss"

    def test_mass_balance_does_not_grad_mc(self) -> None:
        """mass_balance_loss on pre-MC output must not send gradients to MC."""
        mc_layer = torch.nn.Linear(5, 5, bias=False)
        q_corrected = torch.rand(12, 5, requires_grad=True)
        target = torch.rand(12, 5)

        # MC processes q_corrected but loss is computed on pre-MC values
        _q_routed = mc_layer(q_corrected)
        loss = mass_balance_loss(q_corrected, target)
        loss.backward()

        assert mc_layer.weight.grad is None, "MC should not receive gradients from mass_balance_loss"
        assert q_corrected.grad is not None, "q_corrected should receive gradients"

    def test_kge_loss_grads_reach_both(self) -> None:
        """kge_loss through MC should reach both φ-KAN and MC params."""
        bias_cfg = BiasCorrection(phi_inputs=PhiInputs.STATIC)
        phi_kan = TemporalPhiKAN(cfg=bias_cfg, seed=42, device="cpu")

        T, N = 12, 5
        q_prime = torch.rand(T, N, requires_grad=True)
        q_corrected = phi_kan(q_prime)

        mc_layer = torch.nn.Linear(N, N, bias=False)
        q_routed = mc_layer(q_corrected)

        target = torch.rand(T, N)
        loss = kge_loss(q_routed, target)
        loss.backward()

        phi_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in phi_kan.parameters())
        mc_has_grad = mc_layer.weight.grad is not None and mc_layer.weight.grad.abs().sum() > 0
        assert phi_has_grad, "φ-KAN should receive gradients through MC from kge_loss"
        assert mc_has_grad, "MC should receive gradients from kge_loss"


class TestBackwardCompatibility:
    """Verify bias disabled preserves existing behavior."""

    def test_kan_without_bounds_unchanged(self) -> None:
        """kan with output_grid_bounds=False should behave identically to before."""
        nn = kan(
            input_var_names=["a", "b", "c"],
            learnable_parameters=["n", "q_spatial"],
            hidden_size=7,
            num_hidden_layers=1,
            grid=3,
            k=3,
            seed=42,
            device="cpu",
        )
        N = 5
        inputs = torch.rand(N, 3)
        outputs = nn(inputs=inputs)

        assert "n" in outputs
        assert "q_spatial" in outputs
        assert "grid_bounds" not in outputs
        assert outputs["n"].shape == (N,)
        assert outputs["q_spatial"].shape == (N,)

    def test_kan_with_bounds_still_produces_routing_params(self) -> None:
        """With bounds enabled, routing params should still be present and valid."""
        bias_cfg = BiasCorrection()
        nn = kan(
            input_var_names=["a", "b", "c"],
            learnable_parameters=["n", "q_spatial"],
            hidden_size=7,
            num_hidden_layers=1,
            grid=3,
            k=3,
            seed=42,
            device="cpu",
            output_grid_bounds=True,
            bias_cfg=bias_cfg,
        )
        N = 5
        inputs = torch.rand(N, 3)
        outputs = nn(inputs=inputs)

        assert "n" in outputs
        assert "q_spatial" in outputs
        assert "grid_bounds" in outputs
        # Routing params should still be in [0, 1] (sigmoid)
        assert (outputs["n"] >= 0).all() and (outputs["n"] <= 1).all()
        assert (outputs["q_spatial"] >= 0).all() and (outputs["q_spatial"] <= 1).all()
