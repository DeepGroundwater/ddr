"""Tests for leakance (groundwater-surface water exchange) in routing."""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig

from ddr.routing.mmc import MuskingumCunge, _compute_zeta
from ddr.routing.torch_mc import dmc
from ddr.validation.configs import validate_config
from tests.routing.gradient_utils import (
    find_and_retain_grad,
    find_gradient_tensors,
    get_tensor_names,
)
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    assert_tensor_properties,
    create_mock_config_with_leakance,
    create_mock_nn_with_leakance,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters_with_leakance,
    create_mock_streamflow,
)


class TestComputeZeta:
    """Test the _compute_zeta standalone function."""

    @pytest.fixture
    def zeta_inputs(self) -> dict[str, torch.Tensor]:
        """Create standard inputs for zeta computation."""
        n = 10
        return {
            "q_t": torch.ones(n) * 5.0,
            "n": torch.ones(n) * 0.035,
            "q_spatial": torch.ones(n) * 0.5,
            "s0": torch.ones(n) * 0.001,
            "p_spatial": torch.tensor(21.0),
            "length": torch.ones(n) * 1000.0,
            "K_D": torch.ones(n) * 1e-7,
            "d_gw": torch.ones(n) * 0.5,
            "leakance_factor": torch.ones(n) * 0.5,
        }

    def test_zeta_shape_matches_inputs(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta output shape matches input shapes."""
        zeta = _compute_zeta(**zeta_inputs)
        assert_tensor_properties(zeta, (10,))

    def test_zeta_zero_when_leakance_factor_zero(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta is zero when leakance_factor is zero."""
        zeta_inputs["leakance_factor"] = torch.zeros(10)
        zeta = _compute_zeta(**zeta_inputs)
        assert torch.allclose(zeta, torch.zeros(10)), f"Expected all zeros, got {zeta}"

    def test_zeta_positive_when_depth_gt_d_gw(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta is positive when water depth > groundwater depth (losing stream)."""
        # Set d_gw very negative so depth > d_gw always
        zeta_inputs["d_gw"] = torch.ones(10) * -10.0
        zeta_inputs["leakance_factor"] = torch.ones(10)
        zeta = _compute_zeta(**zeta_inputs)
        assert (zeta > 0).all(), f"Expected all positive (losing stream), got {zeta}"

    def test_zeta_negative_when_depth_lt_d_gw(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta is negative when water depth < groundwater depth (gaining stream)."""
        # Set d_gw very large so depth < d_gw always
        zeta_inputs["d_gw"] = torch.ones(10) * 100.0
        zeta_inputs["leakance_factor"] = torch.ones(10)
        zeta = _compute_zeta(**zeta_inputs)
        assert (zeta < 0).all(), f"Expected all negative (gaining stream), got {zeta}"

    def test_zeta_no_nan_or_inf(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta contains no NaN or Inf values."""
        zeta = _compute_zeta(**zeta_inputs)
        assert_no_nan_or_inf(zeta, "zeta")

    def test_zeta_gradient_flow(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that gradients flow through K_D, d_gw, and leakance_factor."""
        for key in ["K_D", "d_gw", "leakance_factor"]:
            zeta_inputs[key] = zeta_inputs[key].clone().requires_grad_(True)

        zeta = _compute_zeta(**zeta_inputs)
        loss = zeta.sum()
        loss.backward()

        for key in ["K_D", "d_gw", "leakance_factor"]:
            assert zeta_inputs[key].grad is not None, f"Gradient should exist for {key}"
            assert not torch.isnan(zeta_inputs[key].grad).any(), f"Gradient for {key} has NaN"
            assert not torch.isinf(zeta_inputs[key].grad).any(), f"Gradient for {key} has Inf"


class TestLeakanceInRouting:
    """Test leakance integration in MuskingumCunge routing."""

    def test_route_timestep_with_leakance(self) -> None:
        """Test that route_timestep works with leakance enabled."""
        cfg = create_mock_config_with_leakance()
        mc = MuskingumCunge(cfg, device="cpu")

        hydrofabric = create_mock_routing_dataclass(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters_with_leakance(num_reaches=10)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        mapper, _, _ = mc.create_pattern_mapper()
        q_prime_clamp = torch.ones(10) * 2.0

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(10) * 5.0
            result = mc.route_timestep(q_prime_clamp, mapper)

        assert_tensor_properties(result, (10,))
        assert_no_nan_or_inf(result, "route_timestep_leakance")
        assert (result >= mc.discharge_lb).all()

    def test_route_timestep_leakance_reduces_discharge(self) -> None:
        """Test that positive zeta (losing stream) reduces discharge vs no leakance."""
        num_reaches = 5

        # Run WITHOUT leakance
        from tests.routing.test_utils import create_mock_config, create_mock_spatial_parameters

        cfg_no_leak = create_mock_config()
        mc_no_leak = MuskingumCunge(cfg_no_leak, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=num_reaches)
        spatial_params_no_leak = create_mock_spatial_parameters(num_reaches=num_reaches)
        mc_no_leak.setup_inputs(hydrofabric, streamflow, spatial_params_no_leak)
        mapper_no_leak, _, _ = mc_no_leak.create_pattern_mapper()

        # Run WITH leakance (losing stream: d_gw very negative => depth > d_gw => zeta > 0)
        cfg_leak = create_mock_config_with_leakance()
        mc_leak = MuskingumCunge(cfg_leak, device="cpu")
        spatial_params_leak = {
            "n": spatial_params_no_leak["n"].clone(),
            "q_spatial": spatial_params_no_leak["q_spatial"].clone(),
            "K_D": torch.ones(num_reaches) * 0.5,  # Normalized, will be denormalized to [1e-8, 1e-6]
            "d_gw": torch.zeros(num_reaches),  # Normalized 0.0 => d_gw = -2.0 (very negative)
            "leakance_factor": torch.ones(num_reaches),  # Max factor
        }
        mc_leak.setup_inputs(hydrofabric, streamflow, spatial_params_leak)
        mapper_leak, _, _ = mc_leak.create_pattern_mapper()

        q_prime_clamp = torch.ones(num_reaches) * 5.0

        def capture_b(A_values, crow, col, b, lower, unit_diag, device):
            return b * 1.1 + 0.5

        with patch("ddr.routing.mmc.triangular_sparse_solve", side_effect=capture_b) as mock_no:
            mc_no_leak.route_timestep(q_prime_clamp, mapper_no_leak)
            b_no_leak = mock_no.call_args[0][3].clone()

        with patch("ddr.routing.mmc.triangular_sparse_solve", side_effect=capture_b) as mock_leak:
            mc_leak.route_timestep(q_prime_clamp, mapper_leak)
            b_leak = mock_leak.call_args[0][3].clone()

        # With losing stream (zeta > 0), b should be smaller (less lateral inflow contribution)
        # b = c2*i_t + c3*Q_t + c4*(q_prime - zeta) vs b = c2*i_t + c3*Q_t + c4*q_prime
        # The c4*(q_prime - zeta) term should be smaller when zeta > 0
        assert (b_leak <= b_no_leak).all(), (
            f"Leakance (losing stream) should reduce b-vector. b_leak={b_leak}, b_no_leak={b_no_leak}"
        )

    def test_forward_with_leakance(self) -> None:
        """Test full forward pass with leakance enabled."""
        cfg = create_mock_config_with_leakance()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_routing_dataclass(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters_with_leakance(num_reaches=10)

        model.set_progress_info(1, 0)

        kwargs = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow,
            "spatial_parameters": spatial_params,
        }

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(10) * 5.0
            output = model(**kwargs)

        assert isinstance(output, dict)
        assert "runoff" in output
        expected_shape = (1, 24)
        assert_tensor_properties(output["runoff"], expected_shape)
        assert_no_nan_or_inf(output["runoff"], "runoff_leakance")

    def test_leakance_disabled_by_default(self) -> None:
        """Test that leakance is disabled by default (backward compatible)."""
        from tests.routing.test_utils import create_mock_config

        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        assert mc.use_leakance is False
        assert mc.K_D is None
        assert mc.d_gw is None
        assert mc.leakance_factor is None


class TestLeakanceGradientFlow:
    """Test gradient flow through leakance parameters."""

    def test_end_to_end_training_with_leakance(self) -> None:
        """Test that gradients propagate through K_D, d_gw, leakance_factor to KAN weights."""
        cfg = create_mock_config_with_leakance()
        model = dmc(cfg, device="cpu")
        num_reaches = 10
        num_timesteps = 24

        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=num_timesteps, num_reaches=num_reaches)
        nn = create_mock_nn_with_leakance()
        spatial_params = nn(inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device))
        optimizer = torch.optim.Adam(params=nn.parameters(), lr=0.01)

        model.epoch = 1
        model.mini_batch = 0

        kwargs: dict[str, Any] = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow,
            "spatial_parameters": spatial_params,
            "retain_grads": True,
        }

        skip_attrs = ["_content", "_metadata", "_parent"]
        find_and_retain_grad(nn, required=True, skip=skip_attrs)
        find_and_retain_grad(model, required=True, skip=skip_attrs)
        find_and_retain_grad(hydrofabric, required=True, skip=skip_attrs)

        output = model(**kwargs)

        test_modules = [model, nn, hydrofabric]
        modules_names = ["model", "nn", "routing_dataclass"]
        ts = [find_gradient_tensors(obj, skip=skip_attrs) for obj in test_modules]
        init_tensors = [t for ts_ in ts for t in ts_]

        optimizer.zero_grad(False)
        loss = output["runoff"].sum()
        loss.retain_grad()
        loss.backward()
        optimizer.step()

        assert loss.grad is not None, "Loss should have gradients after backward pass"
        assert not torch.isnan(loss.grad).any(), "Loss gradients should not contain NaN"
        assert not torch.isinf(loss.grad).any(), "Loss gradients should not contain infinity"

        ts = [find_gradient_tensors(obj, skip=skip_attrs) for obj in test_modules]
        end_tensors = [t for ts_ in ts for t in ts_]
        ns = [
            get_tensor_names(obj, name=name, skip=skip_attrs)
            for obj, name in zip(test_modules, modules_names, strict=False)
        ]
        names = [n for ns_ in ns for n in ns_]

        assert len(init_tensors) == len(end_tensors)
        assert len(names) == len(init_tensors)

        skip_patterns = ["acts_scale_spline", "edge_actscale"]
        unused_spatial_params: list[str] = []
        for param_name in [
            "n",
            "q_spatial",
            "p_spatial",
            "top_width",
            "side_slope",
            "K_D",
            "d_gw",
            "leakance_factor",
        ]:
            if param_name not in cfg.params.parameter_ranges:
                unused_spatial_params.append(f"spatial_parameters['{param_name}']")

        for name, init, end in zip(names, init_tensors, end_tensors, strict=False):
            assert init.requires_grad == end.requires_grad, (
                f"Tensor {name} requires_grad status should not change during training."
            )
            if end.requires_grad:
                if any(pattern in name for pattern in skip_patterns):
                    continue
                if any(unused_param in name for unused_param in unused_spatial_params):
                    continue
                assert end.grad is not None, f"Tensor {name} should have gradients after backward pass"
                assert not torch.isnan(end.grad).any(), f"Tensor {name} gradients should not contain NaN"
                assert not torch.isinf(end.grad).any(), f"Tensor {name} gradients should not contain infinity"

        # Check runoff output
        assert output["runoff"].grad is not None
        assert not torch.isnan(output["runoff"].grad).any()
        assert not torch.isinf(output["runoff"].grad).any()

        # Check KAN output weights
        assert nn.output.weight.grad is not None
        assert not torch.isnan(nn.output.weight.grad).any()
        assert not torch.isinf(nn.output.weight.grad).any()


class TestLeakanceConfigValidation:
    """Test configuration validation for leakance."""

    def test_use_leakance_true_without_param_ranges_raises(self) -> None:
        """Test that use_leakance=True without K_D/d_gw/leakance_factor in parameter_ranges raises."""
        cfg_dict = {
            "name": "mock",
            "mode": "training",
            "geodataset": "lynker_hydrofabric",
            "data_sources": {
                "geospatial_fabric_gpkg": "mock.gpkg",
                "streamflow": "mock://streamflow/store",
                "conus_adjacency": "mock.zarr",
                "gages_adjacency": "mock.zarr",
                "gages": "mock.csv",
            },
            "params": {
                "parameter_ranges": {"n": [0.01, 0.1], "q_spatial": [0.1, 0.9]},
                "defaults": {"p_spatial": 1.0},
                "attribute_minimums": {
                    "velocity": 0.1,
                    "depth": 0.01,
                    "discharge": 0.001,
                    "bottom_width": 0.1,
                    "slope": 0.0001,
                },
                "tau": 7,
                "use_leakance": True,
            },
            "kan": {
                "input_var_names": ["mock"],
                "learnable_parameters": ["n", "q_spatial", "K_D", "d_gw", "leakance_factor"],
            },
            "s3_region": "us-east-1",
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="params.parameter_ranges"):
            validate_config(DictConfig(cfg_dict), save_config=False)

    def test_use_leakance_true_without_kan_learnable_params_raises(self) -> None:
        """Test that use_leakance=True without leakance params in kan.learnable_parameters raises."""
        cfg_dict = {
            "name": "mock",
            "mode": "training",
            "geodataset": "lynker_hydrofabric",
            "data_sources": {
                "geospatial_fabric_gpkg": "mock.gpkg",
                "streamflow": "mock://streamflow/store",
                "conus_adjacency": "mock.zarr",
                "gages_adjacency": "mock.zarr",
                "gages": "mock.csv",
            },
            "params": {
                "parameter_ranges": {
                    "n": [0.01, 0.1],
                    "q_spatial": [0.1, 0.9],
                    "K_D": [1e-8, 1e-6],
                    "d_gw": [-2.0, 2.0],
                    "leakance_factor": [0.0, 1.0],
                },
                "defaults": {"p_spatial": 1.0},
                "attribute_minimums": {
                    "velocity": 0.1,
                    "depth": 0.01,
                    "discharge": 0.001,
                    "bottom_width": 0.1,
                    "slope": 0.0001,
                },
                "tau": 7,
                "use_leakance": True,
            },
            "kan": {
                "input_var_names": ["mock"],
                "learnable_parameters": ["n", "q_spatial"],  # Missing leakance params
            },
            "s3_region": "us-east-1",
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="kan.learnable_parameters"):
            validate_config(DictConfig(cfg_dict), save_config=False)

    def test_use_leakance_false_is_default(self) -> None:
        """Test that use_leakance defaults to False."""
        from tests.routing.test_utils import create_mock_config

        cfg = create_mock_config()
        assert cfg.params.use_leakance is False
