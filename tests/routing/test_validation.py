"""Validation tests comparing refactored classes with original dmc behavior."""

from unittest.mock import patch

import pytest
import torch

from ddr.routing.dmc import dmc as original_dmc
from ddr.routing.mmc import MuskingunCunge
from ddr.routing.torch_mc import TorchMC
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    assert_tensor_properties,
    create_mock_config,
    create_mock_hydrofabric,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)


class TestRefactoredVsOriginalValidation:
    """Validate that refactored classes behave the same as original dmc."""

    @pytest.fixture
    def setup_models_and_data(self):
        """Setup both original and refactored models with test data."""
        cfg = create_mock_config()

        # Create models
        original_model = original_dmc(cfg, device="cpu")
        refactored_model = TorchMC(cfg, device="cpu")
        core_model = MuskingunCunge(cfg, device="cpu")

        # Create test data
        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        return {
            "original": original_model,
            "refactored": refactored_model,
            "core": core_model,
            "hydrofabric": hydrofabric,
            "streamflow": streamflow,
            "spatial_params": spatial_params,
        }

    def test_initialization_parity(self, setup_models_and_data):
        """Test that initialization produces equivalent models."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]

        # Test device setup
        assert original.device_num == refactored.device_num

        # Test tensor attributes match
        assert torch.equal(original.t, refactored.t)
        assert torch.equal(original.p_spatial, refactored.p_spatial)
        assert torch.equal(original.velocity_lb, refactored.velocity_lb)
        assert torch.equal(original.depth_lb, refactored.depth_lb)
        assert torch.equal(original.discharge_lb, refactored.discharge_lb)
        assert torch.equal(original.bottom_width_lb, refactored.bottom_width_lb)

        # Test parameter bounds
        assert original.parameter_bounds == refactored.parameter_bounds

        # Test initial state
        assert original._discharge_t is None
        assert refactored._discharge_t is None
        assert original.network is None
        assert refactored.network is None
        assert original.n is None
        assert refactored.n is None
        assert original.q_spatial is None
        assert refactored.q_spatial is None

    def test_sparse_operations_parity(self, setup_models_and_data):
        """Test that sparse operations produce identical results."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]

        # Test _sparse_eye
        n = 5
        orig_eye = original._sparse_eye(n)
        refact_eye = refactored._sparse_eye(n)

        assert torch.equal(orig_eye.to_dense(), refact_eye.to_dense())

        # Test _sparse_diag
        diag_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        orig_diag = original._sparse_diag(diag_data)
        refact_diag = refactored._sparse_diag(diag_data)

        assert torch.equal(orig_diag.to_dense(), refact_diag.to_dense())

    def test_fill_op_parity(self, setup_models_and_data):
        """Test that fill_op produces identical results."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]
        hydrofabric = data["hydrofabric"]
        streamflow = data["streamflow"]
        spatial_params = data["spatial_params"]

        # Setup network for both models
        original.network = hydrofabric.adjacency_matrix
        # For refactored model, setup inputs so network is available
        refactored.routing_engine.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Test with same data vector
        data_vector = torch.tensor([0.5, -0.3, 0.1, 0.2, -0.1, 0.4, -0.2, 0.3, 0.0, -0.4])

        orig_result = original.fill_op(data_vector)
        refact_result = refactored.fill_op(data_vector)

        # Results should be identical (within floating point precision)
        assert torch.allclose(orig_result.to_dense(), refact_result.to_dense(), atol=1e-6)

    def test_forward_pass_interface_parity(self, setup_models_and_data):
        """Test that forward pass interfaces are identical."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]
        hydrofabric = data["hydrofabric"]
        streamflow = data["streamflow"]
        spatial_params = data["spatial_params"]

        # Set progress tracking
        original.epoch = refactored.epoch = 1
        original.mini_batch = refactored.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock the solver to return deterministic results
        mock_solution = torch.ones(10) * 5.0

        with (
            patch("ddr.routing.dmc.triangular_sparse_solve") as orig_mock,
            patch("ddr.routing.mmc.triangular_sparse_solve") as refact_mock,
        ):
            orig_mock.return_value = mock_solution
            refact_mock.return_value = mock_solution

            orig_output = original(**kwargs)
            refact_output = refactored(**kwargs)

        # Both should return dict with 'runoff' key
        assert isinstance(orig_output, dict)
        assert isinstance(refact_output, dict)
        assert "runoff" in orig_output
        assert "runoff" in refact_output

        # Shapes should match
        assert orig_output["runoff"].shape == refact_output["runoff"].shape

        # Values should be very close (may have small numerical differences)
        assert torch.allclose(orig_output["runoff"], refact_output["runoff"], atol=1e-6)

    def test_parameter_setup_parity(self, setup_models_and_data):
        """Test that parameter setup produces identical results."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]
        hydrofabric = data["hydrofabric"]
        streamflow = data["streamflow"]
        spatial_params = data["spatial_params"]

        # Set progress tracking
        original.epoch = refactored.epoch = 1
        original.mini_batch = refactored.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock solver to avoid running full simulation
        with (
            patch("ddr.routing.dmc.triangular_sparse_solve") as orig_mock,
            patch("ddr.routing.mmc.triangular_sparse_solve") as refact_mock,
        ):
            orig_mock.return_value = torch.ones(10) * 5.0
            refact_mock.return_value = torch.ones(10) * 5.0

            original(**kwargs)
            refactored(**kwargs)

        # Check that parameters were set up identically
        assert torch.equal(original.network, refactored.network)
        assert torch.allclose(original.n, refactored.n, atol=1e-6)
        assert torch.allclose(original.q_spatial, refactored.q_spatial, atol=1e-6)
        assert torch.allclose(original._discharge_t, refactored._discharge_t, atol=1e-6)

    def test_state_management_parity(self, setup_models_and_data):
        """Test that state management behaves identically."""
        data = setup_models_and_data
        original = data["original"]
        refactored = data["refactored"]
        hydrofabric = data["hydrofabric"]
        streamflow = data["streamflow"]
        spatial_params = data["spatial_params"]

        # Set progress tracking
        original.epoch = refactored.epoch = 1
        original.mini_batch = refactored.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Track state changes through multiple calls to solver
        call_count = 0

        def mock_solve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return torch.ones(10) * (5.0 + call_count * 0.1)  # Slightly different each time

        with (
            patch("ddr.routing.dmc.triangular_sparse_solve", side_effect=mock_solve),
            patch("ddr.routing.mmc.triangular_sparse_solve", side_effect=mock_solve),
        ):
            call_count = 0  # Reset for original
            original(**kwargs)
            orig_final_discharge = original._discharge_t.clone()

            call_count = 0  # Reset for refactored
            refactored(**kwargs)
            refact_final_discharge = refactored._discharge_t.clone()

        # Final discharge states should be identical
        assert torch.allclose(orig_final_discharge, refact_final_discharge, atol=1e-6)


class TestCoreMuskingunCungeValidation:
    """Test core MuskingunCunge class against original behavior."""

    @pytest.fixture
    def setup_core_test(self):
        """Setup core model test."""
        cfg = create_mock_config()
        core_model = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=8)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=8)
        spatial_params = create_mock_spatial_parameters(num_reaches=8)

        return core_model, hydrofabric, streamflow, spatial_params

    def test_core_setup_and_forward(self, setup_core_test):
        """Test core model setup and forward pass."""
        core_model, hydrofabric, streamflow, spatial_params = setup_core_test

        # Test setup
        core_model.setup_inputs(hydrofabric, streamflow, spatial_params)
        core_model.set_progress_info(1, 0)

        # Verify setup worked
        assert core_model.hydrofabric is not None
        assert core_model.n is not None
        assert core_model.q_spatial is not None
        assert core_model._discharge_t is not None

        # Test forward pass
        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(8) * 6.0

            output = core_model.forward()

        # Check output
        expected_shape = (2, 12)  # 2 gauges, 12 timesteps
        assert_tensor_properties(output, expected_shape)
        assert_no_nan_or_inf(output, "core_forward_output")

    def test_core_encapsulation(self, setup_core_test):
        """Test that core model properly encapsulates data."""
        core_model, hydrofabric, streamflow, spatial_params = setup_core_test

        # Setup inputs
        core_model.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Test that data is properly stored and accessible
        assert torch.equal(core_model.network, hydrofabric.adjacency_matrix)
        assert torch.equal(core_model.q_prime, streamflow)
        assert core_model.spatial_parameters == spatial_params

        # Test that spatial attributes are properly extracted
        assert_tensor_properties(core_model.length, (8,))
        assert_tensor_properties(core_model.slope, (8,))
        assert_tensor_properties(core_model.top_width, (8,))
        assert_tensor_properties(core_model.side_slope, (8,))
        assert_tensor_properties(core_model.x_storage, (8,))

        # Test that parameters are properly denormalized
        assert_tensor_properties(core_model.n, (8,))
        assert_tensor_properties(core_model.q_spatial, (8,))

        # Values should be in physical ranges (denormalized)
        cfg = core_model.cfg
        n_bounds = cfg.params.parameter_ranges.range.n
        q_bounds = cfg.params.parameter_ranges.range.q_spatial

        assert (core_model.n >= n_bounds[0]).all()
        assert (core_model.n <= n_bounds[1]).all()
        assert (core_model.q_spatial >= q_bounds[0]).all()
        assert (core_model.q_spatial <= q_bounds[1]).all()


class TestBackwardCompatibilityValidation:
    """Test backward compatibility with existing training/evaluation scripts."""

    def test_training_script_compatibility(self):
        """Test compatibility with training script usage patterns."""
        cfg = create_mock_config()

        # This is how dmc is used in train.py
        routing_model = TorchMC(cfg=cfg, device="cpu")  # Using TorchMC as dmc

        # Test epoch and mini_batch assignment (from train.py)
        routing_model.epoch = 1
        routing_model.mini_batch = 0

        assert routing_model.epoch == 1
        assert routing_model.mini_batch == 0

        # Test forward pass with kwargs (from train.py)
        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)

        dmc_kwargs = {
            "hydrofabric": hydrofabric,
            "spatial_parameters": spatial_params,
            "streamflow": streamflow,
        }

        with patch.object(routing_model.routing_engine, "forward") as mock_forward:
            mock_forward.return_value = torch.ones(2, 12) * 5.0

            dmc_output = routing_model(**dmc_kwargs)

        # Should return dict with 'runoff' key
        assert isinstance(dmc_output, dict)
        assert "runoff" in dmc_output

        # Test accessing routing parameters (from train.py)
        # This line exists in train.py: routing_model.n.detach().cpu()
        assert hasattr(routing_model, "n")
        if routing_model.n is not None:
            # Should be able to detach and move to CPU
            n_detached = routing_model.n.detach().cpu()
            assert isinstance(n_detached, torch.Tensor)

    def test_evaluation_script_compatibility(self):
        """Test compatibility with evaluation script usage patterns."""
        cfg = create_mock_config()

        # This is how dmc is used in eval.py
        routing_model = TorchMC(cfg=cfg, device="cpu")

        # Test setting epoch (from eval.py)
        routing_model.epoch = 5  # cfg.eval.epoch

        assert routing_model.epoch == 5

        # Test the same forward pass pattern as training
        hydrofabric = create_mock_hydrofabric(num_reaches=3)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=3)
        spatial_params = create_mock_spatial_parameters(num_reaches=3)

        # In eval.py, there's a different streamflow preparation:
        # q_prime = streamflow_predictions["streamflow"] @ hydrofabric.transition_matrix
        # We'll mock this pattern

        dmc_kwargs = {
            "hydrofabric": hydrofabric,
            "spatial_parameters": spatial_params,
            "streamflow": streamflow,  # Already processed
        }

        with patch.object(routing_model.routing_engine, "forward") as mock_forward:
            mock_forward.return_value = torch.ones(2, 24) * 3.5

            dmc_output = routing_model(**dmc_kwargs)

        # Same output format expected
        assert isinstance(dmc_output, dict)
        assert "runoff" in dmc_output
        assert_tensor_properties(dmc_output["runoff"], (2, 24))

    def test_import_compatibility(self):
        """Test that imports work as expected."""
        # Test original import still works
        from ddr.routing.dmc import dmc as original_dmc_import

        assert original_dmc_import is original_dmc

        # Test that new TorchMC can be imported as dmc alias
        from ddr.routing.torch_mc import dmc as torch_mc_alias

        assert torch_mc_alias is TorchMC

        # Test that new classes can be imported directly
        from ddr.routing.mmc import MuskingunCunge as MC
        from ddr.routing.torch_mc import TorchMC as TMC

        assert MC is MuskingunCunge
        assert TMC is TorchMC

    def test_device_management_compatibility(self):
        """Test device management compatibility."""
        cfg = create_mock_config()

        # Test original usage pattern
        routing_model = TorchMC(cfg=cfg, device="cpu")
        assert routing_model.device_num == "cpu"

        # Test PyTorch .to() method
        model_moved = routing_model.to("cpu")
        assert model_moved is routing_model
        assert model_moved.device_num == "cpu"

        # Test .cpu() method
        model_cpu = routing_model.cpu()
        assert model_cpu is routing_model
        assert model_cpu.device_num == "cpu"

        # Test that tensors are on correct device
        assert routing_model.t.device.type == "cpu"
        assert routing_model.p_spatial.device.type == "cpu"


class TestPerformanceValidation:
    """Test that refactored version maintains performance characteristics."""

    def test_memory_usage_comparable(self):
        """Test that memory usage is comparable."""
        cfg = create_mock_config()

        # Create both models
        _ = original_dmc(cfg, device="cpu")  # Original model for comparison
        refactored = TorchMC(cfg, device="cpu")

        # Both should have similar memory footprint
        # (This is a basic test - in practice you'd use memory profiling tools)

        # Test that refactored model doesn't store duplicate data
        assert refactored.routing_engine is not None
        assert refactored.cfg is refactored.routing_engine.cfg  # Should share config

    def test_forward_pass_efficiency(self):
        """Test that forward pass is efficient."""
        cfg = create_mock_config()
        refactored = TorchMC(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=100)  # Larger network
        streamflow = create_mock_streamflow(num_timesteps=72, num_reaches=100)
        spatial_params = create_mock_spatial_parameters(num_reaches=100)

        refactored.set_progress_info(1, 0)

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock to avoid actual computation
        with patch.object(refactored.routing_engine, "forward") as mock_forward:
            mock_forward.return_value = torch.ones(2, 72) * 5.0

            # This should complete quickly
            output = refactored(**kwargs)

            # Verify mock was called once (no redundant calls)
            mock_forward.assert_called_once()

        assert_tensor_properties(output["runoff"], (2, 72))
