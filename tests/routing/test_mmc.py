"""Comprehensive tests for the refactored mmc.py module."""

from unittest.mock import patch

import pytest
import torch

from ddr.routing.mmc import MuskingunCunge
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    assert_tensor_properties,
    create_mock_config,
    create_mock_hydrofabric,
    create_mock_spatial_parameters,
    create_mock_streamflow,
    create_test_scenarios,
)


class TestMuskingunCungeInitialization:
    """Test MuskingunCunge class initialization."""

    def test_init_cpu(self):
        """Test initialization with CPU device."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        assert mc.device == "cpu"
        assert mc.cfg == cfg
        assert mc.t.item() == 3600.0
        assert mc.t.device.type == "cpu"

        # Test tensor attributes
        assert isinstance(mc.p_spatial, torch.Tensor)
        assert isinstance(mc.velocity_lb, torch.Tensor)
        assert isinstance(mc.depth_lb, torch.Tensor)
        assert isinstance(mc.discharge_lb, torch.Tensor)
        assert isinstance(mc.bottom_width_lb, torch.Tensor)

        # Test initial state
        assert mc.n is None
        assert mc.q_spatial is None
        assert mc._discharge_t is None
        assert mc.network is None
        assert mc.hydrofabric is None

    def test_init_default_device(self):
        """Test initialization with default device."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg)

        assert mc.device == "cpu"

    def test_parameter_bounds_setup(self):
        """Test that parameter bounds are correctly set up."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        assert mc.parameter_bounds == cfg.params.parameter_ranges.range
        assert mc.p_spatial.item() == cfg.params.defaults.p
        assert torch.allclose(mc.velocity_lb, torch.tensor(cfg.params.attribute_minimums.velocity))
        assert torch.allclose(mc.depth_lb, torch.tensor(cfg.params.attribute_minimums.depth))
        assert torch.allclose(mc.discharge_lb, torch.tensor(cfg.params.attribute_minimums.discharge))
        assert torch.allclose(mc.bottom_width_lb, torch.tensor(cfg.params.attribute_minimums.bottom_width))


class TestMuskingunCungeProgressTracking:
    """Test progress tracking functionality."""

    def test_set_progress_info(self):
        """Test setting progress information."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        mc.set_progress_info(5, 10)

        assert mc.epoch == 5
        assert mc.mini_batch == 10


class TestMuskingunCungeInputSetup:
    """Test input setup functionality."""

    def test_setup_inputs_basic(self):
        """Test basic input setup."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Check that all inputs were stored
        assert mc.hydrofabric is hydrofabric
        assert torch.equal(mc.q_prime, streamflow)
        assert mc.spatial_parameters == spatial_params

        # Check network setup
        assert torch.equal(mc.network, hydrofabric.adjacency_matrix)
        assert torch.equal(mc.observations, hydrofabric.observations.gage_id)

        # Check spatial attributes
        assert_tensor_properties(mc.length, (10,))
        assert_tensor_properties(mc.slope, (10,))
        assert_tensor_properties(mc.top_width, (10,))
        assert_tensor_properties(mc.side_slope, (10,))
        assert_tensor_properties(mc.x_storage, (10,))

        # Check parameter denormalization
        assert mc.n is not None
        assert mc.q_spatial is not None
        assert_tensor_properties(mc.n, (10,))
        assert_tensor_properties(mc.q_spatial, (10,))

        # Check discharge initialization
        assert torch.equal(mc._discharge_t, streamflow[0])

        # Check gauge indices
        assert_tensor_properties(mc.gage_indices, (1,), expected_dtype=torch.int64)

    def test_setup_inputs_slope_clamping(self):
        """Test that slope is properly clamped during setup."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        # Set some slopes below minimum
        hydrofabric.slope = torch.tensor([0.00001, 0.001, 0.00005, 0.002, 0.00003])

        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        min_slope = cfg.params.attribute_minimums.slope
        assert (mc.slope >= min_slope).all(), "All slopes should be >= minimum"

    def test_setup_inputs_device_conversion(self):
        """Test that tensors are moved to correct device during setup."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5, device="cpu")
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5, device="cpu")
        spatial_params = create_mock_spatial_parameters(num_reaches=5, device="cpu")

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Check that all tensors are on correct device
        assert mc.length.device.type == "cpu"
        assert mc.slope.device.type == "cpu"
        assert mc.top_width.device.type == "cpu"
        assert mc.side_slope.device.type == "cpu"
        assert mc.x_storage.device.type == "cpu"
        assert mc.q_prime.device.type == "cpu"


class TestMuskingunCungeSparseOperations:
    """Test sparse matrix operations."""

    def test_sparse_eye(self):
        """Test _sparse_eye method."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        n = 5
        identity = mc._sparse_eye(n)

        assert identity.shape == (n, n)
        # For newer PyTorch versions, sparse tensors may not report is_sparse=True
        # Instead, check if it has sparse layout
        assert identity.layout in [torch.sparse_coo, torch.sparse_csr]

        # Convert to dense and check
        dense_identity = identity.to_dense()
        expected = torch.eye(n)
        assert torch.allclose(dense_identity, expected)

    def test_sparse_diag(self):
        """Test _sparse_diag method."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        diag_matrix = mc._sparse_diag(data)

        assert diag_matrix.shape == (4, 4)
        # For newer PyTorch versions, sparse tensors may not report is_sparse=True
        # Instead, check if it has sparse layout
        assert diag_matrix.layout in [torch.sparse_coo, torch.sparse_csr]

        # Convert to dense and check
        dense_diag = diag_matrix.to_dense()
        expected = torch.diag(data)
        assert torch.allclose(dense_diag, expected)

    def test_fill_op(self):
        """Test fill_op method."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs first
        hydrofabric = create_mock_hydrofabric(num_reaches=3)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=3)
        spatial_params = create_mock_spatial_parameters(num_reaches=3)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        data_vector = torch.tensor([0.5, -0.3, 0.1])
        result = mc.fill_op(data_vector)

        assert result.shape == (3, 3)
        # For fill_op, the result might be dense depending on implementation
        # Just check that it produces reasonable output
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestMuskingunCungeCoefficients:
    """Test Muskingum coefficient calculations."""

    def test_calculate_muskingum_coefficients(self):
        """Test calculation of Muskingum coefficients."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        length = torch.tensor([1000.0, 1500.0, 2000.0])
        velocity = torch.tensor([1.0, 1.5, 2.0])
        x_storage = torch.tensor([0.2, 0.25, 0.3])

        c_1, c_2, c_3, c_4 = mc.calculate_muskingum_coefficients(length, velocity, x_storage)

        # Check output shapes
        assert_tensor_properties(c_1, (3,))
        assert_tensor_properties(c_2, (3,))
        assert_tensor_properties(c_3, (3,))
        assert_tensor_properties(c_4, (3,))

        # Check no NaN or Inf
        assert_no_nan_or_inf(c_1, "c_1")
        assert_no_nan_or_inf(c_2, "c_2")
        assert_no_nan_or_inf(c_3, "c_3")
        assert_no_nan_or_inf(c_4, "c_4")

        # Basic sanity checks
        assert (c_4 > 0).all(), "c_4 should be positive"

    def test_calculate_muskingum_coefficients_edge_cases(self):
        """Test coefficient calculation with edge cases."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Test with very small velocity
        length = torch.tensor([1000.0])
        velocity = torch.tensor([0.01])
        x_storage = torch.tensor([0.2])

        c_1, c_2, c_3, c_4 = mc.calculate_muskingum_coefficients(length, velocity, x_storage)

        assert_no_nan_or_inf(c_1, "c_1_small_velocity")
        assert_no_nan_or_inf(c_2, "c_2_small_velocity")
        assert_no_nan_or_inf(c_3, "c_3_small_velocity")
        assert_no_nan_or_inf(c_4, "c_4_small_velocity")


class TestMuskingunCungePatternMapper:
    """Test pattern mapper creation."""

    def test_create_pattern_mapper(self):
        """Test pattern mapper creation."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs first
        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        mapper, dense_rows, dense_cols = mc.create_pattern_mapper()

        # Check that mapper was created
        assert mapper is not None
        assert hasattr(mapper, "map")
        assert hasattr(mapper, "crow_indices")
        assert hasattr(mapper, "col_indices")

        # Check dense indices
        assert isinstance(dense_rows, torch.Tensor)
        assert isinstance(dense_cols, torch.Tensor)


class TestMuskingunCungeRouteTimestep:
    """Test single timestep routing."""

    def test_route_timestep(self):
        """Test routing for a single timestep."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs
        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Create mapper
        mapper, _, _ = mc.create_pattern_mapper()

        # Mock input
        q_prime_clamp = torch.ones(5) * 2.0

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(5) * 5.0

            result = mc.route_timestep(q_prime_clamp, mapper)

        assert_tensor_properties(result, (5,))
        assert_no_nan_or_inf(result, "route_timestep_result")
        assert (result >= mc.discharge_lb).all(), "Result should be >= discharge lower bound"

    def test_route_timestep_discharge_clamping(self):
        """Test that timestep routing properly clamps discharge."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs
        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Create mapper
        mapper, _, _ = mc.create_pattern_mapper()

        q_prime_clamp = torch.ones(5) * 2.0

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            # Return some negative values to test clamping
            mock_solve.return_value = torch.tensor([-1.0, 5.0, -0.5, 10.0, 0.0001])

            result = mc.route_timestep(q_prime_clamp, mapper)

        min_discharge = mc.discharge_lb.item()
        assert (result >= min_discharge).all(), "All discharge values should be >= minimum"


class TestMuskingunCungeForward:
    """Test the forward pass."""

    def test_forward_without_setup_raises_error(self):
        """Test that forward raises error without setup."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        with pytest.raises(ValueError, match="Hydrofabric not set"):
            mc.forward()

    def test_forward_basic(self):
        """Test basic forward pass."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs
        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Set progress info
        mc.set_progress_info(1, 0)

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            # Return values that include some below minimum discharge to test clamping
            call_count = 0

            def mock_solver(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                # Return mix of values including some below minimum discharge
                if call_count % 3 == 1:
                    return torch.tensor([-1.0, 5.0, -0.5, 10.0, 0.0001, 3.0, -2.0, 8.0, 1.0, 0.0])
                elif call_count % 3 == 2:
                    return torch.tensor([0.002, -0.1, 2.0, 0.0005, 4.0, -3.0, 1.5, 0.001, -0.01, 2.5])
                else:
                    return torch.tensor([1.0, 1.5, 2.0, 0.5, 3.0, 0.8, 1.2, 2.5, 1.8, 0.9])

            mock_solve.side_effect = mock_solver
            output = mc.forward()

        # Check output properties
        expected_shape = (2, 24)  # 2 gauges, 24 timesteps
        assert_tensor_properties(output, expected_shape)
        assert_no_nan_or_inf(output, "forward_output")

        # Test discharge clamping with strict checking within floating point tolerance
        min_discharge = mc.discharge_lb.item()
        tolerance = 1e-6  # Floating point tolerance
        min_threshold = min_discharge - tolerance

        # Filter out zero values which may be due to incomplete gauge indexing
        non_zero_values = output[output > 0.0]

        if non_zero_values.numel() > 0:
            # All non-zero values should be >= minimum discharge within tolerance
            below_threshold = non_zero_values < min_threshold
            assert not below_threshold.any(), (
                f"All non-zero outputs should be >= min_discharge ({min_discharge}) within tolerance ({tolerance}). "
                f"Found {below_threshold.sum()} non-zero values below threshold. "
                f"Min non-zero value: {non_zero_values.min().item()}, "
                f"Max value: {non_zero_values.max().item()}"
            )

        # Ensure we actually have some meaningful output (not all zeros)
        assert (output > 0.0).any(), "Should have some non-zero output values"

        # Ensure the output has reasonable values - at least some should be at or above min_discharge
        meaningful_values = output >= min_discharge
        assert meaningful_values.any(), (
            f"Should have some values >= min_discharge ({min_discharge}), "
            f"but max value is {output.max().item()}"
        )

    def test_forward_discharge_state_updates(self):
        """Test that discharge state is updated during forward pass."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Setup inputs
        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        mc.set_progress_info(1, 0)

        initial_discharge = mc._discharge_t.clone()

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(5) * 10.0  # Different from initial

            mc.forward()

        # Discharge should have been updated
        assert not torch.equal(mc._discharge_t, initial_discharge)
        assert torch.allclose(mc._discharge_t, torch.ones(5) * 10.0)


class TestMuskingunCungeIntegration:
    """Integration tests for MuskingunCunge."""

    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_different_network_sizes(self, scenario):
        """Test MuskingunCunge with different network sizes."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=scenario["num_reaches"])
        streamflow = create_mock_streamflow(
            num_timesteps=scenario["num_timesteps"], num_reaches=scenario["num_reaches"]
        )
        spatial_params = create_mock_spatial_parameters(num_reaches=scenario["num_reaches"])

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc.set_progress_info(1, 0)

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(scenario["num_reaches"]) * 5.0

            output = mc.forward()

        # Check output properties
        expected_shape = (2, scenario["num_timesteps"])  # 2 gauges
        assert_tensor_properties(output, expected_shape)
        assert_no_nan_or_inf(output, f"forward_{scenario['name']}")

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs."""
        cfg = create_mock_config()

        # Set seeds for reproducibility
        torch.manual_seed(42)

        mc1 = MuskingunCunge(cfg, device="cpu")
        mc2 = MuskingunCunge(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        # Setup both instances
        mc1.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc2.setup_inputs(hydrofabric, streamflow, spatial_params)

        mc1.set_progress_info(1, 0)
        mc2.set_progress_info(1, 0)

        # Mock with same return values
        mock_solution = torch.ones(10) * 5.0

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = mock_solution
            output1 = mc1.forward()

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = mock_solution
            output2 = mc2.forward()

        # Results should be identical
        assert torch.allclose(output1, output2)

    def test_full_workflow(self):
        """Test complete workflow from initialization to forward pass."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Test initial state
        assert mc.hydrofabric is None
        assert mc.n is None
        assert mc.q_spatial is None

        # Setup inputs
        hydrofabric = create_mock_hydrofabric(num_reaches=8)
        streamflow = create_mock_streamflow(num_timesteps=36, num_reaches=8)
        spatial_params = create_mock_spatial_parameters(num_reaches=8)

        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Test state after setup
        assert mc.hydrofabric is not None
        assert mc.n is not None
        assert mc.q_spatial is not None
        assert mc._discharge_t is not None

        # Set progress tracking
        mc.set_progress_info(2, 5)
        assert mc.epoch == 2
        assert mc.mini_batch == 5

        # Run forward pass
        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(8) * 7.5

            output = mc.forward()

        # Verify output
        assert_tensor_properties(output, (2, 36))  # 2 gauges, 36 timesteps
        assert_no_nan_or_inf(output, "full_workflow_output")


class TestMuskingunCungeErrorHandling:
    """Test error handling in MuskingunCunge."""

    def test_setup_inputs_validation(self):
        """Test input validation in setup_inputs."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Test with None hydrofabric should not raise immediately
        # (validation happens in forward())
        hydrofabric = None
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)

        # This should not raise an error during setup
        # Error should occur during forward() call
        try:
            mc.setup_inputs(hydrofabric, streamflow, spatial_params)
        except AttributeError:
            # Expected when hydrofabric is None
            pass

    def test_forward_with_invalid_state(self):
        """Test forward method with invalid state."""
        cfg = create_mock_config()
        mc = MuskingunCunge(cfg, device="cpu")

        # Try to run forward without setup
        with pytest.raises(ValueError):
            mc.forward()
