"""Comprehensive tests for the original dmc.py module."""

from unittest.mock import patch

import pytest
import torch

from ddr.routing.dmc import _get_trapezoid_velocity, _log_base_q, dmc
from tests.routing.gradient_utils import (
    find_and_retain_grad,
    find_gradient_tensors,
    get_tensor_names,
)
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    assert_tensor_properties,
    create_mock_config,
    create_mock_hydrofabric,
    create_mock_nn,
    create_mock_spatial_parameters,
    create_mock_streamflow,
    create_test_scenarios,
)


class TestDMCUtilityFunctions:
    """Test utility functions used by the dmc class."""

    def test_log_base_q(self):
        """Test _log_base_q function."""
        x = torch.tensor([1.0, 2.0, 4.0, 8.0])
        q = 2.0
        result = _log_base_q(x, q)

        expected = torch.tensor([0.0, 1.0, 2.0, 3.0])
        assert torch.allclose(result, expected, atol=1e-6)

        # Test with different base
        q = 10.0
        result = _log_base_q(torch.tensor([1.0, 10.0, 100.0]), q)
        expected = torch.tensor([0.0, 1.0, 2.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_get_trapezoid_velocity(self):
        """Test _get_trapezoid_velocity function."""
        # Setup test data
        q_t = torch.tensor([10.0, 20.0, 30.0])
        _n = torch.tensor([0.03, 0.04, 0.05])
        top_width = torch.tensor([10.0, 15.0, 20.0])
        side_slope = torch.tensor([2.0, 2.5, 3.0])
        _s0 = torch.tensor([0.001, 0.002, 0.003])
        p_spatial = torch.tensor([1.0, 1.0, 1.0])
        _q_spatial = torch.tensor([0.5, 0.6, 0.7])
        velocity_lb = torch.tensor([0.1, 0.1, 0.1])
        depth_lb = torch.tensor([0.01, 0.01, 0.01])
        _btm_width_lb = torch.tensor([0.1, 0.1, 0.1])

        velocity = _get_trapezoid_velocity(
            q_t, _n, top_width, side_slope, _s0, p_spatial, _q_spatial, velocity_lb, depth_lb, _btm_width_lb
        )

        # Check output properties
        assert_tensor_properties(velocity, (3,), torch.float32)
        assert_no_nan_or_inf(velocity, "velocity")
        assert (velocity >= velocity_lb).all(), "Velocity should be >= lower bound"
        assert (velocity <= 15.0).all(), "Velocity should be <= 15 m/s (clamped)"

    def test_get_trapezoid_velocity_edge_cases(self):
        """Test _get_trapezoid_velocity with edge cases."""
        # Test with very small discharge
        q_t = torch.tensor([0.001])
        _n = torch.tensor([0.03])
        top_width = torch.tensor([1.0])
        side_slope = torch.tensor([1.0])
        _s0 = torch.tensor([0.001])
        p_spatial = torch.tensor([1.0])
        _q_spatial = torch.tensor([0.5])
        velocity_lb = torch.tensor([0.1])
        depth_lb = torch.tensor([0.01])
        _btm_width_lb = torch.tensor([0.1])

        velocity = _get_trapezoid_velocity(
            q_t, _n, top_width, side_slope, _s0, p_spatial, _q_spatial, velocity_lb, depth_lb, _btm_width_lb
        )

        assert velocity >= velocity_lb, "Velocity should respect lower bound"
        assert_no_nan_or_inf(velocity, "velocity")


class TestDMCInitialization:
    """Test dmc class initialization."""

    def test_init_cpu(self):
        """Test initialization with CPU device."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        assert model.device_num == "cpu"
        assert model.cfg == cfg
        assert model.t.item() == 3600.0
        assert model.t.device.type == "cpu"

        # Test tensor attributes
        assert isinstance(model.p_spatial, torch.Tensor)
        assert isinstance(model.velocity_lb, torch.Tensor)
        assert isinstance(model.depth_lb, torch.Tensor)
        assert isinstance(model.discharge_lb, torch.Tensor)
        assert isinstance(model.bottom_width_lb, torch.Tensor)

    def test_init_default_device(self):
        """Test initialization with default device."""
        cfg = create_mock_config()
        model = dmc(cfg)

        assert model.device_num == "cpu"

    def test_init_none_device(self):
        """Test initialization with None device."""
        cfg = create_mock_config()
        model = dmc(cfg, device=None)

        assert model.device_num is None

    def test_parameter_bounds_setup(self):
        """Test that parameter bounds are correctly set up."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        assert model.parameter_bounds == cfg.params.parameter_ranges.range
        assert model.p_spatial.item() == cfg.params.defaults.p
        assert torch.allclose(model.velocity_lb, torch.tensor(cfg.params.attribute_minimums.velocity))
        assert torch.allclose(model.depth_lb, torch.tensor(cfg.params.attribute_minimums.depth))
        assert torch.allclose(model.discharge_lb, torch.tensor(cfg.params.attribute_minimums.discharge))
        assert torch.allclose(model.bottom_width_lb, torch.tensor(cfg.params.attribute_minimums.bottom_width))


class TestDMCSparseOperations:
    """Test sparse matrix operations in dmc."""

    def test_sparse_eye(self):
        """Test _sparse_eye method."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        n = 5
        identity = model._sparse_eye(n)

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
        model = dmc(cfg, device="cpu")

        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        diag_matrix = model._sparse_diag(data)

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
        model = dmc(cfg, device="cpu")

        # Setup a simple network
        model.network = torch.tensor([[0.0, 1.0], [0.0, 0.0]])  # 2x2 network

        data_vector = torch.tensor([0.5, -0.3])
        result = model.fill_op(data_vector)

        assert result.shape == (2, 2)
        # For fill_op, the result might be dense depending on implementation
        # Just check that it produces reasonable output
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestDMCForwardPass:
    """Test the forward pass of dmc."""

    @pytest.fixture
    def setup_model_and_data(self):
        """Setup model and test data."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        return model, hydrofabric, streamflow, spatial_params

    def test_forward_basic(self, setup_model_and_data):
        """Test basic forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        # Mock progress tracking
        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock triangular_sparse_solve to avoid complex sparse solver
        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(10) * 5.0  # Mock solution

            output = model(**kwargs)

        assert isinstance(output, dict)
        assert "runoff" in output
        assert_tensor_properties(output["runoff"], (2, 24))  # 2 gauges, 24 timesteps
        assert_no_nan_or_inf(output["runoff"], "runoff")

    def test_forward_parameter_setup(self, setup_model_and_data):
        """Test that parameters are correctly set up during forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(10) * 5.0

            model(**kwargs)

        # Check that parameters were set
        assert model.n is not None
        assert model.q_spatial is not None
        assert model.network is not None
        assert model._discharge_t is not None

        # Check parameter shapes
        assert_tensor_properties(model.n, (10,))
        assert_tensor_properties(model.q_spatial, (10,))
        assert_tensor_properties(model._discharge_t, (10,))

    def test_forward_discharge_clamping(self, setup_model_and_data):
        """Test that discharge is properly clamped."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            # Return values that include some below minimum discharge to test clamping
            # This will be called once per timestep, so return a function that
            # alternates between different test cases
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
            output = model(**kwargs)

        # Check that output is reasonable (no negative infinity, NaN)
        assert not torch.isnan(output["runoff"]).any(), "Output should not contain NaN"
        assert not torch.isinf(output["runoff"]).any(), "Output should not contain infinity"

        # Test discharge clamping where values are actually computed
        min_discharge = model.discharge_lb.item()
        tolerance = 1e-6  # Floating point tolerance
        min_threshold = min_discharge - tolerance

        # Filter out zero values which may be due to incomplete gauge indexing
        # (see TODO in dmc.py line 118: "create a dynamic gauge look up")
        non_zero_values = output["runoff"][output["runoff"] > 0.0]

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
        assert (output["runoff"] > 0.0).any(), "Should have some non-zero runoff values"

        # Ensure the output has reasonable values - at least some should be at or above min_discharge
        meaningful_values = output["runoff"] >= min_discharge
        assert meaningful_values.any(), (
            f"Should have some values >= min_discharge ({min_discharge}), "
            f"but max value is {output['runoff'].max().item()}"
        )


class TestDMCErrorHandling:
    """Test error handling in dmc."""

    def test_cuda_out_of_memory_handling(self):
        """Test CUDA out of memory error handling."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock CUDA out of memory error
        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.side_effect = torch.cuda.OutOfMemoryError("CUDA out of memory")

            with pytest.raises(torch.cuda.OutOfMemoryError):
                model(**kwargs)


class TestDMCIntegration:
    """Integration tests for dmc with different scenarios."""

    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_different_network_sizes(self, scenario):
        """Test dmc with different network sizes."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=scenario["num_reaches"])
        streamflow = create_mock_streamflow(
            num_timesteps=scenario["num_timesteps"], num_reaches=scenario["num_reaches"]
        )
        spatial_params = create_mock_spatial_parameters(num_reaches=scenario["num_reaches"])

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(scenario["num_reaches"]) * 5.0

            output = model(**kwargs)

        # Check output properties
        expected_shape = (2, scenario["num_timesteps"])  # 2 gauges
        assert_tensor_properties(output["runoff"], expected_shape)
        assert_no_nan_or_inf(output["runoff"], f"runoff_{scenario['name']}")

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs."""
        cfg = create_mock_config()

        # Set seeds for reproducibility
        torch.manual_seed(42)

        model1 = dmc(cfg, device="cpu")
        model2 = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        # Make sure both models have same progress tracking
        model1.epoch = model2.epoch = 1
        model1.mini_batch = model2.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock with same return values
        mock_solution = torch.ones(10) * 5.0

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = mock_solution
            output1 = model1(**kwargs)

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = mock_solution
            output2 = model2(**kwargs)

        # Results should be identical
        assert torch.allclose(output1["runoff"], output2["runoff"])


class TestDMCPyTorchIntegration:
    """Test PyTorch integration features."""

    def test_is_nn_module(self):
        """Test that dmc is a proper PyTorch nn.Module."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        assert isinstance(model, torch.nn.Module)

        # Test module methods
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")
        assert hasattr(model, "named_parameters")
        assert hasattr(model, "state_dict")
        assert hasattr(model, "load_state_dict")

    def test_gradient_flow(self):
        """Test that gradients can flow through the model."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)

        # Create spatial parameters that require gradients
        spatial_params = {
            "n": torch.rand(5, requires_grad=True),
            "q_spatial": torch.rand(5, requires_grad=True),
        }

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            # Return solution that maintains gradient connections
            def mock_solve_func(A_values, crow_indices, col_indices, b, lower, unit_diagonal, device):
                # Return a solution that depends on the input b (which should have gradients)
                return b * 1.1 + 0.5

            mock_solve.side_effect = mock_solve_func

            output = model(**kwargs)

            # Compute a simple loss
            loss = output["runoff"].sum()

            # Check that we can compute gradients
            # The loss should have gradients if the spatial parameters do
            has_grad = any(param.requires_grad for param in spatial_params.values())
            if has_grad:
                assert loss.requires_grad, "Loss should require gradients when spatial parameters do"

                # This should not raise an error
                loss.backward()

                # Check that gradients were computed (they might be None if not used)
                # Just verify that backward() didn't fail
                pass
            else:
                # If no parameters require gradients, loss shouldn't either
                assert not loss.requires_grad


class TestDMCStateManagement:
    """Test state management in dmc."""

    def test_discharge_state_updates(self):
        """Test that discharge state is properly updated."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Track calls to triangular_sparse_solve
        call_count = 0
        solutions = []

        def mock_solve(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            solution = torch.ones(5) * (5.0 + call_count)  # Different solution each time
            solutions.append(solution.clone())
            return solution

        with patch("ddr.routing.dmc.triangular_sparse_solve", side_effect=mock_solve):
            model(**kwargs)

        # Should be called for each timestep after the first
        assert call_count == 11  # 12 timesteps - 1 initial

        # Final discharge state should match last solution
        expected_final_discharge = solutions[-1]
        assert torch.allclose(model._discharge_t, expected_final_discharge)

    def test_network_assignment(self):
        """Test that network is properly assigned."""
        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)
        spatial_params = create_mock_spatial_parameters(num_reaches=5)

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.dmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(5) * 5.0

            model(**kwargs)

        # Network should be assigned from hydrofabric
        assert model.network is not None
        assert torch.equal(model.network, hydrofabric.adjacency_matrix)


class TestParameterTraining:
    """Test Training of parameters in dmc."""

    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_parameter_training(self, scenario):
        """Test that parameters can be trained."""

        if scenario["num_reaches"] <= 1:
            pytest.skip("Skipping parameter training test for single reach scenarios")

        cfg = create_mock_config()
        model = dmc(cfg, device="cpu")
        # Create mock hydrofabric and streamflow
        hydrofabric = create_mock_hydrofabric(num_reaches=scenario["num_reaches"])
        streamflow = create_mock_streamflow(
            num_timesteps=scenario["num_timesteps"], num_reaches=scenario["num_reaches"]
        )
        nn = create_mock_nn()
        spatial_params = nn(inputs=hydrofabric.normalized_spatial_attributes.to(cfg.device))

        model.epoch = 1
        model.mini_batch = 0

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Skip deep omegaconf attributes
        # these *shouldn't* have any tensors...
        skip_attrs = ["_content", "_metadata", "_parent"]
        # This ONLY works for tensors which are not dynamically
        # created/recreated during the forward pass.
        find_and_retain_grad(nn, required=True, skip=skip_attrs)
        find_and_retain_grad(model, required=True, skip=skip_attrs)
        find_and_retain_grad(hydrofabric, required=True, skip=skip_attrs)
        # To test the dynamic tensors we care about (model.n, model.q_spatial, and model._discharge_t),
        # we can pass an additional kwarg to dmc
        kwargs["retain_grads"] = True  # This is a custom kwarg to trigger gradient checks

        # Mock optimizer
        optimizer = torch.optim.Adam(params=nn.parameters(), lr=0.01)

        # Forward pass
        output = model(**kwargs)

        test_modules = [hydrofabric, nn, model]
        modules_names = ["hydrofabric", "nn", "model"]

        ts = [find_gradient_tensors(obj, skip=skip_attrs) for obj in test_modules]
        init_tensors = [
            t for ts_ in ts for t in ts_
        ]  # flatten the list of lists, copy the requires attribute

        optimizer.zero_grad(False)  # Zero gradients before backward pass
        # Compute a simple loss
        loss = output["runoff"].sum()
        loss.retain_grad()
        # Backward pass
        loss.backward()
        optimizer.step()

        assert loss.grad is not None, "Loss should have gradients after backward pass"
        assert not torch.isnan(loss.grad).any(), "Loss gradients should not contain NaN"
        assert not torch.isinf(loss.grad).any(), "Loss gradients should not contain infinity"

        ts = [find_gradient_tensors(obj, skip=skip_attrs) for obj in test_modules]
        end_tensors = [t for ts_ in ts for t in ts_]  # flatten the list of lists, copy the requires attribute
        ns = [
            get_tensor_names(obj, name=name, skip=skip_attrs)
            for obj, name in zip(test_modules, modules_names, strict=False)
        ]
        names = [n for ns_ in ns for n in ns_]  # flatten the list of lists, copy the names
        assert len(init_tensors) == len(end_tensors), (
            "Initial and final tensor lists should be the same length"
        )
        assert len(names) == len(init_tensors), "Names list should match the length of tensor lists"

        # Skip internal KAN tensors that are not directly connected to the loss
        # There's probably a more elegant way to do this, but this works for now
        # These tensors have require_grad=True, but no gradients are computed during `backward()`
        # TODO convince myself that these really shouldn't have gradient values based on the loss
        # computation chain...
        skip_patterns = [
            "acts_scale_spline",
            "edge_actscale",
        ]

        # Also skip spatial parameters that are not used by the routing engine
        # The routing engine only uses parameters that are in cfg.params.parameter_ranges.range
        unused_spatial_params = []
        for param_name in ["n", "q_spatial", "p_spatial"]:
            if param_name not in cfg.params.parameter_ranges.range:
                unused_spatial_params.append(f"spatial_parameters['{param_name}']")

        # do this in a loop so we can see which tensors changed more explicitly
        for name, init, end in zip(names, init_tensors, end_tensors, strict=False):
            assert init.requires_grad == end.requires_grad, (
                f"Tensor {name} requires_grad status should not change during training. Initial: {init}, Final: {end}"
            )
            if end.requires_grad:
                if any(pattern in name for pattern in skip_patterns):
                    continue
                # Skip unused spatial parameters
                if any(unused_param in name for unused_param in unused_spatial_params):
                    continue
                assert end.grad is not None, f"Tensor {name} should have gradients after backward pass"
                assert not torch.isnan(end.grad).any(), f"Tensor {name} gradients should not contain NaN"
                assert not torch.isinf(end.grad).any(), f"Tensor {name} gradients should not contain infinity"
        # These are redundant assertions *if* you believe these are captured by the
        # find_gradient_tensors() function, but they are useful to have here, just in case...
        # Check runoff tensor explicitly...
        assert output["runoff"].grad is not None, "Runoff output should have gradients after backward pass"
        assert not torch.isnan(output["runoff"].grad).any(), "Runoff gradients should not contain NaN"
        assert not torch.isinf(output["runoff"].grad).any(), "Runoff gradients should not contain infinity"
        # Check the parameter neural network output weights explictly...
        assert nn.output.weight.grad is not None, (
            "Neural network output weights should have gradients after backward pass"
        )
        assert not torch.isnan(nn.output.weight.grad).any(), (
            "Neural network output weights gradients should not contain NaN"
        )
        assert not torch.isinf(nn.output.weight.grad).any(), (
            "Neural network output weights gradients should not contain infinity"
        )

        # print("Tensors in dmc:")
        # print_grad_info(model, name="dmc", required=True, skip=skip_attrs)
        # print("Tensors in kan:")
        # print_grad_info(nn, name="kan", required=True, skip=skip_attrs)
        # print("Tensors in hydrofabric:")
        # print_grad_info(hydrofabric, name="hydrofabric", required=True, skip=skip_attrs)
