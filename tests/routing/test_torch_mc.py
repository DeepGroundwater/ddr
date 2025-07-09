"""Comprehensive tests for the refactored torch_mc.py module."""

from unittest.mock import Mock, patch

import pytest
import torch

from ddr.routing.mmc import MuskingunCunge
from ddr.routing.torch_mc import TorchMC, dmc
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


class TestTorchMCInitialization:
    """Test TorchMC class initialization."""

    def test_init_cpu(self):
        """Test initialization with CPU device."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        assert model.device_num == "cpu"
        assert model.cfg == cfg
        assert isinstance(model.routing_engine, MuskingunCunge)

        # Test that PyTorch module properties are set
        assert isinstance(model, torch.nn.Module)
        assert model.t.item() == 3600.0
        assert model.t.device.type == "cpu"

        # Test tensor attributes (copied from routing engine)
        assert isinstance(model.p_spatial, torch.Tensor)
        assert isinstance(model.velocity_lb, torch.Tensor)
        assert isinstance(model.depth_lb, torch.Tensor)
        assert isinstance(model.discharge_lb, torch.Tensor)
        assert isinstance(model.bottom_width_lb, torch.Tensor)

        # Test compatibility attributes
        assert model._discharge_t is None
        assert model.network is None
        assert model.n is None
        assert model.q_spatial is None
        assert model.epoch == 0
        assert model.mini_batch == 0

    def test_init_default_device(self):
        """Test initialization with default device."""
        cfg = create_mock_config()
        model = TorchMC(cfg)

        assert model.device_num == "cpu"

    def test_init_none_device(self):
        """Test initialization with None device."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device=None)

        assert model.device_num == "cpu"

    def test_routing_engine_setup(self):
        """Test that routing engine is properly initialized."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        assert isinstance(model.routing_engine, MuskingunCunge)
        assert model.routing_engine.device == "cpu"
        assert model.routing_engine.cfg == cfg


class TestTorchMCDeviceManagement:
    """Test device management functionality."""

    def test_to_method_cpu_to_cpu(self):
        """Test .to() method from CPU to CPU."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        model_cpu = model.to("cpu")

        assert model_cpu is model  # Should return self
        assert model_cpu.device_num == "cpu"
        assert model_cpu.routing_engine.device == "cpu"
        assert model_cpu.t.device.type == "cpu"

    def test_to_method_device_object(self):
        """Test .to() method with torch.device object."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        device = torch.device("cpu")
        model_moved = model.to(device)

        assert model_moved is model
        assert model_moved.device_num == "cpu"

    def test_cpu_method(self):
        """Test .cpu() method."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        model_cpu = model.cpu()

        assert model_cpu is model
        assert model_cpu.device_num == "cpu"
        assert model_cpu.routing_engine.device == "cpu"

    def test_cuda_method_default(self):
        """Test .cuda() method with default device."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        # Mock the to() method to avoid actual CUDA operations
        with patch.object(model, "to") as mock_to:
            mock_to.return_value = model
            model_cuda = model.cuda()

            mock_to.assert_called_once_with("cuda")
            assert model_cuda is model

    def test_cuda_method_with_device_int(self):
        """Test .cuda() method with device index."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        with patch.object(model, "to") as mock_to:
            mock_to.return_value = model
            model.cuda(0)

            mock_to.assert_called_once_with("cuda:0")

    def test_cuda_method_with_device_object(self):
        """Test .cuda() method with torch.device object."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        device = torch.device("cpu")  # Using CPU since we don't have CUDA
        with patch.object(model, "to") as mock_to:
            mock_to.return_value = model
            model.cuda(device)

            mock_to.assert_called_once_with("cpu")


class TestTorchMCProgressTracking:
    """Test progress tracking functionality."""

    def test_set_progress_info(self):
        """Test setting progress information."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        model.set_progress_info(5, 10)

        assert model.epoch == 5
        assert model.mini_batch == 10
        assert model.routing_engine.epoch == 5
        assert model.routing_engine.mini_batch == 10


class TestTorchMCForwardPass:
    """Test the forward pass of TorchMC."""

    @pytest.fixture
    def setup_model_and_data(self):
        """Setup model and test data."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=10)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=10)
        spatial_params = create_mock_spatial_parameters(num_reaches=10)

        return model, hydrofabric, streamflow, spatial_params

    def test_forward_basic(self, setup_model_and_data):
        """Test basic forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        model.set_progress_info(1, 0)

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock the routing engine forward method
        with patch.object(model.routing_engine, "forward") as mock_forward:
            mock_forward.return_value = torch.ones(2, 24) * 5.0  # 2 gauges, 24 timesteps

            output = model(**kwargs)

        assert isinstance(output, dict)
        assert "runoff" in output
        assert_tensor_properties(output["runoff"], (2, 24))
        assert_no_nan_or_inf(output["runoff"], "runoff")

    def test_forward_routing_engine_setup(self, setup_model_and_data):
        """Test that routing engine is properly set up during forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with (
            patch.object(model.routing_engine, "setup_inputs") as mock_setup,
            patch.object(model.routing_engine, "forward") as mock_forward,
        ):
            mock_forward.return_value = torch.ones(2, 24) * 5.0

            model(**kwargs)

            # Check that setup_inputs was called correctly
            mock_setup.assert_called_once_with(
                hydrofabric=hydrofabric, streamflow=streamflow, spatial_parameters=spatial_params
            )

    def test_forward_compatibility_attributes_update(self, setup_model_and_data):
        """Test that compatibility attributes are updated during forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Setup mock routing engine with some state
        with (
            patch.object(model.routing_engine, "setup_inputs"),
            patch.object(model.routing_engine, "forward") as mock_forward,
        ):
            # Mock routing engine state
            model.routing_engine.network = hydrofabric.adjacency_matrix
            model.routing_engine.n = torch.ones(10) * 0.03
            model.routing_engine.q_spatial = torch.ones(10) * 0.5
            model.routing_engine._discharge_t = torch.ones(10) * 2.0

            mock_forward.return_value = torch.ones(2, 24) * 5.0

            model(**kwargs)

            # Check that compatibility attributes were updated
            assert model.network is not None
            assert model.n is not None
            assert model.q_spatial is not None
            assert model._discharge_t is not None

            assert torch.equal(model.network, model.routing_engine.network)
            assert torch.equal(model.n, model.routing_engine.n)
            assert torch.equal(model.q_spatial, model.routing_engine.q_spatial)
            assert torch.equal(model._discharge_t, model.routing_engine._discharge_t)

    def test_forward_device_handling(self, setup_model_and_data):
        """Test device handling in forward pass."""
        model, hydrofabric, streamflow, spatial_params = setup_model_and_data

        # Test with streamflow on different device (should be moved to model device)
        streamflow_wrong_device = streamflow.clone()  # Same device for testing

        kwargs = {
            "hydrofabric": hydrofabric,
            "streamflow": streamflow_wrong_device,
            "spatial_parameters": spatial_params,
        }

        with (
            patch.object(model.routing_engine, "setup_inputs") as mock_setup,
            patch.object(model.routing_engine, "forward") as mock_forward,
        ):
            mock_forward.return_value = torch.ones(2, 24) * 5.0

            model(**kwargs)

            # Check that streamflow was moved to correct device
            called_args = mock_setup.call_args
            assert called_args[1]["streamflow"].device.type == model.device_num


class TestTorchMCCompatibilityMethods:
    """Test compatibility methods for backward compatibility."""

    def test_fill_op_delegation(self):
        """Test that fill_op delegates to routing engine."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        data_vector = torch.tensor([1.0, 2.0, 3.0])

        with patch.object(model.routing_engine, "fill_op") as mock_fill_op:
            mock_fill_op.return_value = torch.eye(3)

            result = model.fill_op(data_vector)

            mock_fill_op.assert_called_once_with(data_vector)
            assert torch.equal(result, torch.eye(3))

    def test_sparse_eye_delegation(self):
        """Test that _sparse_eye delegates to routing engine."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        with patch.object(model.routing_engine, "_sparse_eye") as mock_sparse_eye:
            mock_sparse_eye.return_value = torch.eye(5).to_sparse()

            model._sparse_eye(5)

            mock_sparse_eye.assert_called_once_with(5)

    def test_sparse_diag_delegation(self):
        """Test that _sparse_diag delegates to routing engine."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        data = torch.tensor([1.0, 2.0, 3.0])

        with patch.object(model.routing_engine, "_sparse_diag") as mock_sparse_diag:
            mock_sparse_diag.return_value = torch.diag(data).to_sparse()

            model._sparse_diag(data)

            mock_sparse_diag.assert_called_once_with(data)

    def test_route_timestep_delegation(self):
        """Test that route_timestep delegates to routing engine."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        q_prime_clamp = torch.ones(5) * 2.0
        mapper = Mock()

        with patch.object(model.routing_engine, "route_timestep") as mock_route:
            mock_route.return_value = torch.ones(5) * 5.0

            result = model.route_timestep(q_prime_clamp, mapper)

            mock_route.assert_called_once_with(q_prime_clamp=q_prime_clamp, mapper=mapper)
            assert torch.equal(result, torch.ones(5) * 5.0)


class TestTorchMCStateDict:
    """Test state dictionary functionality."""

    def test_state_dict(self):
        """Test state_dict method."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        model.set_progress_info(3, 7)

        state = model.state_dict()

        # Check that custom attributes are included
        assert "cfg" in state
        assert "device_num" in state
        assert "epoch" in state
        assert "mini_batch" in state

        assert state["cfg"] == cfg
        assert state["device_num"] == "cpu"
        assert state["epoch"] == 3
        assert state["mini_batch"] == 7

    def test_load_state_dict(self):
        """Test load_state_dict method."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        # Create mock state dict
        state_dict = {"cfg": cfg, "device_num": "cpu", "epoch": 5, "mini_batch": 12}

        model.load_state_dict(state_dict, strict=False)

        assert model.cfg == cfg
        assert model.device_num == "cpu"
        assert model.epoch == 5
        assert model.mini_batch == 12
        assert model.routing_engine.epoch == 5
        assert model.routing_engine.mini_batch == 12

    def test_load_state_dict_missing_keys(self):
        """Test load_state_dict with missing keys uses defaults."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        # Set some initial values
        model.set_progress_info(2, 4)

        # Create incomplete state dict
        state_dict = {"device_num": "cpu"}

        model.load_state_dict(state_dict, strict=False)

        # Should use original cfg and default epoch/mini_batch
        assert model.cfg == cfg
        assert model.device_num == "cpu"
        assert model.epoch == 0  # Default
        assert model.mini_batch == 0  # Default


class TestTorchMCPyTorchIntegration:
    """Test PyTorch integration features."""

    def test_is_nn_module(self):
        """Test that TorchMC is a proper PyTorch nn.Module."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        assert isinstance(model, torch.nn.Module)

        # Test module methods exist
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")
        assert hasattr(model, "named_parameters")
        assert hasattr(model, "state_dict")
        assert hasattr(model, "load_state_dict")
        assert hasattr(model, "to")
        assert hasattr(model, "cpu")
        assert hasattr(model, "cuda")

    def test_no_learnable_parameters(self):
        """Test that TorchMC has no learnable parameters by default."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        params = list(model.parameters())
        assert len(params) == 0, "TorchMC should have no learnable parameters"

    def test_gradient_flow_compatibility(self):
        """Test gradient flow compatibility."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=5)
        streamflow = create_mock_streamflow(num_timesteps=12, num_reaches=5)

        # Create spatial parameters that require gradients
        spatial_params = {
            "n": torch.rand(5, requires_grad=True),
            "q_spatial": torch.rand(5, requires_grad=True),
        }

        model.set_progress_info(1, 0)

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            # Return solution that maintains gradient connections
            def mock_solve_func(A_values, crow_indices, col_indices, b, lower, unit_diagonal, device):
                # Return a solution that depends on the input b (which should have gradients)
                return b * 1.1 + 0.5

            mock_solve.side_effect = mock_solve_func

            output = model(**kwargs)

            # Compute a simple loss
            loss = output["runoff"].sum()

            # Check that gradients can flow
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


class TestTorchMCIntegration:
    """Integration tests for TorchMC."""

    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_different_network_sizes(self, scenario):
        """Test TorchMC with different network sizes."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        hydrofabric = create_mock_hydrofabric(num_reaches=scenario["num_reaches"])
        streamflow = create_mock_streamflow(
            num_timesteps=scenario["num_timesteps"], num_reaches=scenario["num_reaches"]
        )
        spatial_params = create_mock_spatial_parameters(num_reaches=scenario["num_reaches"])

        model.set_progress_info(1, 0)

        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        with patch.object(model.routing_engine, "forward") as mock_forward:
            expected_shape = (2, scenario["num_timesteps"])
            mock_forward.return_value = torch.ones(expected_shape) * 5.0

            output = model(**kwargs)

        # Check output properties
        assert_tensor_properties(output["runoff"], expected_shape)
        assert_no_nan_or_inf(output["runoff"], f"runoff_{scenario['name']}")

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        # Test device management
        model_cpu = model.cpu()
        assert model_cpu.device_num == "cpu"

        # Setup data
        hydrofabric = create_mock_hydrofabric(num_reaches=8)
        streamflow = create_mock_streamflow(num_timesteps=36, num_reaches=8)
        spatial_params = create_mock_spatial_parameters(num_reaches=8)

        # Set progress info
        model.set_progress_info(2, 5)

        # Prepare kwargs
        kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

        # Mock the routing engine forward pass
        with (
            patch.object(model.routing_engine, "setup_inputs") as mock_setup,
            patch.object(model.routing_engine, "forward") as mock_forward,
        ):
            mock_forward.return_value = torch.ones(2, 36) * 7.5

            # Run forward pass
            output = model(**kwargs)

            # Verify setup was called
            mock_setup.assert_called_once()

            # Verify output
            assert isinstance(output, dict)
            assert "runoff" in output
            assert_tensor_properties(output["runoff"], (2, 36))

        # Test state dict functionality
        state = model.state_dict()
        assert "epoch" in state
        assert state["epoch"] == 2

        # Create new model and load state
        new_model = TorchMC(cfg, device="cpu")
        new_model.load_state_dict(state, strict=False)
        assert new_model.epoch == 2
        assert new_model.mini_batch == 5


class TestTorchMCBackwardCompatibility:
    """Test backward compatibility features."""

    def test_dmc_alias(self):
        """Test that dmc is an alias for TorchMC."""
        assert dmc is TorchMC

    def test_interface_compatibility(self):
        """Test that TorchMC has same interface as original dmc."""
        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")

        # Test that all expected attributes exist
        expected_attributes = [
            "t",
            "parameter_bounds",
            "p_spatial",
            "velocity_lb",
            "depth_lb",
            "discharge_lb",
            "bottom_width_lb",
            "_discharge_t",
            "network",
            "n",
            "q_spatial",
            "epoch",
            "mini_batch",
        ]

        for attr in expected_attributes:
            assert hasattr(model, attr), f"Missing attribute: {attr}"

        # Test that all expected methods exist
        expected_methods = [
            "forward",
            "fill_op",
            "_sparse_eye",
            "_sparse_diag",
            "route_timestep",
            "set_progress_info",
            "to",
            "cpu",
            "cuda",
            "state_dict",
            "load_state_dict",
        ]

        for method in expected_methods:
            assert hasattr(model, method), f"Missing method: {method}"
            assert callable(getattr(model, method)), f"Not callable: {method}"

    def test_drop_in_replacement(self):
        """Test that TorchMC can be used as drop-in replacement."""
        cfg = create_mock_config()

        # This is how the original dmc would be used
        routing_model = dmc(cfg=cfg, device="cpu")

        # Test basic initialization
        assert isinstance(routing_model, TorchMC)
        assert routing_model.device_num == "cpu"

        # Test progress tracking (as used in training scripts)
        routing_model.epoch = 1
        routing_model.mini_batch = 0

        assert routing_model.epoch == 1
        assert routing_model.mini_batch == 0

        # Test forward pass interface
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

            # Should return dict with 'runoff' key (same as original)
            assert isinstance(dmc_output, dict)
            assert "runoff" in dmc_output


class TestParameterTraining:
    """Test Training of parameters in dmc."""

    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_parameter_training(self, scenario):
        """Test that parameters can be trained."""

        if scenario["num_reaches"] <= 1:
            pytest.skip("Skipping parameter training test for single reach scenarios")

        cfg = create_mock_config()
        model = TorchMC(cfg, device="cpu")
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
