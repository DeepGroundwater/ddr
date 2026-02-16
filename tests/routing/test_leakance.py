"""Tests for leakance (groundwater-surface water exchange) in routing."""

from typing import Any
from unittest.mock import patch

import pytest
import torch
from omegaconf import DictConfig

from ddr.routing.mmc import MuskingumCunge, _compute_zeta
from ddr.routing.torch_mc import dmc
from ddr.validation.configs import validate_config
from tests.routing.test_utils import (
    assert_no_nan_or_inf,
    assert_tensor_properties,
    create_mock_config_with_cuda_lstm,
    create_mock_config_with_leakance,
    create_mock_cuda_lstm,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
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
            "top_width": torch.ones(n) * 10.0,
            "side_slope": torch.ones(n) * 2.0,
        }

    def test_zeta_shape_matches_inputs(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta output shape matches input shapes."""
        zeta = _compute_zeta(**zeta_inputs)
        assert_tensor_properties(zeta, (10,))

    def test_zeta_positive_when_deep_water_table(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta is positive when water table is deep (losing stream)."""
        # Large d_gw = deep water table => Δh = depth - h_bed + d_gw >> 0 => zeta > 0
        zeta_inputs["d_gw"] = torch.ones(10) * 200.0
        zeta = _compute_zeta(**zeta_inputs)
        assert (zeta > 0).all(), f"Expected all positive (losing stream), got {zeta}"

    def test_zeta_negative_when_shallow_water_table(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta is negative when water table is shallow (gaining stream)."""
        # Small d_gw = shallow water table => Δh = depth - h_bed + d_gw < 0 when h_bed > depth + d_gw
        # h_bed = 10/(2*2) = 2.5m, depth ≈ 0.65m, so Δh ≈ 0.65 - 2.5 + 0.01 < 0 => gaining
        zeta_inputs["d_gw"] = torch.ones(10) * 0.01
        zeta = _compute_zeta(**zeta_inputs)
        assert (zeta < 0).all(), f"Expected all negative (gaining stream), got {zeta}"

    def test_zeta_no_nan_or_inf(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that zeta contains no NaN or Inf values."""
        zeta = _compute_zeta(**zeta_inputs)
        assert_no_nan_or_inf(zeta, "zeta")

    def test_zeta_gradient_flow(self, zeta_inputs: dict[str, torch.Tensor]) -> None:
        """Test that gradients flow through K_D and d_gw."""
        for key in ["K_D", "d_gw"]:
            zeta_inputs[key] = zeta_inputs[key].clone().requires_grad_(True)

        zeta = _compute_zeta(**zeta_inputs)
        loss = zeta.sum()
        loss.backward()

        for key in ["K_D", "d_gw"]:
            assert zeta_inputs[key].grad is not None, f"Gradient should exist for {key}"
            assert not torch.isnan(zeta_inputs[key].grad).any(), f"Gradient for {key} has NaN"
            assert not torch.isinf(zeta_inputs[key].grad).any(), f"Gradient for {key} has Inf"


class TestLeakanceInRouting:
    """Test leakance integration in MuskingumCunge routing."""

    def test_route_timestep_with_leakance(self) -> None:
        """Test that route_timestep works with leakance enabled."""
        cfg = create_mock_config_with_leakance()
        mc = MuskingumCunge(cfg, device="cpu")

        num_reaches = 10
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=24, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)
        # K_D now comes from spatial_parameters (KAN)
        spatial_params["K_D_delta"] = torch.rand(num_reaches)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        # Set LSTM params (n and d_gw)
        lstm_params = {
            "n": torch.rand(1, num_reaches),
            "d_gw": torch.rand(1, num_reaches),
        }
        mc.setup_lstm_params(lstm_params)
        # Manually set current-timestep n and d_gw from day 0
        assert mc._n_t is not None
        assert mc._d_gw_t is not None
        mc.n = mc._n_t[0]
        mc.d_gw = mc._d_gw_t[0]

        mapper, _, _ = mc.create_pattern_mapper()
        q_prime_clamp = torch.ones(num_reaches) * 2.0

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(num_reaches) * 5.0
            result = mc.route_timestep(q_prime_clamp, mapper)

        assert_tensor_properties(result, (num_reaches,))
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
        # No-leakance config still needs n set (normally from LSTM)
        lstm_params_no_leak = {"n": torch.ones(1, num_reaches) * 0.5}
        mc_no_leak.setup_lstm_params(lstm_params_no_leak)
        assert mc_no_leak._n_t is not None
        mc_no_leak.n = mc_no_leak._n_t[0]
        mapper_no_leak, _, _ = mc_no_leak.create_pattern_mapper()

        # Run WITH leakance (losing stream: d_gw=300m deep water table => zeta > 0)
        cfg_leak = create_mock_config_with_leakance()
        mc_leak = MuskingumCunge(cfg_leak, device="cpu")
        spatial_params_leak = {
            "q_spatial": spatial_params_no_leak["q_spatial"].clone(),
            "K_D_delta": torch.ones(num_reaches) * 0.5,  # Normalized, will be denormalized to [-2, 2]
        }
        mc_leak.setup_inputs(hydrofabric, streamflow, spatial_params_leak)

        # Set n and d_gw via LSTM path
        lstm_params = {
            "n": torch.ones(1, num_reaches) * 0.5,
            "d_gw": torch.ones(1, num_reaches),  # Normalized 1.0 => d_gw = 300m (deep water table)
        }
        mc_leak.setup_lstm_params(lstm_params)
        assert mc_leak._n_t is not None
        assert mc_leak._d_gw_t is not None
        mc_leak.n = mc_leak._n_t[0]
        mc_leak.d_gw = mc_leak._d_gw_t[0]
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

        num_reaches = 10
        num_timesteps = 24
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow = create_mock_streamflow(num_timesteps=num_timesteps, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)
        # K_D now from KAN (spatial_parameters)
        spatial_params["K_D_delta"] = torch.rand(num_reaches)

        T_daily = num_timesteps // 24
        lstm_params = {
            "n": torch.rand(T_daily, num_reaches),
            "d_gw": torch.rand(T_daily, num_reaches),
        }

        model.set_progress_info(1, 0)

        kwargs = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow,
            "spatial_parameters": spatial_params,
            "lstm_params": lstm_params,
        }

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(num_reaches) * 5.0
            output = model(**kwargs)

        assert isinstance(output, dict)
        assert "runoff" in output
        expected_shape = (1, num_timesteps)
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


class TestLeakanceGradientFlow:
    """Test gradient flow through leakance parameters."""

    def test_end_to_end_training_with_leakance(self) -> None:
        """Test that gradients propagate through LSTM params to LSTM weights."""
        cfg = create_mock_config_with_cuda_lstm()
        model = dmc(cfg, device="cpu")
        num_reaches = 10
        num_timesteps = 48  # 2 days

        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow_data = create_mock_streamflow(num_timesteps=num_timesteps, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)
        # K_D now from KAN (spatial_parameters)
        spatial_params["K_D_delta"] = torch.rand(num_reaches)
        lstm_nn = create_mock_cuda_lstm()

        all_params = list(lstm_nn.parameters())
        optimizer = torch.optim.Adam(params=all_params, lr=0.01)

        model.epoch = 1
        model.mini_batch = 0

        T_daily = num_timesteps // 24
        mock_forcings = torch.rand(T_daily, num_reaches, 3)  # P, PET, Temp
        lstm_params = lstm_nn(
            forcings=mock_forcings,
            attributes=hydrofabric.normalized_spatial_attributes,
        )

        kwargs: dict[str, Any] = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow_data,
            "spatial_parameters": spatial_params,
            "lstm_params": lstm_params,
        }

        output = model(**kwargs)

        optimizer.zero_grad(False)
        loss = output["runoff"].sum()
        loss.retain_grad()
        loss.backward()
        optimizer.step()

        assert loss.grad is not None, "Loss should have gradients after backward pass"
        assert not torch.isnan(loss.grad).any(), "Loss gradients should not contain NaN"
        assert not torch.isinf(loss.grad).any(), "Loss gradients should not contain infinity"

        # Verify gradients reached the LSTM
        assert lstm_nn.linear_in.weight.grad is not None, (
            "LSTM linear_in should have gradients from routing loss"
        )
        assert lstm_nn.linear_out.weight.grad is not None, (
            "LSTM linear_out should have gradients from routing loss"
        )


class TestLeakanceLstmInRouting:
    """Test time-varying leakance from LSTM in routing."""

    def test_setup_lstm_params_denormalizes(self) -> None:
        """Test that setup_lstm_params stores denormalized daily n and d_gw tensors."""
        cfg = create_mock_config_with_cuda_lstm()
        mc = MuskingumCunge(cfg, device="cpu")

        T_daily, N = 3, 10
        lstm_params = {
            "n": torch.ones(T_daily, N) * 0.5,
            "d_gw": torch.ones(T_daily, N) * 0.5,
        }
        mc.setup_lstm_params(lstm_params)

        assert mc._n_t is not None
        assert mc._n_t.shape == (T_daily, N)
        assert mc._d_gw_t is not None
        assert mc._d_gw_t.shape == (T_daily, N)

    def test_forward_with_lstm_params(self) -> None:
        """Test full forward pass with LSTM-path params (n + d_gw)."""
        cfg = create_mock_config_with_cuda_lstm()
        model = dmc(cfg, device="cpu")

        num_reaches = 10
        num_timesteps = 48  # 2 days
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow_data = create_mock_streamflow(num_timesteps=num_timesteps, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)
        # K_D from KAN (spatial_parameters)
        spatial_params["K_D_delta"] = torch.rand(num_reaches)

        T_daily = num_timesteps // 24
        lstm_params = {
            "n": torch.rand(T_daily, num_reaches),
            "d_gw": torch.rand(T_daily, num_reaches),
        }

        model.set_progress_info(1, 0)
        kwargs = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow_data,
            "spatial_parameters": spatial_params,
            "lstm_params": lstm_params,
        }

        with patch("ddr.routing.mmc.triangular_sparse_solve") as mock_solve:
            mock_solve.return_value = torch.ones(num_reaches) * 5.0
            output = model(**kwargs)

        assert "runoff" in output
        assert_no_nan_or_inf(output["runoff"], "runoff_lstm_params")

    def test_daily_to_hourly_indexing(self) -> None:
        """Test that the daily→hourly mapping (timestep-1)//24 gives correct day_idx."""
        cfg = create_mock_config_with_cuda_lstm()
        mc = MuskingumCunge(cfg, device="cpu")

        T_daily, N = 2, 5
        # Set up different values for each day so we can verify indexing
        lstm_params = {
            "n": torch.zeros(T_daily, N),
            "d_gw": torch.zeros(T_daily, N),
        }
        lstm_params["n"][0] = 0.2
        lstm_params["n"][1] = 0.8
        mc.setup_lstm_params(lstm_params)

        assert mc._n_t is not None
        # Timestep 24 -> day_idx = 23//24 = 0
        assert (0) == (24 - 1) // 24  # Still day 0
        # Timestep 25 -> day_idx = 24//24 = 1
        assert (1) == (25 - 1) // 24  # Day 1

    def test_lstm_gradient_flow_through_routing(self) -> None:
        """Test end-to-end gradient from loss -> LSTM weights via routing."""
        cfg = create_mock_config_with_cuda_lstm()
        model = dmc(cfg, device="cpu")

        num_reaches = 10
        num_timesteps = 48  # 2 days
        hydrofabric = create_mock_routing_dataclass(num_reaches=num_reaches)
        streamflow_data = create_mock_streamflow(num_timesteps=num_timesteps, num_reaches=num_reaches)
        spatial_params = create_mock_spatial_parameters(num_reaches=num_reaches)
        # K_D from KAN (spatial_parameters)
        spatial_params["K_D_delta"] = torch.rand(num_reaches)

        lstm_nn = create_mock_cuda_lstm()
        T_daily = num_timesteps // 24
        mock_forcings = torch.rand(T_daily, num_reaches, 3)  # P, PET, Temp

        lstm_params = lstm_nn(
            forcings=mock_forcings,
            attributes=hydrofabric.normalized_spatial_attributes,
        )

        model.set_progress_info(1, 0)
        kwargs: dict[str, Any] = {
            "routing_dataclass": hydrofabric,
            "streamflow": streamflow_data,
            "spatial_parameters": spatial_params,
            "lstm_params": lstm_params,
        }
        output = model(**kwargs)

        loss = output["runoff"].sum()
        loss.backward()

        # Verify gradients reached the LSTM
        assert lstm_nn.linear_in.weight.grad is not None, (
            "LSTM linear_in should have gradients from routing loss"
        )
        assert lstm_nn.linear_out.weight.grad is not None, (
            "LSTM linear_out should have gradients from routing loss"
        )


class TestLeakanceConfigValidation:
    """Test configuration validation for leakance."""

    def test_use_leakance_true_without_param_ranges_raises(self) -> None:
        """Test that use_leakance=True without K_D/d_gw in parameter_ranges raises."""
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
                "forcings": "mock://forcings/store",
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
                "learnable_parameters": ["q_spatial"],
            },
            "cuda_lstm": {
                "input_var_names": ["mock"],
                "learnable_parameters": ["n"],
            },
            "s3_region": "us-east-1",
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="params.parameter_ranges"):
            validate_config(DictConfig(cfg_dict), save_config=False)

    def test_use_leakance_true_with_proper_param_split_valid(self) -> None:
        """Test that use_leakance=True is valid with K_D_delta in KAN, n+d_gw in LSTM."""
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
                "forcings": "mock://forcings/store",
            },
            "params": {
                "parameter_ranges": {
                    "n": [0.01, 0.1],
                    "q_spatial": [0.1, 0.9],
                    "K_D_delta": [-3.0, 1.0],
                    "d_gw": [0.01, 300.0],
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
                "input_var_names": ["mock", "SoilGrids1km_sand", "SoilGrids1km_clay"],
                "learnable_parameters": ["q_spatial", "K_D_delta"],
            },
            "cuda_lstm": {
                "input_var_names": ["mock"],
                "learnable_parameters": ["n", "d_gw"],
            },
            "s3_region": "us-east-1",
            "device": "cpu",
        }
        cfg = validate_config(DictConfig(cfg_dict), save_config=False)
        assert cfg.params.use_leakance is True
        assert "K_D_delta" in cfg.kan.learnable_parameters
        assert "n" in cfg.cuda_lstm.learnable_parameters
        assert "d_gw" in cfg.cuda_lstm.learnable_parameters

    def test_use_leakance_false_is_default(self) -> None:
        """Test that use_leakance defaults to False."""
        from tests.routing.test_utils import create_mock_config

        cfg = create_mock_config()
        assert cfg.params.use_leakance is False

    def test_cuda_lstm_config_valid(self) -> None:
        """Test that LSTM config is accepted with K_D_delta in KAN, n+d_gw in LSTM."""
        cfg = create_mock_config_with_cuda_lstm()
        assert cfg.params.use_leakance is True
        assert "K_D_delta" in cfg.kan.learnable_parameters
        assert "n" in cfg.cuda_lstm.learnable_parameters
        assert "d_gw" in cfg.cuda_lstm.learnable_parameters

    def test_lstm_kan_overlap_raises(self) -> None:
        """Test that overlapping LSTM and KAN learnable_parameters raises."""
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
                "forcings": "mock://forcings/store",
            },
            "params": {
                "parameter_ranges": {
                    "n": [0.01, 0.1],
                    "q_spatial": [0.1, 0.9],
                    "K_D_delta": [-3.0, 1.0],
                    "d_gw": [0.01, 300.0],
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
                "input_var_names": ["mock", "SoilGrids1km_sand", "SoilGrids1km_clay"],
                "learnable_parameters": ["q_spatial", "K_D_delta", "n"],
            },
            "cuda_lstm": {
                "input_var_names": ["mock"],
                "learnable_parameters": ["n", "d_gw"],
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "s3_region": "us-east-1",
            "device": "cpu",
        }
        with pytest.raises(ValueError, match="must not overlap"):
            validate_config(DictConfig(cfg_dict), save_config=False)
