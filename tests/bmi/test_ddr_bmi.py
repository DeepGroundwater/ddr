"""Unit tests for the DDR BMI wrapper.

Tests verify that the BMI interface contract is satisfied without
requiring real hydrofabric data or trained KAN checkpoints. Heavy
integration tests (marked ``@pytest.mark.integration``) test against
real data on HPC.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from ddr.bmi.config import BmiInitConfig
from ddr.bmi.ddr_bmi import (
    _INPUT_VAR_NAMES,
    _OUTPUT_VAR_NAMES,
    _VAR_TYPES,
    _VAR_UNITS,
    DdrBmi,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bmi() -> DdrBmi:
    """Return an un-initialized DdrBmi instance."""
    return DdrBmi()


# ---------------------------------------------------------------------------
# Constructor / pre-init guards
# ---------------------------------------------------------------------------


class TestPreInit:
    def test_not_initialized_by_default(self, bmi):
        assert not bmi._initialized

    def test_update_before_init_raises(self, bmi):
        with pytest.raises(RuntimeError, match="not initialized"):
            bmi.update()

    def test_update_until_before_init_raises(self, bmi):
        with pytest.raises(RuntimeError, match="not initialized"):
            bmi.update_until(3600.0)

    def test_get_value_before_init_raises(self, bmi):
        dest = np.empty(1, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not initialized"):
            bmi.get_value("channel_exit_water_x-section__volume_flow_rate", dest)


# ---------------------------------------------------------------------------
# BmiInitConfig
# ---------------------------------------------------------------------------


class TestBmiInitConfig:
    def test_defaults(self):
        cfg = BmiInitConfig(
            ddr_config="/tmp/ddr.yaml",
            kan_checkpoint="/tmp/kan.pt",
        )
        assert cfg.device == "cpu"
        assert cfg.timestep_seconds == 3600.0
        assert cfg.hydrofabric_gpkg is None
        assert cfg.conus_adjacency is None

    def test_custom_values(self):
        cfg = BmiInitConfig(
            ddr_config="/tmp/ddr.yaml",
            kan_checkpoint="/tmp/kan.pt",
            device="cuda:0",
            timestep_seconds=900.0,
        )
        assert cfg.device == "cuda:0"
        assert cfg.timestep_seconds == 900.0


# ---------------------------------------------------------------------------
# Variable metadata (no initialization needed)
# ---------------------------------------------------------------------------


class TestVariableInfo:
    def test_component_name(self, bmi):
        assert bmi.get_component_name() == "DDR-MuskingumCunge"

    def test_input_var_names(self, bmi):
        names = bmi.get_input_var_names()
        assert isinstance(names, tuple)
        assert "land_surface_water_source__volume_flow_rate" in names
        assert "land_surface_water_source__id" in names

    def test_output_var_names(self, bmi):
        names = bmi.get_output_var_names()
        assert isinstance(names, tuple)
        assert "channel_exit_water_x-section__volume_flow_rate" in names
        assert "channel_water__id" in names

    def test_input_count(self, bmi):
        assert bmi.get_input_item_count() == len(_INPUT_VAR_NAMES)

    def test_output_count(self, bmi):
        assert bmi.get_output_item_count() == len(_OUTPUT_VAR_NAMES)

    def test_var_units(self, bmi):
        assert bmi.get_var_units("channel_exit_water_x-section__volume_flow_rate") == "m3 s-1"
        assert bmi.get_var_units("channel_water__id") == "-"

    def test_var_type(self, bmi):
        assert bmi.get_var_type("channel_exit_water_x-section__volume_flow_rate") == "float32"
        assert bmi.get_var_type("land_surface_water_source__id") == "int32"

    def test_var_itemsize(self, bmi):
        assert bmi.get_var_itemsize("channel_exit_water_x-section__volume_flow_rate") == 4
        assert bmi.get_var_itemsize("land_surface_water_source__id") == 4

    def test_var_location(self, bmi):
        assert bmi.get_var_location("channel_exit_water_x-section__volume_flow_rate") == "node"

    @pytest.mark.parametrize("name", list(_VAR_UNITS.keys()))
    def test_all_vars_have_units(self, bmi, name):
        assert bmi.get_var_units(name) != ""

    @pytest.mark.parametrize("name", list(_VAR_TYPES.keys()))
    def test_all_vars_have_types(self, bmi, name):
        assert bmi.get_var_type(name) != ""


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


class TestTime:
    def test_start_time(self, bmi):
        assert bmi.get_start_time() == 0.0

    def test_end_time(self, bmi):
        assert bmi.get_end_time() == float("inf")

    def test_time_units(self, bmi):
        assert bmi.get_time_units() == "s"

    def test_time_step(self, bmi):
        assert bmi.get_time_step() == 3600.0

    def test_current_time_starts_at_zero(self, bmi):
        assert bmi.get_current_time() == 0.0


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


class TestGrid:
    def test_grid_type(self, bmi):
        assert bmi.get_grid_type(0) == "unstructured"

    def test_grid_rank(self, bmi):
        assert bmi.get_grid_rank(0) == 1

    def test_grid_spacing_raises(self, bmi):
        with pytest.raises(NotImplementedError):
            bmi.get_grid_spacing(0, np.empty(1))

    def test_grid_origin_raises(self, bmi):
        with pytest.raises(NotImplementedError):
            bmi.get_grid_origin(0, np.empty(1))


# ---------------------------------------------------------------------------
# set_value / get_value (with mocked internals)
# ---------------------------------------------------------------------------


class TestSetGetValue:
    """Test set_value/get_value using a manually configured (not fully initialized) BMI."""

    @pytest.fixture
    def mock_bmi(self, bmi):
        """Create a BMI instance with mocked internal state for testing get/set."""
        import torch

        bmi._initialized = True
        bmi._num_segments = 3
        bmi._segment_ids = np.array([101, 102, 103], dtype=np.int64)
        bmi._lateral_inflow = np.zeros(3, dtype=np.float64)
        bmi._nexus_ids = np.empty(0, dtype=np.int32)
        bmi._nexus_to_seg_idx = {101: 0, 102: 1, 103: 2}

        # Mock MuskingumCunge
        mc = MagicMock()
        mc._discharge_t = torch.tensor([1.0, 2.0, 3.0])
        mc.n = torch.tensor([0.03, 0.04, 0.05])
        mc.slope = torch.tensor([0.001, 0.002, 0.003])
        mc.q_spatial = torch.tensor([0.5, 0.5, 0.5])
        mc.p_spatial = torch.tensor([10.0, 10.0, 10.0])
        mc.top_width = torch.tensor([20.0, 30.0, 40.0])
        mc.side_slope = torch.tensor([2.0, 2.0, 2.0])
        mc.depth_lb = 0.01
        mc.bottom_width_lb = 0.1
        mc.network = MagicMock()
        mc.network._nnz.return_value = 2
        bmi._mc = mc
        return bmi

    def test_set_lateral_inflow_direct(self, mock_bmi):
        """Direct array assignment when no nexus IDs are set."""
        src = np.array([10.0, 20.0, 30.0])
        mock_bmi.set_value("land_surface_water_source__volume_flow_rate", src)
        np.testing.assert_array_equal(mock_bmi._lateral_inflow, src)

    def test_set_lateral_inflow_via_nexus_ids(self, mock_bmi):
        """Nexus-indexed inflow remapping."""
        mock_bmi.set_value("land_surface_water_source__id", np.array([103, 101]))
        mock_bmi.set_value(
            "land_surface_water_source__volume_flow_rate",
            np.array([99.0, 11.0]),
        )
        assert mock_bmi._lateral_inflow[2] == 99.0  # seg 103 → idx 2
        assert mock_bmi._lateral_inflow[0] == 11.0  # seg 101 → idx 0

    def test_set_ngen_dt(self, mock_bmi):
        mock_bmi.set_value("ngen_dt", np.array([900]))
        assert mock_bmi._ngen_dt == 900

    def test_set_unknown_variable_does_not_raise(self, mock_bmi):
        """BMI convention: unknown set_value calls should not crash."""
        mock_bmi.set_value("nonexistent_variable", np.array([1.0]))

    def test_get_discharge(self, mock_bmi):
        dest = np.empty(3, dtype=np.float32)
        result = mock_bmi.get_value("channel_exit_water_x-section__volume_flow_rate", dest)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_segment_ids(self, mock_bmi):
        dest = np.empty(3, dtype=np.int64)
        result = mock_bmi.get_value("channel_water__id", dest)
        np.testing.assert_array_equal(result, [101, 102, 103])

    def test_get_unknown_output_raises(self, mock_bmi):
        dest = np.empty(3, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown output variable"):
            mock_bmi.get_value("nonexistent_output", dest)

    def test_get_value_at_indices(self, mock_bmi):
        dest = np.empty(2, dtype=np.float32)
        result = mock_bmi.get_value_at_indices(
            "channel_exit_water_x-section__volume_flow_rate",
            dest,
            np.array([0, 2]),
        )
        np.testing.assert_array_almost_equal(result, [1.0, 3.0])

    def test_set_value_at_indices(self, mock_bmi):
        mock_bmi.set_value_at_indices(
            "land_surface_water_source__volume_flow_rate",
            np.array([0, 2]),
            np.array([5.0, 15.0]),
        )
        assert mock_bmi._lateral_inflow[0] == 5.0
        assert mock_bmi._lateral_inflow[2] == 15.0

    def test_grid_size_after_init(self, mock_bmi):
        assert mock_bmi.get_grid_size(0) == 3

    def test_grid_node_count(self, mock_bmi):
        assert mock_bmi.get_grid_node_count(0) == 3

    def test_grid_edge_count(self, mock_bmi):
        assert mock_bmi.get_grid_edge_count(0) == 2


# ---------------------------------------------------------------------------
# Sub-stepping and interpolation
# ---------------------------------------------------------------------------


class TestSubStepping:
    """Test update_until sub-stepping with constant and linear interpolation."""

    @pytest.fixture
    def stepping_bmi(self, bmi):
        """BMI configured for sub-stepping: 15-min routing with 1-hour coupling."""
        import torch

        bmi._initialized = True
        bmi._cold_started = True
        bmi._num_segments = 3
        bmi._timestep = 900.0  # 15-min routing
        bmi._interpolation = "constant"
        bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])
        bmi._prev_lateral_inflow = np.zeros(3, dtype=np.float64)
        bmi._has_prev_inflow = False
        bmi._device = "cpu"

        bmi._cfg = MagicMock()
        bmi._cfg.params.attribute_minimums = {"discharge": 0.001}

        mc = MagicMock()
        mc._discharge_t = torch.tensor([1.0, 2.0, 3.0])
        mc.discharge_lb = torch.tensor(0.001)
        mc.route_timestep = MagicMock(return_value=torch.tensor([1.5, 2.5, 3.5]))
        bmi._mc = mc
        bmi._mapper = MagicMock()
        return bmi

    def test_constant_substep_count(self, stepping_bmi):
        """3600s interval / 900s step = 4 route_timestep calls."""
        stepping_bmi.update_until(3600.0)
        assert stepping_bmi._mc.route_timestep.call_count == 4

    def test_time_advances_correctly(self, stepping_bmi):
        stepping_bmi.update_until(3600.0)
        assert stepping_bmi._current_time == pytest.approx(3600.0)

    def test_inflows_cleared_after_update(self, stepping_bmi):
        stepping_bmi.update_until(3600.0)
        np.testing.assert_array_equal(stepping_bmi._lateral_inflow, [0.0, 0.0, 0.0])

    def test_prev_inflow_stored_after_update(self, stepping_bmi):
        stepping_bmi.update_until(3600.0)
        np.testing.assert_array_equal(stepping_bmi._prev_lateral_inflow, [10.0, 20.0, 30.0])
        assert stepping_bmi._has_prev_inflow

    def test_constant_uses_same_inflow_all_substeps(self, stepping_bmi):
        """All sub-steps get identical inflows under constant interpolation."""
        import torch

        called_inflows = []

        def capture_inflows(q_prime_clamp: torch.Tensor, mapper: object) -> torch.Tensor:
            called_inflows.append(q_prime_clamp.numpy().copy())
            return torch.tensor([1.5, 2.5, 3.5])

        stepping_bmi._mc.route_timestep = capture_inflows
        stepping_bmi.update_until(3600.0)

        assert len(called_inflows) == 4
        for inflow in called_inflows:
            np.testing.assert_array_almost_equal(inflow, [10.0, 20.0, 30.0])

    def test_linear_interpolation_substeps(self, stepping_bmi):
        """Linear interpolation ramps from previous to current inflows."""
        import torch

        stepping_bmi._interpolation = "linear"
        stepping_bmi._has_prev_inflow = True
        stepping_bmi._prev_lateral_inflow = np.array([0.0, 0.0, 0.0])
        stepping_bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])

        called_inflows = []

        def capture_inflows(q_prime_clamp: torch.Tensor, mapper: object) -> torch.Tensor:
            called_inflows.append(q_prime_clamp.numpy().copy())
            return torch.tensor([1.5, 2.5, 3.5])

        stepping_bmi._mc.route_timestep = capture_inflows
        stepping_bmi.update_until(3600.0)

        assert len(called_inflows) == 4
        # alpha = 1/4: 0.75*[0,0,0] + 0.25*[10,20,30] = [2.5, 5.0, 7.5]
        np.testing.assert_array_almost_equal(called_inflows[0], [2.5, 5.0, 7.5])
        # alpha = 2/4: [5.0, 10.0, 15.0]
        np.testing.assert_array_almost_equal(called_inflows[1], [5.0, 10.0, 15.0])
        # alpha = 3/4: [7.5, 15.0, 22.5]
        np.testing.assert_array_almost_equal(called_inflows[2], [7.5, 15.0, 22.5])
        # alpha = 4/4: [10.0, 20.0, 30.0]
        np.testing.assert_array_almost_equal(called_inflows[3], [10.0, 20.0, 30.0])

    def test_linear_falls_back_without_prev(self, stepping_bmi):
        """Linear interpolation uses constant when no previous inflows exist."""
        import torch

        stepping_bmi._interpolation = "linear"
        stepping_bmi._has_prev_inflow = False

        called_inflows = []

        def capture_inflows(q_prime_clamp: torch.Tensor, mapper: object) -> torch.Tensor:
            called_inflows.append(q_prime_clamp.numpy().copy())
            return torch.tensor([1.5, 2.5, 3.5])

        stepping_bmi._mc.route_timestep = capture_inflows
        stepping_bmi.update_until(3600.0)

        # Falls back to constant — all sub-steps identical
        for inflow in called_inflows:
            np.testing.assert_array_almost_equal(inflow, [10.0, 20.0, 30.0])

    def test_no_substep_when_dt_matches(self, stepping_bmi):
        """Single step when routing dt == coupling interval."""
        stepping_bmi._timestep = 3600.0
        stepping_bmi.update_until(3600.0)
        assert stepping_bmi._mc.route_timestep.call_count == 1

    def test_multi_coupling_intervals(self, stepping_bmi):
        """Simulate two ngen coupling intervals to test prev_inflow handoff."""
        import torch

        stepping_bmi._interpolation = "linear"

        called_inflows = []

        def capture_inflows(q_prime_clamp: torch.Tensor, mapper: object) -> torch.Tensor:
            called_inflows.append(q_prime_clamp.numpy().copy())
            return torch.tensor([1.5, 2.5, 3.5])

        stepping_bmi._mc.route_timestep = capture_inflows

        # First coupling: constant fallback (no prev)
        stepping_bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])
        stepping_bmi.update_until(3600.0)
        assert len(called_inflows) == 4
        # All constant (no prev)
        for inflow in called_inflows[:4]:
            np.testing.assert_array_almost_equal(inflow, [10.0, 20.0, 30.0])

        # Second coupling: linear from [10,20,30] to [20,40,60]
        stepping_bmi._lateral_inflow = np.array([20.0, 40.0, 60.0])
        stepping_bmi.update_until(7200.0)
        assert len(called_inflows) == 8
        # Sub-step 1: alpha=0.25 → 0.75*[10,20,30] + 0.25*[20,40,60] = [12.5,25,37.5]
        np.testing.assert_array_almost_equal(called_inflows[4], [12.5, 25.0, 37.5])
        # Sub-step 4: alpha=1.0 → [20,40,60]
        np.testing.assert_array_almost_equal(called_inflows[7], [20.0, 40.0, 60.0])


# ---------------------------------------------------------------------------
# Config interpolation field
# ---------------------------------------------------------------------------


class TestInterpolationConfig:
    def test_default_is_constant(self):
        cfg = BmiInitConfig(
            ddr_config="/tmp/ddr.yaml",
            kan_checkpoint="/tmp/kan.pt",
        )
        assert cfg.interpolation == "constant"

    def test_linear(self):
        cfg = BmiInitConfig(
            ddr_config="/tmp/ddr.yaml",
            kan_checkpoint="/tmp/kan.pt",
            interpolation="linear",
        )
        assert cfg.interpolation == "linear"

    def test_invalid_interpolation_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            BmiInitConfig(
                ddr_config="/tmp/ddr.yaml",
                kan_checkpoint="/tmp/kan.pt",
                interpolation="cubic",
            )


# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_cleans_up(self, bmi):
        bmi._initialized = True
        bmi._mc = MagicMock()
        bmi._mapper = MagicMock()
        bmi.finalize()
        assert bmi._mc is None
        assert bmi._mapper is None
        assert not bmi._initialized
