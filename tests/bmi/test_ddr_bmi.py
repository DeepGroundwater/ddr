"""Unit tests for the DDR BMI wrapper.

Tests verify that the BMI interface contract is satisfied without
requiring real hydrofabric data or trained KAN checkpoints. Heavy
integration tests (marked ``@pytest.mark.integration``) test against
real data on HPC.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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


@pytest.fixture
def mock_bmi(bmi: DdrBmi) -> DdrBmi:
    """Create a BMI instance with mocked internal state for testing get/set."""
    bmi._initialized = True
    bmi._num_segments = 3
    bmi._segment_ids = np.array([101, 102, 103], dtype=np.int64)
    bmi._lateral_inflow = np.zeros(3, dtype=np.float64)
    bmi._prev_lateral_inflow = np.zeros(3, dtype=np.float64)
    bmi._has_prev_inflow = False
    bmi._nexus_ids = np.empty(0, dtype=np.int32)
    bmi._nexus_to_seg_idx = {101: 0, 102: 1, 103: 2}
    bmi._discharge = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    bmi._velocity = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    bmi._depth = np.array([0.1, 0.2, 0.3], dtype=np.float32)

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


@pytest.fixture
def stepping_bmi(bmi: DdrBmi) -> DdrBmi:
    """BMI configured for sub-stepping: 15-min routing with 1-hour coupling."""
    bmi._initialized = True
    bmi._cold_started = True
    bmi._num_segments = 3
    bmi._timestep = 900.0  # 15-min routing
    bmi._interpolation = "constant"
    bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])
    bmi._prev_lateral_inflow = np.zeros(3, dtype=np.float64)
    bmi._has_prev_inflow = False
    bmi._device = "cpu"
    bmi._discharge = np.zeros(3, dtype=np.float32)
    bmi._velocity = np.zeros(3, dtype=np.float32)
    bmi._depth = np.zeros(3, dtype=np.float32)
    bmi._segment_ids = np.array([1, 2, 3], dtype=np.int64)

    bmi._cfg = MagicMock()
    bmi._cfg.params.attribute_minimums = {"discharge": 0.001}

    mc = MagicMock()
    mc._discharge_t = torch.tensor([1.0, 2.0, 3.0])
    mc.discharge_lb = torch.tensor(0.001)
    mc.route_timestep = MagicMock(return_value=torch.tensor([1.5, 2.5, 3.5]))
    mc.n = torch.tensor([0.03, 0.04, 0.05])
    mc.slope = torch.tensor([0.001, 0.002, 0.003])
    mc.q_spatial = torch.tensor([0.5, 0.5, 0.5])
    mc.p_spatial = torch.tensor([10.0, 10.0, 10.0])
    mc.top_width = torch.tensor([20.0, 30.0, 40.0])
    mc.side_slope = torch.tensor([2.0, 2.0, 2.0])
    mc.depth_lb = 0.01
    mc.bottom_width_lb = 0.1
    bmi._mc = mc
    bmi._mapper = MagicMock()
    return bmi


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

    def test_get_value_ptr_returns_empty_before_init(self, bmi):
        """get_value_ptr returns the empty pre-init arrays."""
        arr = bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        assert len(arr) == 0

    def test_get_value_ptr_unknown_raises(self, bmi):
        with pytest.raises(ValueError, match="Unknown output variable"):
            bmi.get_value_ptr("nonexistent_variable")


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
        assert cfg.interpolation == "constant"

    def test_custom_values(self):
        cfg = BmiInitConfig(
            ddr_config="/tmp/ddr.yaml",
            kan_checkpoint="/tmp/kan.pt",
            device="cuda:0",
            timestep_seconds=900.0,
            interpolation="linear",
        )
        assert cfg.device == "cuda:0"
        assert cfg.timestep_seconds == 900.0
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

    def test_grid_size_after_init(self, mock_bmi):
        assert mock_bmi.get_grid_size(0) == 3

    def test_grid_node_count(self, mock_bmi):
        assert mock_bmi.get_grid_node_count(0) == 3

    def test_grid_edge_count(self, mock_bmi):
        assert mock_bmi.get_grid_edge_count(0) == 2


# ---------------------------------------------------------------------------
# get_value / get_value_ptr (persistent output arrays)
# ---------------------------------------------------------------------------


class TestGetValue:
    """Verify get_value copies from persistent arrays and get_value_ptr returns references."""

    def test_get_discharge(self, mock_bmi):
        dest = np.empty(3, dtype=np.float32)
        result = mock_bmi.get_value("channel_exit_water_x-section__volume_flow_rate", dest)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_get_segment_ids(self, mock_bmi):
        dest = np.empty(3, dtype=np.int64)
        result = mock_bmi.get_value("channel_water__id", dest)
        np.testing.assert_array_equal(result, [101, 102, 103])

    def test_get_velocity(self, mock_bmi):
        dest = np.empty(3, dtype=np.float32)
        result = mock_bmi.get_value("channel_water_flow__speed", dest)
        np.testing.assert_array_almost_equal(result, [0.5, 1.0, 1.5])

    def test_get_depth(self, mock_bmi):
        dest = np.empty(3, dtype=np.float32)
        result = mock_bmi.get_value("channel_water__mean_depth", dest)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3])

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


class TestGetValuePtr:
    """Verify get_value_ptr returns stable references (not copies)."""

    def test_ptr_returns_reference_not_copy(self, mock_bmi):
        """Modifying the returned array should modify the internal state."""
        ptr = mock_bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        assert ptr is mock_bmi._discharge

    def test_ptr_velocity_is_reference(self, mock_bmi):
        ptr = mock_bmi.get_value_ptr("channel_water_flow__speed")
        assert ptr is mock_bmi._velocity

    def test_ptr_depth_is_reference(self, mock_bmi):
        ptr = mock_bmi.get_value_ptr("channel_water__mean_depth")
        assert ptr is mock_bmi._depth

    def test_ptr_segment_ids_is_reference(self, mock_bmi):
        ptr = mock_bmi.get_value_ptr("channel_water__id")
        assert ptr is mock_bmi._segment_ids

    def test_ptr_stability_after_in_place_update(self, mock_bmi):
        """get_value_ptr pointer remains valid after in-place array mutation."""
        ptr_before = mock_bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        # Simulate in-place update (what _update_output_cache does)
        mock_bmi._discharge[:] = [9.0, 8.0, 7.0]
        ptr_after = mock_bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        assert ptr_before is ptr_after
        np.testing.assert_array_equal(ptr_before, [9.0, 8.0, 7.0])

    def test_all_output_vars_have_ptr(self, mock_bmi):
        """Every output variable should be accessible via get_value_ptr."""
        for name in _OUTPUT_VAR_NAMES:
            ptr = mock_bmi.get_value_ptr(name)
            assert isinstance(ptr, np.ndarray)


# ---------------------------------------------------------------------------
# set_value (nexus remapping)
# ---------------------------------------------------------------------------


class TestSetValue:
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

    def test_set_value_at_indices(self, mock_bmi):
        mock_bmi.set_value_at_indices(
            "land_surface_water_source__volume_flow_rate",
            np.array([0, 2]),
            np.array([5.0, 15.0]),
        )
        assert mock_bmi._lateral_inflow[0] == 5.0
        assert mock_bmi._lateral_inflow[2] == 15.0

    def test_set_zero_inflows(self, mock_bmi):
        """Zero inflows should be accepted without error."""
        mock_bmi.set_value(
            "land_surface_water_source__volume_flow_rate",
            np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(mock_bmi._lateral_inflow, [0.0, 0.0, 0.0])

    def test_set_negative_inflows(self, mock_bmi):
        """Negative inflows should be accepted (clamping happens in update)."""
        mock_bmi.set_value(
            "land_surface_water_source__volume_flow_rate",
            np.array([-1.0, 0.0, 5.0]),
        )
        assert mock_bmi._lateral_inflow[0] == -1.0


# ---------------------------------------------------------------------------
# Output cache update
# ---------------------------------------------------------------------------


class TestOutputCache:
    """Verify _update_output_cache populates persistent arrays from MC state."""

    def test_cache_updates_discharge(self, mock_bmi):
        mock_bmi._mc._discharge_t = torch.tensor([10.0, 20.0, 30.0])
        mock_bmi._update_output_cache()
        np.testing.assert_array_almost_equal(mock_bmi._discharge, [10.0, 20.0, 30.0])

    def test_cache_updates_velocity_and_depth(self, mock_bmi):
        """Verify velocity/depth against hand-computed expected values."""
        mock_bmi._update_output_cache()
        # Hand-compute expected depth for segment 0:
        #   Q=1.0, n=0.03, slope=0.001, q_spatial=0.5, p_spatial=10.0
        #   q_eps = 0.5 + 1e-6 ≈ 0.5
        #   num = 1.0 * 0.03 * (0.5 + 1) = 0.045
        #   den = 10.0 * sqrt(0.001) = 0.31623
        #   depth = (num / den) ^ (3 / (5 + 3*0.5)) = 0.14230 ^ 0.46154
        q = torch.tensor([1.0, 2.0, 3.0])
        n = torch.tensor([0.03, 0.04, 0.05])
        s0 = torch.tensor([0.001, 0.002, 0.003])
        q_eps = torch.tensor([0.5, 0.5, 0.5]) + 1e-6
        p = torch.tensor([10.0, 10.0, 10.0])
        tw = torch.tensor([20.0, 30.0, 40.0])
        z = torch.tensor([2.0, 2.0, 2.0])

        num = q * n * (q_eps + 1)
        den = p * torch.pow(s0, 0.5)
        expected_depth = torch.pow(num / (den + 1e-8), 3.0 / (5.0 + 3.0 * q_eps))
        expected_depth = torch.clamp(expected_depth, min=0.01)

        bw = torch.clamp(tw - 2 * z * expected_depth, min=0.1)
        area = (tw + bw) * expected_depth / 2
        wp = bw + 2 * expected_depth * torch.sqrt(1 + z**2)
        R = area / wp
        expected_v = (1.0 / n) * torch.pow(R, 2.0 / 3.0) * torch.pow(s0, 0.5)
        expected_v = torch.clamp(expected_v, min=0.0, max=15.0)

        np.testing.assert_array_almost_equal(mock_bmi._depth, expected_depth.numpy(), decimal=5)
        np.testing.assert_array_almost_equal(mock_bmi._velocity, expected_v.numpy(), decimal=5)

    def test_cache_uses_inplace_mutation(self, mock_bmi):
        """Arrays should be the same objects before and after cache update."""
        discharge_ref = mock_bmi._discharge
        velocity_ref = mock_bmi._velocity
        depth_ref = mock_bmi._depth
        mock_bmi._update_output_cache()
        assert mock_bmi._discharge is discharge_ref
        assert mock_bmi._velocity is velocity_ref
        assert mock_bmi._depth is depth_ref

    def test_cache_noop_when_mc_is_none(self, bmi):
        """_update_output_cache should not crash when MC engine is None."""
        bmi._mc = None
        bmi._update_output_cache()  # should not raise


# ---------------------------------------------------------------------------
# Nexus mapping
# ---------------------------------------------------------------------------


class TestNexusMapping:
    """Test _build_nexus_mapping from a temporary GeoPackage."""

    @pytest.fixture
    def gpkg_with_flowpaths(self, tmp_path):
        """Create a minimal GeoPackage (SQLite) with flowpaths table."""
        import sqlite3

        gpkg = tmp_path / "test.gpkg"
        con = sqlite3.connect(str(gpkg))
        con.execute("CREATE TABLE flowpaths (id TEXT, toid TEXT)")
        # wb-101 → nex-201, wb-102 → nex-202, wb-103 → nex-203
        con.execute("INSERT INTO flowpaths VALUES ('wb-101', 'nex-201')")
        con.execute("INSERT INTO flowpaths VALUES ('wb-102', 'nex-202')")
        con.execute("INSERT INTO flowpaths VALUES ('wb-103', 'nex-203')")
        con.commit()
        con.close()
        return gpkg

    def test_builds_correct_mapping(self, bmi, gpkg_with_flowpaths):
        """Nexus 201 should map to the array index of wb-101."""
        divide_ids = ["cat-101", "cat-102", "cat-103"]
        bmi._num_segments = 3
        mapping = bmi._build_nexus_mapping(gpkg_with_flowpaths, divide_ids)
        # nex-201 → wb-101 → idx 0
        assert mapping[201] == 0
        # nex-202 → wb-102 → idx 1
        assert mapping[202] == 1
        # nex-203 → wb-103 → idx 2
        assert mapping[203] == 2

    def test_mapping_with_different_ids(self, bmi, tmp_path):
        """Nexus and flowpath IDs don't have to share the same number."""
        import sqlite3

        gpkg = tmp_path / "diff_ids.gpkg"
        con = sqlite3.connect(str(gpkg))
        con.execute("CREATE TABLE flowpaths (id TEXT, toid TEXT)")
        # wb-50 → nex-999, wb-51 → nex-1000
        con.execute("INSERT INTO flowpaths VALUES ('wb-50', 'nex-999')")
        con.execute("INSERT INTO flowpaths VALUES ('wb-51', 'nex-1000')")
        con.commit()
        con.close()

        divide_ids = ["cat-50", "cat-51"]
        bmi._num_segments = 2
        mapping = bmi._build_nexus_mapping(gpkg, divide_ids)
        assert mapping[999] == 0  # nex-999 → wb-50 → idx 0
        assert mapping[1000] == 1  # nex-1000 → wb-51 → idx 1

    def test_fallback_on_missing_table(self, bmi, tmp_path):
        """Falls back to identity mapping if flowpaths table doesn't exist."""
        import sqlite3

        gpkg = tmp_path / "empty.gpkg"
        con = sqlite3.connect(str(gpkg))
        con.execute("CREATE TABLE other_table (x INT)")
        con.commit()
        con.close()

        divide_ids = ["cat-10", "cat-20"]
        bmi._num_segments = 2
        mapping = bmi._build_nexus_mapping(gpkg, divide_ids)
        # Fallback: ID == index
        assert mapping[10] == 0
        assert mapping[20] == 1

    def test_mapping_with_no_divide_ids(self, bmi, gpkg_with_flowpaths):
        """When divide_ids is None, use sequential indices."""
        bmi._num_segments = 3
        mapping = bmi._build_nexus_mapping(gpkg_with_flowpaths, None)
        # With no divide_ids, seg_id_to_idx is {0: 0, 1: 1, 2: 2}
        # No flowpath IDs match integers 0, 1, 2, so mapping should fall back
        # to what the GeoPackage provides (wb-101 etc don't match 0,1,2)
        # The result depends on whether any fp_int matches seg_id_to_idx
        assert isinstance(mapping, dict)


# ---------------------------------------------------------------------------
# Sub-stepping and interpolation
# ---------------------------------------------------------------------------


class TestSubStepping:
    """Test update_until sub-stepping with constant and linear interpolation."""

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
        called_inflows: list[np.ndarray] = []

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
        stepping_bmi._interpolation = "linear"
        stepping_bmi._has_prev_inflow = True
        stepping_bmi._prev_lateral_inflow = np.array([0.0, 0.0, 0.0])
        stepping_bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])

        called_inflows: list[np.ndarray] = []

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
        stepping_bmi._interpolation = "linear"
        stepping_bmi._has_prev_inflow = False

        called_inflows: list[np.ndarray] = []

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
        stepping_bmi._interpolation = "linear"

        called_inflows: list[np.ndarray] = []

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
# update() delegates to update_until()
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_advances_by_one_timestep(self, stepping_bmi):
        stepping_bmi._timestep = 3600.0
        stepping_bmi.update()
        assert stepping_bmi._current_time == pytest.approx(3600.0)
        assert stepping_bmi._mc.route_timestep.call_count == 1

    def test_update_twice_advances_two_timesteps(self, stepping_bmi):
        stepping_bmi._timestep = 3600.0
        stepping_bmi.update()
        stepping_bmi._lateral_inflow = np.array([5.0, 10.0, 15.0])
        stepping_bmi.update()
        assert stepping_bmi._current_time == pytest.approx(7200.0)
        assert stepping_bmi._mc.route_timestep.call_count == 2


# ---------------------------------------------------------------------------
# Finalize and re-initialization
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

    def test_finalize_then_update_raises(self, mock_bmi):
        """After finalize, update should raise."""
        mock_bmi.finalize()
        with pytest.raises(RuntimeError, match="not initialized"):
            mock_bmi.update()


# ---------------------------------------------------------------------------
# Cold-start
# ---------------------------------------------------------------------------


class TestColdStart:
    """Verify cold-start happens exactly once on the first update_until call."""

    def test_cold_start_triggers_on_first_update(self, stepping_bmi):
        stepping_bmi._cold_started = False
        hotstart_return = torch.tensor([1.0, 2.0, 3.0])
        with patch("ddr.bmi.ddr_bmi.compute_hotstart_discharge", return_value=hotstart_return):
            stepping_bmi.update_until(3600.0)
        assert stepping_bmi._cold_started is True

    def test_cold_start_does_not_retrigger(self, stepping_bmi):
        """After first update, cold_started stays True through subsequent updates."""
        stepping_bmi._cold_started = False
        hotstart_return = torch.tensor([1.0, 2.0, 3.0])
        with patch("ddr.bmi.ddr_bmi.compute_hotstart_discharge", return_value=hotstart_return):
            stepping_bmi.update_until(3600.0)
            stepping_bmi._lateral_inflow = np.array([5.0, 10.0, 15.0])
            stepping_bmi.update_until(7200.0)
        assert stepping_bmi._cold_started is True


# ---------------------------------------------------------------------------
# Pointer stability across update_until
# ---------------------------------------------------------------------------


class TestPointerStabilityAcrossUpdate:
    """Verify get_value_ptr returns the same object after update_until."""

    def test_discharge_ptr_stable_across_update(self, stepping_bmi):
        ptr_before = stepping_bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        stepping_bmi.update_until(3600.0)
        ptr_after = stepping_bmi.get_value_ptr("channel_exit_water_x-section__volume_flow_rate")
        assert ptr_before is ptr_after

    def test_velocity_ptr_stable_across_update(self, stepping_bmi):
        ptr_before = stepping_bmi.get_value_ptr("channel_water_flow__speed")
        stepping_bmi.update_until(3600.0)
        ptr_after = stepping_bmi.get_value_ptr("channel_water_flow__speed")
        assert ptr_before is ptr_after

    def test_depth_ptr_stable_across_update(self, stepping_bmi):
        ptr_before = stepping_bmi.get_value_ptr("channel_water__mean_depth")
        stepping_bmi.update_until(3600.0)
        ptr_after = stepping_bmi.get_value_ptr("channel_water__mean_depth")
        assert ptr_before is ptr_after


# ---------------------------------------------------------------------------
# No-op guard (update_until with remaining <= 0)
# ---------------------------------------------------------------------------


class TestNoOpGuard:
    """Verify update_until is a no-op when target time <= current time."""

    def test_noop_preserves_inflows(self, stepping_bmi):
        """Inflows should NOT be consumed when remaining <= 0."""
        stepping_bmi._current_time = 3600.0
        stepping_bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])
        stepping_bmi.update_until(3600.0)  # same time → no-op
        np.testing.assert_array_equal(stepping_bmi._lateral_inflow, [10.0, 20.0, 30.0])

    def test_noop_does_not_route(self, stepping_bmi):
        """route_timestep should not be called when remaining <= 0."""
        stepping_bmi._current_time = 3600.0
        stepping_bmi.update_until(3600.0)
        assert stepping_bmi._mc.route_timestep.call_count == 0

    def test_noop_backward_time(self, stepping_bmi):
        """Backward time should also be a no-op."""
        stepping_bmi._current_time = 7200.0
        stepping_bmi._lateral_inflow = np.array([10.0, 20.0, 30.0])
        stepping_bmi.update_until(3600.0)
        np.testing.assert_array_equal(stepping_bmi._lateral_inflow, [10.0, 20.0, 30.0])
        assert stepping_bmi._mc.route_timestep.call_count == 0


# ---------------------------------------------------------------------------
# Prev inflow independence
# ---------------------------------------------------------------------------


class TestPrevInflowIndependence:
    """Verify _prev_lateral_inflow is a copy, not a reference to _lateral_inflow."""

    def test_prev_and_current_are_different_objects(self, stepping_bmi):
        stepping_bmi.update_until(3600.0)
        assert stepping_bmi._prev_lateral_inflow is not stepping_bmi._lateral_inflow

    def test_zeroing_current_does_not_affect_prev(self, stepping_bmi):
        stepping_bmi.update_until(3600.0)
        # _lateral_inflow is zeroed, _prev_lateral_inflow should still have old values
        np.testing.assert_array_equal(stepping_bmi._lateral_inflow, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(stepping_bmi._prev_lateral_inflow, [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# get_var_nbytes / get_var_grid
# ---------------------------------------------------------------------------


class TestVarMeta:
    def test_get_var_nbytes_raises(self, bmi):
        with pytest.raises(NotImplementedError):
            bmi.get_var_nbytes("channel_exit_water_x-section__volume_flow_rate")

    def test_get_var_grid_returns_zero(self, bmi):
        assert bmi.get_var_grid("channel_exit_water_x-section__volume_flow_rate") == 0
        assert bmi.get_var_grid("land_surface_water_source__id") == 0

    def test_get_grid_shape(self, mock_bmi):
        shape = np.empty(1, dtype=np.int32)
        result = mock_bmi.get_grid_shape(0, shape)
        assert result[0] == 3
