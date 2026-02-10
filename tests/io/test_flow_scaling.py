"""Unit tests for flow scaling pure functions."""

import pytest
import torch

from ddr.io.readers import build_flow_scale_tensor, compute_flow_scale_factor


class TestComputeFlowScaleFactor:
    """Tests for compute_flow_scale_factor."""

    @pytest.mark.parametrize(
        "drain, comid_drain, unit_area, expected",
        [
            # Negative diff within unit area → fraction < 1
            (90, 100, 20, 0.5),
            (95, 100, 20, 0.75),
            (80, 100, 20, 1.0),  # abs(diff)=20 >= unit_area=20 → 1.0
            # Positive diff → no scaling
            (110, 100, 20, 1.0),
            (100, 100, 20, 1.0),  # diff == 0, >= 0
            # Zero / negative unit area → 1.0
            (90, 100, 0, 1.0),
            (90, 100, -5, 1.0),
            # abs(diff) > unit area → 1.0
            (70, 100, 20, 1.0),
            # Tiny diff
            (99.9, 100, 20, (20 - 0.1) / 20),
        ],
    )
    def test_core_formula(self, drain, comid_drain, unit_area, expected):
        result = compute_flow_scale_factor(drain, comid_drain, unit_area)
        assert result == pytest.approx(expected, abs=1e-9)

    @pytest.mark.parametrize(
        "drain, comid_drain, unit_area",
        [
            (float("nan"), 100, 20),
            (90, float("nan"), 20),
            (90, 100, float("nan")),
            (float("nan"), float("nan"), float("nan")),
        ],
    )
    def test_nan_inputs_return_one(self, drain, comid_drain, unit_area):
        assert compute_flow_scale_factor(drain, comid_drain, unit_area) == 1.0


class TestBuildFlowScaleTensor:
    """Tests for build_flow_scale_tensor."""

    def _make_gage_dict(self, staids, drains, comid_drains, unit_areas):
        return {
            "STAID": staids,
            "DRAIN_SQKM": drains,
            "COMID_DRAIN_SQKM": comid_drains,
            "COMID_UNITAREA_SQKM": unit_areas,
        }

    def test_correct_shape_and_values(self):
        gage_dict = self._make_gage_dict(
            staids=["01000000", "02000000"],
            drains=[90.0, 50.0],
            comid_drains=[100.0, 60.0],
            unit_areas=[20.0, 20.0],
        )
        # gage 0 → seg 2, gage 1 → seg 5
        result = build_flow_scale_tensor(
            batch=["01000000", "02000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[2, 5],
            num_segments=8,
        )
        assert result.shape == (8,)
        assert result[2].item() == pytest.approx(0.5)
        assert result[5].item() == pytest.approx(0.5)
        # All other segments should be 1.0
        for i in [0, 1, 3, 4, 6, 7]:
            assert result[i].item() == 1.0

    def test_missing_metadata_returns_all_ones(self):
        gage_dict: dict[str, list] = {
            "STAID": ["01000000"],
            "DRAIN_SQKM": [90.0],
            # Missing COMID_DRAIN_SQKM and COMID_UNITAREA_SQKM
        }
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=5,
        )
        assert result.shape == (5,)
        assert torch.all(result == 1.0)

    def test_missing_comid_unitarea_returns_all_ones(self):
        gage_dict: dict[str, list] = {
            "STAID": ["01000000"],
            "DRAIN_SQKM": [90.0],
            "COMID_DRAIN_SQKM": [100.0],
            # Missing COMID_UNITAREA_SQKM
        }
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=5,
        )
        assert torch.all(result == 1.0)

    def test_gage_not_in_dict_skipped(self):
        """A STAID in batch that's not in gage_dict should be silently skipped."""
        gage_dict = self._make_gage_dict(
            staids=["01000000"],
            drains=[90.0],
            comid_drains=[100.0],
            unit_areas=[20.0],
        )
        result = build_flow_scale_tensor(
            batch=["99999999"],  # not in gage_dict
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=3,
        )
        assert torch.all(result == 1.0)


class TestBuildFlowScaleTensorFromCSV:
    """Tests for the FLOW_SCALE fast path in build_flow_scale_tensor."""

    def test_uses_precomputed_flow_scale(self):
        """When FLOW_SCALE is in gage_dict, use it directly."""
        gage_dict: dict[str, list] = {
            "STAID": ["01000000", "02000000"],
            "FLOW_SCALE": [0.75, 0.5],
        }
        result = build_flow_scale_tensor(
            batch=["01000000", "02000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[1, 3],
            num_segments=5,
        )
        assert result[1].item() == pytest.approx(0.75)
        assert result[3].item() == pytest.approx(0.5)
        assert result[0].item() == 1.0
        assert result[2].item() == 1.0
        assert result[4].item() == 1.0

    def test_precomputed_overrides_computation(self):
        """FLOW_SCALE takes precedence over raw COMID_DRAIN_SQKM/COMID_UNITAREA_SQKM columns."""
        gage_dict: dict[str, list] = {
            "STAID": ["01000000"],
            "DRAIN_SQKM": [90.0],
            "COMID_DRAIN_SQKM": [100.0],
            "COMID_UNITAREA_SQKM": [20.0],
            "FLOW_SCALE": [0.99],
        }
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=3,
        )
        # Should use FLOW_SCALE (0.99), not computed value (0.5)
        assert result[0].item() == pytest.approx(0.99)

    def test_fallback_when_no_flow_scale(self):
        """Without FLOW_SCALE, falls back to computation from raw columns."""
        gage_dict: dict[str, list] = {
            "STAID": ["01000000"],
            "DRAIN_SQKM": [90.0],
            "COMID_DRAIN_SQKM": [100.0],
            "COMID_UNITAREA_SQKM": [20.0],
        }
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=3,
        )
        assert result[0].item() == pytest.approx(0.5)

    def test_nan_flow_scale_defaults_to_one(self):
        """NaN in FLOW_SCALE should default to 1.0."""
        gage_dict: dict[str, list] = {
            "STAID": ["01000000"],
            "FLOW_SCALE": [float("nan")],
        }
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=3,
        )
        assert result[0].item() == 1.0
