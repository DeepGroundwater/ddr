"""Tests for references.geo_io.build_gage_references — compute_flow_scale and DA_VALID logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The references module isn't installed as a package — add it to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "references" / "geo_io"))

from build_gage_references import compute_flow_scale


class TestAbsDiffInline:
    """Tests for the inline ABS_DIFF = abs(DRAIN_SQKM - COMID_DRAIN_SQKM) computation."""

    def test_abs_diff_computed(self) -> None:
        df = pd.DataFrame({"DRAIN_SQKM": [100.0, 4.0], "COMID_DRAIN_SQKM": [105.0, 8.0]})
        df["ABS_DIFF"] = (df["DRAIN_SQKM"] - df["COMID_DRAIN_SQKM"]).abs()
        np.testing.assert_array_almost_equal(df["ABS_DIFF"].values, [5.0, 4.0])

    def test_abs_diff_symmetric(self) -> None:
        """ABS_DIFF should be the same regardless of which area is larger."""
        df = pd.DataFrame({"DRAIN_SQKM": [100.0, 110.0], "COMID_DRAIN_SQKM": [110.0, 100.0]})
        df["ABS_DIFF"] = (df["DRAIN_SQKM"] - df["COMID_DRAIN_SQKM"]).abs()
        np.testing.assert_array_almost_equal(df["ABS_DIFF"].values, [10.0, 10.0])


class TestDaValid:
    """Tests for DA_VALID = ABS_DIFF <= COMID_UNITAREA_SQKM."""

    def test_valid_when_within_unit_area(self) -> None:
        df = pd.DataFrame({"ABS_DIFF": [5.0, 50.0], "COMID_UNITAREA_SQKM": [10.0, 50.0]})
        da_valid = df["ABS_DIFF"] <= df["COMID_UNITAREA_SQKM"]
        assert da_valid.tolist() == [True, True]

    def test_invalid_when_exceeds_unit_area(self) -> None:
        df = pd.DataFrame({"ABS_DIFF": [100.0], "COMID_UNITAREA_SQKM": [30.0]})
        da_valid = df["ABS_DIFF"] <= df["COMID_UNITAREA_SQKM"]
        assert da_valid.tolist() == [False]


class TestComputeFlowScale:
    """Tests for compute_flow_scale()."""

    def test_no_scaling_when_gage_downstream(self) -> None:
        """scale = 1.0 when DRAIN_SQKM >= COMID_DRAIN_SQKM (gage at or downstream of outlet)."""
        df = pd.DataFrame({"DRAIN_SQKM": [200.0], "COMID_DRAIN_SQKM": [180.0], "COMID_UNITAREA_SQKM": [50.0]})
        scale = compute_flow_scale(df)
        assert scale.values[0] == 1.0

    def test_scaling_when_gage_upstream(self) -> None:
        """scale < 1.0 when gage is upstream of COMID outlet and mismatch < unit area."""
        df = pd.DataFrame({"DRAIN_SQKM": [80.0], "COMID_DRAIN_SQKM": [100.0], "COMID_UNITAREA_SQKM": [50.0]})
        scale = compute_flow_scale(df)
        expected = (50.0 - 20.0) / 50.0  # 0.6
        np.testing.assert_almost_equal(scale.values[0], expected)

    def test_no_scaling_when_mismatch_exceeds_unit_area(self) -> None:
        """scale = 1.0 when |diff| >= unit area (can't scale further)."""
        df = pd.DataFrame({"DRAIN_SQKM": [10.0], "COMID_DRAIN_SQKM": [100.0], "COMID_UNITAREA_SQKM": [50.0]})
        scale = compute_flow_scale(df)
        assert scale.values[0] == 1.0
