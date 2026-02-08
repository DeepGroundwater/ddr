"""Tests for references.geo_io.build_gage_references — compute_error_metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The references module isn't installed as a package — add it to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "references" / "geo_io"))

from build_gage_references import compute_error_metrics


class TestComputeErrorMetrics:
    """Tests for compute_error_metrics()."""

    def test_abs_diff_computed(self) -> None:
        df = pd.DataFrame({"DRAIN_SQKM": [100.0, 4.0], "COMID_DRAIN_SQKM": [105.0, 8.0]})
        result = compute_error_metrics(df)
        assert "ABS_DIFF" in result.columns
        np.testing.assert_array_almost_equal(result["ABS_DIFF"].values, [5.0, 4.0])

    def test_small_basin_abs_diff_vs_rel_err(self) -> None:
        """Demonstrate that ABS_DIFF is small even when REL_ERR is large for small basins."""
        df = pd.DataFrame({"DRAIN_SQKM": [4.0], "COMID_DRAIN_SQKM": [8.0]})
        result = compute_error_metrics(df)
        assert result["ABS_DIFF"].values[0] == 4.0  # Small absolute diff
        assert abs(result["REL_ERR"].values[0]) == 1.0  # 100% relative error

    def test_pct_diff_sign_convention(self) -> None:
        df = pd.DataFrame({"DRAIN_SQKM": [100.0], "COMID_DRAIN_SQKM": [110.0]})
        result = compute_error_metrics(df)
        assert result["PCT_DIFF"].values[0] < 0  # Negative when COMID > USGS

    def test_existing_columns_preserved(self) -> None:
        """Ensure existing PCT_DIFF and REL_ERR still computed correctly."""
        df = pd.DataFrame({"DRAIN_SQKM": [200.0], "COMID_DRAIN_SQKM": [190.0]})
        result = compute_error_metrics(df)
        assert "PCT_DIFF" in result.columns
        assert "REL_ERR" in result.columns
        assert "ABS_DIFF" in result.columns
        np.testing.assert_almost_equal(result["ABS_DIFF"].values[0], 10.0)
        assert result["PCT_DIFF"].values[0] > 0  # Positive when USGS > COMID
