"""Tests for the mass-conserving gauge outflow extraction in the MERIT dataset."""

import numpy as np

from ddr.geodatazoo.merit import _gage_outflow_indices


class TestGageOutflowIndices:
    def test_returns_gauge_reach_itself(self) -> None:
        # CONUS index -> compressed index
        index_mapping = {100: 0, 200: 1, 300: 2}
        gage_idx = np.array([300])

        result = _gage_outflow_indices(gage_idx, index_mapping)

        # The gauge reach itself, NOT its upstream columns: the MC solve at the
        # gauge reach already carries upstream flow plus its own lateral inflow.
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.array([2]))

    def test_one_entry_per_gauge_in_order(self) -> None:
        index_mapping = {10: 0, 20: 1, 30: 2, 40: 3}
        gage_idx = np.array([40, 10])

        result = _gage_outflow_indices(gage_idx, index_mapping)

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([3]))
        np.testing.assert_array_equal(result[1], np.array([0]))

    def test_headwater_gauge(self) -> None:
        # A headwater gauge (no upstream reaches) maps to its own reach — same
        # rule, no special case needed anymore.
        index_mapping = {7: 0}
        result = _gage_outflow_indices(np.array([7]), index_mapping)
        np.testing.assert_array_equal(result[0], np.array([0]))
