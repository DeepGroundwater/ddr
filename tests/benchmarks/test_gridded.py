"""Unit tests for ddr_benchmarks.gridded - CONUS gridded (DDM30) benchmark helpers.

Synthetic network used throughout (positions are topologically ordered,
dn[i] = downstream position or -1):

    0 ─┐
    1 ─┴─► 2 ──► 4      3 ──► 4      5 (isolated terminal)

    dn = [2, 2, 4, 4, -1, -1]
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("ddr_benchmarks")

from ddr_benchmarks.gridded import (
    assign_cells,
    cell_areas_km2,
    downstream_closure,
    snap_gauges,
    topo_accumulate,
)

DN = np.array([2, 2, 4, 4, -1, -1])


class TestAssignCells:
    """Points -> flat DDM30 cell ids (0.5-deg global grid, row 0 at -55.75)."""

    def test_known_cell(self):
        """(40.48N, -77.13W) lies in the cell centered (40.25, -77.25) = 138445."""
        assert assign_cells(np.array([40.48]), np.array([-77.13]))[0] == 138445

    def test_cell_center_roundtrip(self):
        lats = np.array([-55.75, 40.25, 83.75])
        lons = np.array([-179.75, -77.25, 179.75])
        cells = assign_cells(lats, lons)
        rows, cols = cells // 720, cells % 720
        np.testing.assert_allclose(rows * 0.5 - 55.75, lats)
        np.testing.assert_allclose(cols * 0.5 - 179.75, lons)


class TestDownstreamClosure:
    def test_includes_chains(self):
        """Forcing only cell 0 pulls in its downstream chain 2, 4."""
        assert downstream_closure(np.array([0]), DN).tolist() == [0, 2, 4]

    def test_union_of_forced(self):
        assert downstream_closure(np.array([0, 3]), DN).tolist() == [0, 2, 3, 4]

    def test_result_sorted_ascending(self):
        """Ascending positions preserve topological order."""
        out = downstream_closure(np.array([3, 0, 1]), DN)
        assert (np.diff(out) > 0).all()

    def test_isolated_terminal(self):
        assert downstream_closure(np.array([5]), DN).tolist() == [5]


class TestTopoAccumulate:
    def test_vector_accumulation(self):
        """Each node accumulates everything upstream of it (plus itself)."""
        acc = topo_accumulate(np.ones(6), DN)
        assert acc.tolist() == [1, 1, 3, 1, 5, 1]

    def test_time_matrix_accumulation(self):
        """(T, N) input accumulates per timestep."""
        values = np.ones((2, 6))
        values[1] = 2.0
        acc = topo_accumulate(values, DN)
        assert acc[0].tolist() == [1, 1, 3, 1, 5, 1]
        assert acc[1].tolist() == [2, 2, 6, 2, 10, 2]

    def test_rejects_non_topological(self):
        with pytest.raises(AssertionError):
            topo_accumulate(np.ones(2), np.array([1, 0]))


class TestCellAreas:
    def test_equator_cell(self):
        """A 0.5-deg cell at the equator is ~3,091 km^2."""
        assert cell_areas_km2(np.array([0.25]))[0] == pytest.approx(3091, rel=0.01)

    def test_shrinks_with_latitude(self):
        areas = cell_areas_km2(np.array([0.25, 40.25, 60.25]))
        assert (np.diff(areas) < 0).all()
        assert areas[1] == pytest.approx(3091 * np.cos(np.radians(40.25)), rel=0.01)


class TestSnapGauges:
    """3x3 DA-matched snapping of gauges to grid cells."""

    @pytest.fixture()
    def grid(self):
        # Two adjacent in-network cells: 138445 (upstream area 9000) and its
        # east neighbor 138446 (area 70000) - a mainstem vs tributary situation.
        upstream_area = {138445: 9000.0, 138446: 70000.0}
        return upstream_area

    def test_picks_da_matched_neighbor(self, grid):
        """A gauge whose nearest cell has 8x its DA snaps to the 3x3 neighbor
        with the matching drained area instead."""
        gauges = pd.DataFrame(
            {"STAID": ["01567000"], "LAT_GAGE": [40.30], "LNG_GAGE": [-76.80], "DRAIN_SQKM": [8687.0]}
        )
        # nearest center for (40.30, -76.80) is 138446 (40.25, -76.75)
        out = snap_gauges(gauges, grid)
        assert out.cell.tolist() == [138445]
        assert out.da_ratio.iloc[0] == pytest.approx(9000 / 8687, rel=1e-6)

    def test_drops_unmatchable(self, grid):
        """A gauge with no 3x3 cell within the DA tolerance is dropped."""
        gauges = pd.DataFrame(
            {"STAID": ["99999999"], "LAT_GAGE": [40.30], "LNG_GAGE": [-76.80], "DRAIN_SQKM": [500.0]}
        )
        assert len(snap_gauges(gauges, grid)) == 0

    def test_deduplicates_cells_keeping_best(self, grid):
        """Two gauges snapping to one cell keep only the better DA match."""
        gauges = pd.DataFrame(
            {
                "STAID": ["a", "b"],
                "LAT_GAGE": [40.30, 40.20],
                "LNG_GAGE": [-77.20, -77.30],
                "DRAIN_SQKM": [9100.0, 6000.0],
            }
        )
        out = snap_gauges(gauges, grid)
        assert out.STAID.tolist() == ["a"]


class TestQprimeAxisOrder:
    """A Q' store written either way round must aggregate identically.

    zarr answers an out-of-range read with fill values rather than raising, so a
    reader that indexes axes positionally turns a transposed store into an all-NaN
    lateral inflow and a silently zero baseline instead of an error.
    """

    def _store(self, dims: tuple[str, str]):  # returns an xr.Dataset; xarray is imported lazily
        import numpy as np
        import xarray as xr

        vals = np.arange(6, dtype="float32").reshape(2, 3)  # 2 divides, 3 days
        data = vals if dims[0] == "divide_id" else vals.T
        return xr.Dataset(
            {"Qr": (dims, data)},
            coords={"divide_id": [10, 20], "time": pd.date_range("2000-01-01", periods=3)},
        )

    def test_contract_and_transposed_agree(self) -> None:
        import numpy as np
        from ddr_benchmarks.gridded_runner import qprime_matrix

        a = qprime_matrix(self._store(("divide_id", "time")), [10, 20])
        b = qprime_matrix(self._store(("time", "divide_id")), [10, 20])
        assert a.shape == (3, 2)  # (time, divide)
        assert np.array_equal(a, b)
        assert np.array_equal(a[:, 0], [0.0, 1.0, 2.0])

    def test_rejects_a_shape_matching_neither_layout(self) -> None:
        import numpy as np
        import pytest as pt
        import xarray as xr
        from ddr_benchmarks.gridded_runner import qprime_matrix

        bad = xr.Dataset(
            {"Qr": (("divide_id", "band", "time"), np.zeros((2, 2, 3), dtype="float32"))},
            coords={"divide_id": [10, 20], "time": pd.date_range("2000-01-01", periods=3)},
        )
        with pt.raises(ValueError, match="divide_id"):
            qprime_matrix(bad, [10, 20])
