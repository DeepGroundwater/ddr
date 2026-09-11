"""CONUS-scale acceptance benchmark for gridded (DDM30) routing.

Runs the full pipeline on real local data: MERIT dHBV2 Q' aggregated to
DDM30 cells, routed with fixed Muskingum-Cunge parameters, evaluated
against USGS daily observations at DA-matched large gauges (>5000 km2).

Thresholds are set with margin below the measured 2026-09-02 baseline
(median NSE 0.629, 296/321 gauges improved by routing, neg-solve 0.049%).
Marked integration: requires /mnt/ssd1 data stores and data/ddm30.
"""

from pathlib import Path

import pytest

pytest.importorskip("ddr_benchmarks")

from ddr_benchmarks.gridded_runner import GriddedPaths, run_conus_benchmark

REPO = Path(__file__).parents[2]
PATHS = GriddedPaths(
    ddm30_zarr=REPO / "data/ddm30/ddm30_adjacency.zarr",
    riv_shapefile=Path("/mnt/ssd1/data/merit/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"),
    qr_icechunk=Path("/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"),
    obs_icechunk=Path("/mnt/ssd1/data/icechunk/usgs_daily_observations"),
    gages_csv=REPO / "references/gage_info/dhbv2_gages.csv",
    cache_dir=REPO / "data/ddm30",
    # the area-weighted store, so the gate measures the forcing training reads
    qprime_icechunk=Path("/mnt/ssd1/data/icechunk/ddm30_conus_uh_retrospective_regridded.ic"),
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (
            PATHS.ddm30_zarr.exists()
            and PATHS.qr_icechunk.exists()
            and PATHS.qprime_icechunk is not None
            and PATHS.qprime_icechunk.exists()
        ),
        reason="local DDM30/dHBV2 data not available",
    ),
]


@pytest.fixture(scope="module")
def result():
    return run_conus_benchmark(PATHS, start="1995-10-01", end="1997-09-30")


class TestConusGriddedAcceptance:
    def test_enough_gauges_evaluated(self, result):
        """The DA-matched, obs-covered gauge set stays CONUS-scale."""
        assert len(result["gauges"]) >= 200

    def test_solver_stability(self, result):
        """Negative triangular solves stay below 0.5% of solves."""
        assert result["neg_solve_rate"] < 0.005

    def test_routed_skill(self, result):
        """Median daily NSE of routed discharge across gauges."""
        assert result["gauges"].nse_routed.median() >= 0.55

    def test_routing_beats_summed_baseline(self, result):
        """Muskingum-Cunge routing must add skill over instantaneous
        accumulation at at least 75% of gauges."""
        g = result["gauges"]
        improved = (g.nse_routed > g.nse_summed).mean()
        assert improved >= 0.75

    def test_median_gain_positive(self, result):
        g = result["gauges"]
        assert (g.nse_routed - g.nse_summed).median() > 0.0
