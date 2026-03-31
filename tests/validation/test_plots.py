"""Tests for ddr.validation.plots — routing hydrograph plot generation."""

from pathlib import Path

import numpy as np
import xarray as xr

from ddr.validation.plots import _select_plot_segments, plot_routing_hydrograph


def _make_predictions(
    n_segments: int = 5,
    n_timesteps: int = 10,
    start: str = "1995-10-01",
) -> xr.DataArray:
    """Create a synthetic predictions DataArray for testing."""
    rng = np.random.default_rng(42)
    data = rng.random((n_segments, n_timesteps)).astype(np.float32) * 100
    seg_ids = [f"seg-{i}" for i in range(n_segments)]
    times = np.arange(start, n_timesteps, dtype="datetime64[D]")
    return xr.DataArray(
        data=data,
        dims=["catchment_ids", "time"],
        coords={"catchment_ids": seg_ids, "time": times},
        attrs={"units": "m3/s", "long_name": "Streamflow"},
    )


class TestSelectPlotSegments:
    """Tests for _select_plot_segments()."""

    def test_selects_target_catchments_when_provided(self) -> None:
        pred = _make_predictions()
        result = _select_plot_segments(pred, target_catchments=["seg-1", "seg-3"])
        assert result == ["seg-1", "seg-3"]

    def test_filters_out_missing_target_catchments(self) -> None:
        pred = _make_predictions()
        result = _select_plot_segments(pred, target_catchments=["seg-1", "seg-999"])
        assert result == ["seg-1"]

    def test_all_targets_missing_falls_back_to_max_mean(self) -> None:
        pred = _make_predictions(n_segments=3, n_timesteps=5)
        pred.values[1, :] = 9999.0
        result = _select_plot_segments(pred, target_catchments=["seg-999"])
        assert result == ["seg-1"]

    def test_falls_back_to_max_mean_discharge(self) -> None:
        pred = _make_predictions(n_segments=3, n_timesteps=5)
        # Manually set segment 2 to have the highest mean
        pred.values[2, :] = 9999.0
        result = _select_plot_segments(pred)
        assert result == ["seg-2"]

    def test_single_segment(self) -> None:
        pred = _make_predictions(n_segments=1, n_timesteps=5)
        result = _select_plot_segments(pred)
        assert result == ["seg-0"]


class TestPlotRoutingHydrograph:
    """Tests for plot_routing_hydrograph()."""

    def test_creates_png_file(self, tmp_path: Path) -> None:
        pred = _make_predictions()
        out = tmp_path / "hydro.png"
        result = plot_routing_hydrograph(pred, path=out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        pred = _make_predictions()
        out = tmp_path / "nested" / "dir" / "hydro.png"
        plot_routing_hydrograph(pred, path=out)
        assert out.exists()

    def test_single_segment_plot(self, tmp_path: Path) -> None:
        pred = _make_predictions(n_segments=1, n_timesteps=10)
        out = tmp_path / "single_seg.png"
        plot_routing_hydrograph(pred, path=out)
        assert out.exists()

    def test_single_timestep_plot(self, tmp_path: Path) -> None:
        pred = _make_predictions(n_segments=3, n_timesteps=1)
        out = tmp_path / "single_ts.png"
        plot_routing_hydrograph(pred, path=out)
        assert out.exists()

    def test_multi_segment_with_targets(self, tmp_path: Path) -> None:
        pred = _make_predictions(n_segments=5)
        out = tmp_path / "multi.png"
        plot_routing_hydrograph(pred, path=out, target_catchments=["seg-0", "seg-2", "seg-4"])
        assert out.exists()

    def test_custom_dpi(self, tmp_path: Path) -> None:
        pred = _make_predictions()
        out_lo = tmp_path / "lo.png"
        out_hi = tmp_path / "hi.png"
        plot_routing_hydrograph(pred, path=out_lo, dpi=72)
        plot_routing_hydrograph(pred, path=out_hi, dpi=300)
        # Higher DPI should produce a larger file
        assert out_hi.stat().st_size > out_lo.stat().st_size
