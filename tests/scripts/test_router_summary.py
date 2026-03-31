"""Tests for the routing summary printed after ``ddr route`` completes."""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# router.py lives under scripts/ which is not a package, so we import
# the print_routing_summary function by adding the scripts directory to
# sys.path.  This mirrors how the script is invoked at the CLI.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from router import print_routing_summary


def _make_dataset(
    n_segments: int = 5,
    n_timesteps: int = 10,
    start: str = "1995-10-01",
) -> xr.Dataset:
    """Create a minimal routing output dataset for testing."""
    rng = np.random.default_rng(42)
    data = rng.random((n_segments, n_timesteps)).astype(np.float32) * 100
    seg_ids = [f"seg-{i}" for i in range(n_segments)]
    times = np.arange(start, n_timesteps, dtype="datetime64[D]")
    pred_da = xr.DataArray(
        data=data,
        dims=["catchment_ids", "time"],
        coords={"catchment_ids": seg_ids, "time": times},
        attrs={"units": "m3/s"},
    )
    return xr.Dataset(data_vars={"predictions": pred_da})


class TestPrintRoutingSummary:
    """Tests for print_routing_summary()."""

    def test_returns_string(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=12.4)
        assert isinstance(result, str)

    def test_contains_segment_count(self, tmp_path: Path) -> None:
        ds = _make_dataset(n_segments=2847)
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        assert "2,847" in result

    def test_contains_time_range(self, tmp_path: Path) -> None:
        ds = _make_dataset(start="1995-10-01", n_timesteps=10)
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        assert "1995-10-01" in result
        assert "1995-10-10" in result

    def test_contains_discharge_stats(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        assert "Min:" in result
        assert "Mean:" in result
        assert "Max:" in result

    def test_contains_runtime(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=42.7)
        assert "42.7s" in result

    def test_contains_plot_path_when_provided(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        plot = tmp_path / "routing_summary.png"
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0, plot_path=plot)
        assert "routing_summary.png" in result

    def test_omits_plot_line_when_no_path(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        assert "Plot:" not in result

    def test_contains_zarr_output_path(self, tmp_path: Path) -> None:
        ds = _make_dataset()
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        assert "chrout.zarr" in result

    def test_single_segment_single_timestep(self, tmp_path: Path) -> None:
        ds = _make_dataset(n_segments=1, n_timesteps=1)
        result = print_routing_summary(ds, save_path=tmp_path, runtime_seconds=0.1)
        # Should not crash and should contain the expected sections
        assert "DDR Routing Complete" in result
        assert "Segments routed:" in result

    def test_prints_to_stdout(self, tmp_path: Path, capsys: object) -> None:
        """Verify the summary is actually printed, not just returned."""
        import _pytest.capture

        assert isinstance(capsys, _pytest.capture.CaptureFixture)
        ds = _make_dataset()
        print_routing_summary(ds, save_path=tmp_path, runtime_seconds=1.0)
        captured = capsys.readouterr()
        assert "DDR Routing Complete" in captured.out
