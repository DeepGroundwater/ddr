"""Tests for benchmark utility functions — reorder, save_results, load_summed_q_prime."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xarray as xr

pytest.importorskip("ddr_benchmarks")

# ============================================================================
# Mock RivTree — just needs `nodes_idx` as a pandas Series
# ============================================================================
import pandas as pd
from ddr_benchmarks.benchmark import (
    load_summed_q_prime,
    reorder_to_diffroute,
    reorder_to_topo,
    save_results,
)


def _make_riv(nodes_idx: pd.Series):
    """Create a mock RivTree-like object with nodes_idx."""
    riv = MagicMock()
    riv.nodes_idx = nodes_idx
    return riv


# ============================================================================
# Reorder functions
# ============================================================================


class TestReorderToDiffroute:
    """Tests for reorder_to_diffroute()."""

    def test_shape(self) -> None:
        topo_order = np.array([10, 20, 30, 40, 50])
        nodes_idx = pd.Series([0, 1, 2, 3, 4], index=[10, 20, 30, 40, 50])
        riv = _make_riv(nodes_idx)
        data = torch.rand(1, 5, 10)
        result = reorder_to_diffroute(data, topo_order, riv)
        assert result.shape == data.shape

    def test_identity(self) -> None:
        """When orders match, output == input."""
        topo_order = np.array([10, 20, 30])
        nodes_idx = pd.Series([0, 1, 2], index=[10, 20, 30])
        riv = _make_riv(nodes_idx)
        data = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        result = reorder_to_diffroute(data, topo_order, riv)
        torch.testing.assert_close(result, data)

    def test_known_permutation(self) -> None:
        """Known permutation: DiffRoute order is [30, 10, 20] vs topo [10, 20, 30]."""
        topo_order = np.array([10, 20, 30])
        nodes_idx = pd.Series([0, 1, 2], index=[30, 10, 20])  # DFS order
        riv = _make_riv(nodes_idx)
        data = torch.tensor([[[1.0], [2.0], [3.0]]])  # node 10=1, 20=2, 30=3
        result = reorder_to_diffroute(data, topo_order, riv)
        # DiffRoute index 0 = node 30 = topo idx 2 = value 3
        # DiffRoute index 1 = node 10 = topo idx 0 = value 1
        # DiffRoute index 2 = node 20 = topo idx 1 = value 2
        expected = torch.tensor([[[3.0], [1.0], [2.0]]])
        torch.testing.assert_close(result, expected)


class TestReorderToTopo:
    """Tests for reorder_to_topo()."""

    def test_2d(self) -> None:
        topo_order = np.array([10, 20, 30])
        nodes_idx = pd.Series([0, 1, 2], index=[30, 10, 20])  # DFS order
        riv = _make_riv(nodes_idx)
        # 2D: (nodes, time) in DFS order [30, 10, 20]
        data = torch.tensor([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0]])
        result = reorder_to_topo(data, topo_order, riv)
        # topo order [10, 20, 30]: DFS indices [1, 2, 0]
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        torch.testing.assert_close(result, expected)

    def test_3d(self) -> None:
        topo_order = np.array([10, 20, 30])
        nodes_idx = pd.Series([0, 1, 2], index=[30, 10, 20])
        riv = _make_riv(nodes_idx)
        data = torch.tensor([[[3.0], [1.0], [2.0]]])  # (1, nodes, time) DFS order
        result = reorder_to_topo(data, topo_order, riv)
        expected = torch.tensor([[[1.0], [2.0], [3.0]]])
        torch.testing.assert_close(result, expected)

    def test_roundtrip(self) -> None:
        topo_order = np.array([10, 20, 30])
        nodes_idx = pd.Series([0, 1, 2], index=[30, 10, 20])
        riv = _make_riv(nodes_idx)
        original = torch.tensor([[[1.0], [2.0], [3.0]]])  # topo order
        diffroute_order = reorder_to_diffroute(original, topo_order, riv)
        recovered = reorder_to_topo(diffroute_order, topo_order, riv)
        torch.testing.assert_close(recovered, original)


# ============================================================================
# save_results()
# ============================================================================


@pytest.mark.filterwarnings("ignore::UserWarning:zarr")
@pytest.mark.filterwarnings("ignore:.*Unstable.*:FutureWarning")
@pytest.mark.filterwarnings("ignore:.*does not have a Zarr V3 specification")
@pytest.mark.filterwarnings("ignore:.*Consolidated metadata.*:UserWarning")
class TestSaveResults:
    """Tests for save_results()."""

    @pytest.fixture
    def _mock_cfg(self, tmp_path):
        cfg = MagicMock()
        cfg.params.save_path = tmp_path
        cfg.experiment.checkpoint = None
        cfg.experiment.start_time = "2000/01/01"
        cfg.experiment.end_time = "2000/02/01"
        return cfg

    @pytest.fixture
    def _dates(self):
        dates = MagicMock()
        dates.daily_time_range = pd.date_range("2000-01-01", "2000-01-12", freq="D")
        return dates

    def test_creates_zarr(self, _mock_cfg, _dates, tmp_path) -> None:
        ddr_daily = np.random.rand(3, 10).astype(np.float32)
        dr_daily = np.random.rand(3, 10).astype(np.float32)
        obs = np.random.rand(3, 10).astype(np.float32)
        gage_ids = np.array(["g1", "g2", "g3"])
        save_results(_mock_cfg, ddr_daily, dr_daily, obs, gage_ids, _dates)
        assert (tmp_path / "benchmark_results.zarr").exists()

    def test_has_data_vars(self, _mock_cfg, _dates, tmp_path) -> None:
        ddr_daily = np.random.rand(3, 10).astype(np.float32)
        dr_daily = np.random.rand(3, 10).astype(np.float32)
        obs = np.random.rand(3, 10).astype(np.float32)
        gage_ids = np.array(["g1", "g2", "g3"])
        save_results(_mock_cfg, ddr_daily, dr_daily, obs, gage_ids, _dates)
        ds = xr.open_zarr(tmp_path / "benchmark_results.zarr")
        assert "ddr_predictions" in ds
        assert "diffroute_predictions" in ds
        assert "observations" in ds

    def test_shapes_match(self, _mock_cfg, _dates, tmp_path) -> None:
        ddr_daily = np.random.rand(3, 10).astype(np.float32)
        dr_daily = np.random.rand(3, 10).astype(np.float32)
        obs = np.random.rand(3, 10).astype(np.float32)
        gage_ids = np.array(["g1", "g2", "g3"])
        save_results(_mock_cfg, ddr_daily, dr_daily, obs, gage_ids, _dates)
        ds = xr.open_zarr(tmp_path / "benchmark_results.zarr")
        assert ds.ddr_predictions.shape == (3, 10)
        assert ds.diffroute_predictions.shape == (3, 10)
        assert ds.observations.shape == (3, 10)

    def test_attrs_include_version(self, _mock_cfg, _dates, tmp_path) -> None:
        ddr_daily = np.random.rand(3, 10).astype(np.float32)
        dr_daily = np.random.rand(3, 10).astype(np.float32)
        obs = np.random.rand(3, 10).astype(np.float32)
        gage_ids = np.array(["g1", "g2", "g3"])
        save_results(_mock_cfg, ddr_daily, dr_daily, obs, gage_ids, _dates)
        ds = xr.open_zarr(tmp_path / "benchmark_results.zarr")
        assert "ddr_version" in ds.attrs


# ============================================================================
# load_summed_q_prime()
# ============================================================================


@pytest.mark.filterwarnings("ignore::UserWarning:zarr")
@pytest.mark.filterwarnings("ignore:.*does not have a Zarr V3 specification")
@pytest.mark.filterwarnings("ignore:.*Consolidated metadata.*:UserWarning")
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
class TestLoadSummedQPrime:
    """Tests for load_summed_q_prime()."""

    def test_missing_path_returns_none(self) -> None:
        result = load_summed_q_prime(
            "/nonexistent/path.zarr",
            np.array(["g1", "g2"]),
            np.random.rand(2, 10),
            warmup=0,
        )
        assert result is None

    def test_no_common_gages_returns_none(self, tmp_path) -> None:
        # Create a zarr store with different gage IDs
        ds = xr.Dataset(
            {
                "predictions": xr.DataArray(
                    np.random.rand(2, 10).astype(np.float32),
                    dims=["gage_ids", "time"],
                    coords={"gage_ids": ["x1", "x2"]},
                )
            }
        )
        store_path = tmp_path / "sqp.zarr"
        ds.to_zarr(store_path)

        result = load_summed_q_prime(
            str(store_path),
            np.array(["g1", "g2"]),
            np.random.rand(2, 10),
            warmup=0,
        )
        assert result is None

    def test_valid_returns_metrics(self, tmp_path) -> None:
        gage_ids = np.array(["g1", "g2", "g3"])
        preds = np.random.rand(3, 10).astype(np.float32)
        ds = xr.Dataset(
            {
                "predictions": xr.DataArray(
                    preds,
                    dims=["gage_ids", "time"],
                    coords={"gage_ids": gage_ids},
                )
            }
        )
        store_path = tmp_path / "sqp.zarr"
        ds.to_zarr(store_path)

        obs = np.random.rand(3, 10).astype(np.float32)
        result = load_summed_q_prime(str(store_path), gage_ids, obs, warmup=0)
        assert result is not None
        metrics, sqp_preds, mask = result
        assert sqp_preds.shape == (3, 10)
        assert mask.all()

    def test_aligns_gage_ordering(self, tmp_path) -> None:
        """Gages in different order should be aligned correctly."""
        sqp_gages = np.array(["g2", "g1"])
        preds = np.array([[20.0, 20.0], [10.0, 10.0]], dtype=np.float32)
        ds = xr.Dataset(
            {
                "predictions": xr.DataArray(
                    preds,
                    dims=["gage_ids", "time"],
                    coords={"gage_ids": sqp_gages},
                )
            }
        )
        store_path = tmp_path / "sqp.zarr"
        ds.to_zarr(store_path)

        bench_gages = np.array(["g1", "g2"])
        obs = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        result = load_summed_q_prime(str(store_path), bench_gages, obs, warmup=0)
        assert result is not None
        _, sqp_aligned, _ = result
        # g1 was index 1 in sqp (value 10), g2 was index 0 (value 20)
        np.testing.assert_array_almost_equal(sqp_aligned[0], [10.0, 10.0])
        np.testing.assert_array_almost_equal(sqp_aligned[1], [20.0, 20.0])

    def test_uses_shorter_time(self, tmp_path) -> None:
        gage_ids = np.array(["g1"])
        preds = np.random.rand(1, 5).astype(np.float32)  # shorter
        ds = xr.Dataset(
            {
                "predictions": xr.DataArray(
                    preds,
                    dims=["gage_ids", "time"],
                    coords={"gage_ids": gage_ids},
                )
            }
        )
        store_path = tmp_path / "sqp.zarr"
        ds.to_zarr(store_path)

        obs = np.random.rand(1, 10).astype(np.float32)  # longer
        result = load_summed_q_prime(str(store_path), gage_ids, obs, warmup=0)
        assert result is not None
        _, sqp_preds, _ = result
        assert sqp_preds.shape[1] == 5  # uses shorter
