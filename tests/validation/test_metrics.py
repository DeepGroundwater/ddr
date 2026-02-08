"""Tests for ddr.validation.metrics.Metrics."""

import json

import numpy as np
import pytest

from ddr.validation.metrics import Metrics


class TestMetricsPerfectPrediction:
    """Test metrics with perfect predictions."""

    def test_perfect_prediction(self) -> None:
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = target.copy()
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.nse[0], 1.0)
        assert np.isclose(m.kge[0], 1.0)
        assert np.isclose(m.rmse[0], 0.0)
        assert np.isclose(m.bias[0], 0.0)
        assert np.isclose(m.corr[0], 1.0)

    def test_constant_prediction_nse_zero(self) -> None:
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = np.full_like(target, target.mean())
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.nse[0], 0.0, atol=1e-10)


class TestMetricsKnownValues:
    """Test individual metrics with known computed values."""

    def test_rmse_known_values(self) -> None:
        pred = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[4.0, 5.0, 6.0]])
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.rmse[0], 3.0)

    def test_mae_known_values(self) -> None:
        pred = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[4.0, 5.0, 6.0]])
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.mae[0], 3.0)

    def test_bias_known_values(self) -> None:
        pred = np.array([[3.0, 3.0, 3.0]])
        target = np.array([[1.0, 1.0, 1.0]])
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.bias[0], 2.0)

    def test_pbias_known_values(self) -> None:
        pred = np.array([[120.0, 120.0]])
        target = np.array([[100.0, 100.0]])
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.pbias[0], 20.0)

    def test_kge_known_values(self) -> None:
        # Manual KGE: r=1, alpha=std_p/std_t, beta=mean_p/mean_t
        pred = np.array([[2.0, 4.0, 6.0, 8.0, 10.0]])
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        m = Metrics(pred=pred, target=target)

        # r = 1.0 (perfect linear), alpha = 2.0, beta = 2.0
        # KGE = 1 - sqrt((1-1)^2 + (2-1)^2 + (2-1)^2) = 1 - sqrt(2) ≈ -0.4142
        expected_kge = 1 - np.sqrt(2)
        assert np.isclose(m.kge[0], expected_kge, atol=1e-4)

    def test_nse_equals_r2(self) -> None:
        pred = np.array([[1.0, 3.0, 5.0, 2.0, 4.0]])
        target = np.array([[1.5, 2.5, 4.5, 3.0, 3.5]])
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.nse[0], m.r2[0])


class TestMetricsFDC:
    """Test flow duration curve metrics."""

    def test_fdc_rmse_sorted_distributions(self) -> None:
        pred = np.sort(np.random.default_rng(42).uniform(0.1, 10, (1, 200)))[:, ::-1].copy()
        target = np.sort(np.random.default_rng(43).uniform(0.1, 10, (1, 200)))[:, ::-1].copy()
        m = Metrics(pred=pred, target=target)

        assert m.fdc_rmse.shape == (1,)
        assert np.isfinite(m.fdc_rmse[0])


class TestMetricsSpearman:
    """Test Spearman correlation."""

    def test_spearman_monotonic(self) -> None:
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pred = np.array([[10.0, 20.0, 30.0, 40.0, 50.0]])  # Monotonic transform
        m = Metrics(pred=pred, target=target)

        assert np.isclose(m.corr_spearman[0], 1.0)


class TestMetricsNaN:
    """Test NaN handling."""

    def test_nan_in_prediction_raises(self) -> None:
        pred = np.array([[1.0, np.nan, 3.0]])
        target = np.array([[1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="pred contains NaN"):
            Metrics(pred=pred, target=target)

    def test_nan_in_target_handled(self) -> None:
        pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        target = np.array([[1.0, np.nan, 3.0, 4.0, 5.0]])
        m = Metrics(pred=pred, target=target)

        # Metrics should be computed on non-NaN subset
        assert np.isfinite(m.corr[0])

    def test_all_nan_target_row(self) -> None:
        pred = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        target = np.array([[np.nan, np.nan, np.nan], [4.0, 5.0, 6.0]])
        m = Metrics(pred=pred, target=target)

        # First row metrics should be NaN (no valid target data)
        assert np.isnan(m.nse[0])
        # Second row should be fine
        assert np.isclose(m.nse[1], 1.0)


class TestMetricsInputShape:
    """Test input shape handling."""

    def test_1d_input_expansion(self) -> None:
        pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = Metrics(pred=pred, target=target)

        assert m.ngrid == 1
        assert m.nt == 5

    def test_multi_grid(self) -> None:
        rng = np.random.default_rng(42)
        pred = rng.uniform(0.1, 10, (3, 100))
        target = rng.uniform(0.1, 10, (3, 100))
        m = Metrics(pred=pred, target=target)

        assert m.ngrid == 3
        assert m.nse.shape == (3,)
        assert m.rmse.shape == (3,)
        assert m.kge.shape == (3,)


class TestMetricsSerialization:
    """Test model serialization."""

    def test_model_dump_json(self) -> None:
        pred = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        m = Metrics(pred=pred, target=target)

        result = m.model_dump_json()
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "rmse" in parsed
        assert "nse" in parsed


class TestMetricsFlowSplit:
    """Test low/mid/high flow split."""

    def test_low_mid_high_flow_split(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.uniform(0.1, 100, (1, 1000))
        pred = data.copy()
        target = data.copy()
        m = Metrics(pred=pred, target=target)

        assert np.isfinite(m.rmse_low[0])
        assert np.isfinite(m.rmse_mid[0])
        assert np.isfinite(m.rmse_high[0])
        # Perfect prediction → all rmse should be 0
        assert np.isclose(m.rmse_low[0], 0.0)
        assert np.isclose(m.rmse_mid[0], 0.0)
        assert np.isclose(m.rmse_high[0], 0.0)


class TestMetricsSingleTimestep:
    """Test edge case with single timestep."""

    def test_single_timestep_correlation(self) -> None:
        pred = np.array([[5.0]])
        target = np.array([[5.0]])
        m = Metrics(pred=pred, target=target)

        # Only 1 timestep → correlation needs >1 points
        assert np.isnan(m.corr[0])
        assert np.isnan(m.nse[0])
