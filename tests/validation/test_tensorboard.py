"""Tests for the TensorBoard logging wrapper."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ddr.validation.tensorboard import _TB_AVAILABLE, TBLogger, _NoOpTBLogger, create_tb_logger


class TestNoOpLogger:
    """All methods on _NoOpTBLogger must be callable without error."""

    def test_all_methods_are_no_ops(self):
        logger = _NoOpTBLogger()
        logger.log_loss(0.5, global_step=0)
        logger.log_learning_rate(1e-3, global_step=0)
        logger.log_grad_norm(1.2, global_step=0)
        logger.log_metrics(np.array([0.8]), np.array([0.1]), np.array([0.7]), global_step=0)
        logger.log_routing_params(n_vals=MagicMock(), global_step=0)
        logger.log_benchmark_metrics(metrics=MagicMock(), model_name="ddr", global_step=0)
        logger.close()


class TestCreateTbLogger:
    def test_disabled_returns_noop(self, tmp_path):
        logger = create_tb_logger(enabled=False, log_dir=tmp_path / "tb")
        assert isinstance(logger, _NoOpTBLogger)

    @pytest.mark.skipif(not _TB_AVAILABLE, reason="tensorboard not installed")
    def test_enabled_and_installed_returns_real(self, tmp_path):
        logger = create_tb_logger(enabled=True, log_dir=tmp_path / "tb")
        assert isinstance(logger, TBLogger)
        logger.close()

    def test_enabled_but_missing_returns_noop_with_warning(self, tmp_path):
        with patch("ddr.validation.tensorboard._TB_AVAILABLE", False):
            logger = create_tb_logger(enabled=True, log_dir=tmp_path / "tb")
            assert isinstance(logger, _NoOpTBLogger)


@pytest.mark.skipif(not _TB_AVAILABLE, reason="tensorboard not installed")
class TestTBLoggerInterval:
    def test_log_interval_filters_steps(self, tmp_path):
        logger = TBLogger(log_dir=tmp_path / "tb", log_interval=5)
        logger._writer = MagicMock()

        logger.log_loss(0.5, global_step=0)
        assert logger._writer.add_scalar.call_count == 1

        logger._writer.reset_mock()
        logger.log_loss(0.5, global_step=3)
        assert logger._writer.add_scalar.call_count == 0

        logger._writer.reset_mock()
        logger.log_loss(0.5, global_step=5)
        assert logger._writer.add_scalar.call_count == 1

    def test_interval_1_logs_every_step(self, tmp_path):
        logger = TBLogger(log_dir=tmp_path / "tb", log_interval=1)
        logger._writer = MagicMock()

        for step in range(5):
            logger.log_loss(0.5, global_step=step)
        assert logger._writer.add_scalar.call_count == 5


@pytest.mark.skipif(not _TB_AVAILABLE, reason="tensorboard not installed")
class TestBenchmarkLogging:
    def test_log_benchmark_metrics_writes_expected_tags(self, tmp_path):
        logger = TBLogger(log_dir=tmp_path / "tb", log_interval=1)
        logger._writer = MagicMock()

        metrics = MagicMock()
        metrics.nse = np.array([0.8, 0.9, np.nan])
        metrics.kge = np.array([0.7, 0.85, 0.6])
        metrics.rmse = np.array([0.1, 0.2, 0.15])
        metrics.bias = np.array([0.01, -0.02, 0.005])
        metrics.fhv = np.array([5.0, 3.0, np.inf])
        metrics.flv = np.array([-2.0, -1.0, -3.0])

        logger.log_benchmark_metrics(metrics, model_name="ddr", global_step=0)

        tags = [call.args[0] for call in logger._writer.add_scalar.call_args_list]
        assert "benchmark/ddr/nse_mean" in tags
        assert "benchmark/ddr/nse_median" in tags
        assert "benchmark/ddr/kge_mean" in tags
        assert "benchmark/ddr/rmse_mean" in tags
        assert "benchmark/ddr/bias_mean" in tags
        assert "benchmark/ddr/fhv_mean" in tags
        assert "benchmark/ddr/flv_mean" in tags

    def test_all_nan_metric_skipped(self, tmp_path):
        logger = TBLogger(log_dir=tmp_path / "tb", log_interval=1)
        logger._writer = MagicMock()

        metrics = MagicMock()
        metrics.nse = np.array([np.nan, np.nan])
        metrics.kge = np.array([0.7])
        metrics.rmse = np.array([0.1])
        metrics.bias = np.array([0.01])
        metrics.fhv = np.array([5.0])
        metrics.flv = np.array([-2.0])

        logger.log_benchmark_metrics(metrics, model_name="test")

        tags = [call.args[0] for call in logger._writer.add_scalar.call_args_list]
        assert "benchmark/test/nse_mean" not in tags
        assert "benchmark/test/kge_mean" in tags
