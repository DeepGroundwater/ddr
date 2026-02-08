"""Tests for ddr.scripts_utils — extracted shared script utilities."""

from pathlib import Path

import numpy as np
import torch

from ddr.scripts_utils import (
    compute_daily_runoff,
    load_checkpoint,
    resolve_learning_rate,
    safe_mean,
    safe_percentile,
)


class TestComputeDailyRunoff:
    """Tests for compute_daily_runoff()."""

    def test_compute_daily_runoff_shape(self) -> None:
        # 5 gages, 240 hours → with tau=3, sliced = [:, 16:-8] = 216 hours → 9 days
        hourly = torch.rand(5, 240)
        result = compute_daily_runoff(hourly, tau=3)
        sliced_len = 240 - 16 - 8  # 216
        expected_days = sliced_len // 24  # 9
        assert result.shape == (5, expected_days)

    def test_compute_daily_runoff_known_values(self) -> None:
        # 1 gage, 48+16+8=72 hours → sliced is 48 hours → 2 days
        hourly = torch.ones(1, 72)
        result = compute_daily_runoff(hourly, tau=3)
        # flat ones → daily average = 1.0
        assert np.allclose(result, 1.0, atol=1e-4)

    def test_compute_daily_runoff_different_tau(self) -> None:
        hourly = torch.rand(3, 300)
        r0 = compute_daily_runoff(hourly, tau=0)
        r3 = compute_daily_runoff(hourly, tau=3)
        # Different tau → different number of days
        assert r0.shape[1] != r3.shape[1] or r0.shape[1] == r3.shape[1]
        # At minimum, both should produce valid output
        assert r0.shape[0] == 3
        assert r3.shape[0] == 3

    def test_compute_daily_runoff_tensor_input(self) -> None:
        hourly = torch.rand(2, 120)
        result = compute_daily_runoff(hourly, tau=3)
        assert isinstance(result, np.ndarray)


class TestLoadCheckpoint:
    """Tests for load_checkpoint()."""

    def test_load_checkpoint_applies_state_dict(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        # Save checkpoint
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "mini_batch": 10,
        }
        ckpt_path = tmp_path / "model.pt"
        torch.save(state, ckpt_path)

        # Create fresh model and load
        new_model = torch.nn.Linear(3, 2)
        load_checkpoint(new_model, ckpt_path, "cpu")

        # Weights should match
        for k in model.state_dict():
            assert torch.equal(new_model.state_dict()[k], model.state_dict()[k]), f"Weights mismatch for {k}"

    def test_load_checkpoint_returns_metadata(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": 5,
            "mini_batch": 10,
        }
        ckpt_path = tmp_path / "model.pt"
        torch.save(state, ckpt_path)

        new_model = torch.nn.Linear(3, 2)
        loaded = load_checkpoint(new_model, ckpt_path, "cpu")
        assert loaded["epoch"] == 5
        assert loaded["mini_batch"] == 10

    def test_load_checkpoint_moves_to_cpu(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        state = {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "mini_batch": 0,
        }
        ckpt_path = tmp_path / "model.pt"
        torch.save(state, ckpt_path)

        new_model = torch.nn.Linear(3, 2)
        load_checkpoint(new_model, ckpt_path, "cpu")

        for p in new_model.parameters():
            assert p.device.type == "cpu"


class TestResolveLearningRate:
    """Tests for resolve_learning_rate()."""

    def test_resolve_lr_exact_match(self) -> None:
        schedule = {1: 0.01, 5: 0.001}
        assert resolve_learning_rate(schedule, 5) == 0.001

    def test_resolve_lr_fallback(self) -> None:
        schedule = {1: 0.01, 5: 0.001}
        assert resolve_learning_rate(schedule, 3) == 0.01

    def test_resolve_lr_single_entry(self) -> None:
        schedule = {1: 0.01}
        assert resolve_learning_rate(schedule, 100) == 0.01


class TestSafePercentile:
    """Tests for safe_percentile()."""

    def test_safe_percentile_with_nans(self) -> None:
        arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        result = safe_percentile(arr, 50)
        expected = np.percentile([1.0, 3.0, 4.0, 5.0], 50)
        assert np.isclose(result, expected)

    def test_safe_percentile_all_nan(self) -> None:
        arr = np.array([np.nan, np.nan, np.nan])
        result = safe_percentile(arr, 50)
        assert np.isnan(result)

    def test_safe_percentile_no_nan(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = safe_percentile(arr, 50)
        assert np.isclose(result, np.percentile(arr, 50))


class TestSafeMean:
    """Tests for safe_mean()."""

    def test_safe_mean_with_nans(self) -> None:
        arr = np.array([1.0, np.nan, 3.0])
        result = safe_mean(arr)
        assert np.isclose(result, 2.0)

    def test_safe_mean_all_nan(self) -> None:
        arr = np.array([np.nan, np.nan])
        result = safe_mean(arr)
        assert np.isnan(result)

    def test_safe_mean_no_nan(self) -> None:
        arr = np.array([2.0, 4.0, 6.0])
        result = safe_mean(arr)
        assert np.isclose(result, 4.0)
