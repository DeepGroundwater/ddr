"""Tests for ddr.scripts_utils — extracted shared script utilities."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ddr.io.functions import downsample
from ddr.scripts_utils import (
    compute_daily_runoff,
    load_checkpoint,
    resolve_learning_rate,
    safe_mean,
    safe_percentile,
    tau_trim_and_downsample,
)


class TestTauTrimAndDownsample:
    """Signed-at-zero tau convention: slice [tau : -(24-tau)], day i ↔ obs day i."""

    def test_shape(self) -> None:
        # 5 gages, 240 hours (10 hourly-days) → slice removes 24 h → 9 days
        hourly = torch.rand(5, 240)
        result = tau_trim_and_downsample(hourly, tau=9)
        assert result.shape == (5, 9)

    def test_tau_zero_is_day_aligned(self) -> None:
        # Day 0 = hours [0, 24). Encode the day index in the data.
        hourly = torch.arange(72, dtype=torch.float32).unsqueeze(0) // 24
        result = tau_trim_and_downsample(hourly, tau=0)
        assert torch.allclose(result[0], torch.tensor([0.0, 1.0]))

    def test_tau_advances_window(self) -> None:
        # With tau=6, pooled day 0 covers hours [6, 30): 18 h of day-0 value
        # (0.0) and 6 h of day-1 value (1.0) → mean 6/24 = 0.25.
        hourly = torch.arange(72, dtype=torch.float32).unsqueeze(0) // 24
        result = tau_trim_and_downsample(hourly, tau=6)
        assert torch.allclose(result[0], torch.tensor([0.25, 1.25]))

    def test_legacy_equivalence_pin(self) -> None:
        # ddrs findings §5i: legacy tau=3 cut hours [16:-8]; the new convention
        # cuts the same hour window at tau=16.
        hourly = torch.rand(3, 240)
        new = tau_trim_and_downsample(hourly, tau=16)
        legacy_sliced = hourly[:, 16:-8]
        legacy = downsample(legacy_sliced, rho=legacy_sliced.shape[1] // 24)
        assert torch.allclose(new, legacy)

    def test_preserves_gradients(self) -> None:
        hourly = torch.rand(2, 120, requires_grad=True)
        result = tau_trim_and_downsample(hourly, tau=9)
        result.sum().backward()
        assert hourly.grad is not None

    def test_rejects_out_of_range_tau(self) -> None:
        hourly = torch.rand(1, 96)
        with pytest.raises(ValueError):
            tau_trim_and_downsample(hourly, tau=24)
        with pytest.raises(ValueError):
            tau_trim_and_downsample(hourly, tau=-1)


class TestComputeDailyRunoff:
    """Numpy wrapper around tau_trim_and_downsample."""

    def test_returns_numpy(self) -> None:
        hourly = torch.rand(2, 120)
        result = compute_daily_runoff(hourly, tau=9)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)

    def test_flat_input_daily_mean(self) -> None:
        hourly = torch.ones(1, 72)
        result = compute_daily_runoff(hourly, tau=9)
        assert np.allclose(result, 1.0, atol=1e-4)


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
