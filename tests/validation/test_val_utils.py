"""Tests for ddr.validation.utils — save_state, log_metrics."""

from pathlib import Path

import numpy as np
import torch

from ddr.validation.utils import log_metrics, save_state


class TestSaveState:
    """Tests for save_state()."""

    def test_save_state_creates_file(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        kan_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        generator = torch.Generator()

        save_state(
            epoch=1,
            generator=generator,
            mini_batch=-1,
            mlp=model,
            kan_optimizer=kan_optimizer,
            name="test",
            saved_model_path=tmp_path,
        )

        files = list(tmp_path.glob("*.pt"))
        assert len(files) == 1

    def test_save_state_contents(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        kan_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        generator = torch.Generator()

        save_state(
            epoch=5,
            generator=generator,
            mini_batch=-1,
            mlp=model,
            kan_optimizer=kan_optimizer,
            name="test",
            saved_model_path=tmp_path,
        )

        files = list(tmp_path.glob("*.pt"))
        state = torch.load(files[0], weights_only=False)
        assert "model_state_dict" in state
        assert "kan_optimizer_state_dict" in state
        assert "epoch" in state
        assert state["epoch"] == 6  # mini_batch=-1 → epoch+1

    def test_save_state_dual_optimizer(self, tmp_path: Path) -> None:
        model = torch.nn.Linear(3, 2)
        lstm = torch.nn.Linear(5, 2)
        kan_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        lstm_optimizer = torch.optim.Adadelta(lstm.parameters())
        generator = torch.Generator()

        save_state(
            epoch=1,
            generator=generator,
            mini_batch=0,
            mlp=model,
            kan_optimizer=kan_optimizer,
            name="test",
            saved_model_path=tmp_path,
            leakance_nn=lstm,
            lstm_optimizer=lstm_optimizer,
        )

        files = list(tmp_path.glob("*.pt"))
        state = torch.load(files[0], weights_only=False)
        assert "kan_optimizer_state_dict" in state
        assert "lstm_optimizer_state_dict" in state
        assert "leakance_nn_state_dict" in state


class TestLogMetrics:
    """Tests for log_metrics()."""

    def test_log_metrics_no_crash(self) -> None:
        nse = np.array([0.8, 0.9, 0.7])
        rmse = np.array([0.1, 0.05, 0.2])
        kge = np.array([0.85, 0.92, 0.78])

        # Should not raise
        log_metrics(nse=nse, rmse=rmse, kge=kge, epoch=1, mini_batch=0)
        log_metrics(nse=nse, rmse=rmse, kge=kge)  # Without epoch/mini_batch
