"""Integration test that exercises the full training pipeline.

Runs 2 epochs of real training on 10 small-drainage gages with real HPC data.
Verifies: no NaN/Inf, loss stays bounded, gradients flow, parameters update.

Skipped by default. Run with: uv run pytest -m integration tests/integration/ -v
"""

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch
from torch.utils.data import DataLoader, RandomSampler

from ddr import CudaLSTM, ddr_functions, dmc, forcings_reader, kan, streamflow
from ddr.geodatazoo.merit import Merit
from ddr.routing.utils import select_columns
from ddr.scripts_utils import resolve_learning_rate
from ddr.validation import Config, hydrograph_loss

pytestmark = pytest.mark.integration


@dataclass
class StepMetrics:
    """Metrics captured for a single training step."""

    epoch: int
    step: int
    loss: float
    n_mean: float
    n_std: float
    n_min: float
    n_max: float
    gate_mean: float
    gate_on_frac: float
    kan_grad_norm: float
    lstm_grad_norm: float
    has_nan: bool
    has_inf: bool


@dataclass
class TrainingMetrics:
    """Aggregated metrics across all training steps."""

    steps: list[StepMetrics] = field(default_factory=list)

    @property
    def losses(self) -> list[float]:
        return [s.loss for s in self.steps]

    def epoch_avg_loss(self, epoch: int) -> float:
        epoch_losses = [s.loss for s in self.steps if s.epoch == epoch]
        return sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("inf")


def _compute_grad_norm(model: torch.nn.Module) -> Any:
    """L2 norm of all gradients in the model."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def _run_training_loop(
    cfg: Config,
    dataset: Merit,
    nn: kan,
    lstm_nn: CudaLSTM,
    routing_model: dmc,
    flow: streamflow,
    forcings_nn: forcings_reader,
) -> TrainingMetrics:
    """Simplified training loop that returns metrics instead of saving checkpoints."""
    metrics = TrainingMetrics()

    data_generator = torch.Generator()
    data_generator.manual_seed(cfg.seed)

    lr = resolve_learning_rate(cfg.experiment.learning_rate, 1)
    kan_optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    lstm_optimizer = torch.optim.Adam(params=lstm_nn.parameters(), lr=lr)

    sampler = RandomSampler(data_source=dataset, generator=data_generator)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.experiment.batch_size,
        num_workers=0,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=True,
    )

    for epoch in range(1, cfg.experiment.epochs + 1):
        if epoch in cfg.experiment.learning_rate:
            lr = cfg.experiment.learning_rate[epoch]
            for param_group in kan_optimizer.param_groups:
                param_group["lr"] = lr
            for param_group in lstm_optimizer.param_groups:
                param_group["lr"] = lr

        for i, routing_dataclass in enumerate(dataloader, start=0):
            routing_model.set_progress_info(epoch=epoch, mini_batch=i)
            kan_optimizer.zero_grad()
            lstm_optimizer.zero_grad()

            streamflow_predictions = flow(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )
            attr_names = routing_dataclass.attribute_names
            normalized_attrs = routing_dataclass.normalized_spatial_attributes.to(cfg.device)
            kan_attrs = select_columns(normalized_attrs, list(cfg.kan.input_var_names), attr_names)
            spatial_params = nn(inputs=kan_attrs)
            forcing_data = forcings_nn(
                routing_dataclass=routing_dataclass, device=cfg.device, dtype=torch.float32
            )
            lstm_attrs = select_columns(normalized_attrs, list(cfg.cuda_lstm.input_var_names), attr_names)
            lstm_params = lstm_nn(forcings=forcing_data, attributes=lstm_attrs)

            dmc_output = routing_model(
                routing_dataclass=routing_dataclass,
                spatial_parameters=spatial_params,
                streamflow=streamflow_predictions,
                lstm_params=lstm_params,
            )

            num_days = len(dmc_output["runoff"][0][13 : (-11 + cfg.params.tau)]) // 24
            daily_runoff = ddr_functions.downsample(
                dmc_output["runoff"][:, 13 : (-11 + cfg.params.tau)],
                rho=num_days,
            )

            nan_mask = routing_dataclass.observations.isnull().any(dim="time")
            np_nan_mask = nan_mask.streamflow.values

            filtered_ds = routing_dataclass.observations.where(~nan_mask, drop=True)
            filtered_observations = torch.tensor(
                filtered_ds.streamflow.values, device=cfg.device, dtype=torch.float32
            )[:, 1:-1]

            filtered_predictions = daily_runoff[~np_nan_mask]

            pred = filtered_predictions[:, cfg.experiment.warmup :]
            target = filtered_observations[:, cfg.experiment.warmup :]
            loss_cfg = cfg.experiment.loss
            loss = hydrograph_loss(
                pred=pred,
                target=target,
                overall_weight=loss_cfg.overall_weight,
                peak_weight=loss_cfg.peak_weight,
                baseflow_weight=loss_cfg.baseflow_weight,
                timing_weight=loss_cfg.timing_weight,
                peak_percentile=loss_cfg.peak_percentile,
                baseflow_percentile=loss_cfg.baseflow_percentile,
                eps=loss_cfg.eps,
            )

            loss.backward()
            kan_optimizer.step()
            lstm_optimizer.step()

            # Capture metrics
            n_vals = routing_model.n.detach().cpu()
            gate_raw = spatial_params.get("leakance_gate", torch.tensor([0.5]))
            gate_raw = gate_raw.detach().cpu()

            step = StepMetrics(
                epoch=epoch,
                step=i,
                loss=loss.item(),
                n_mean=n_vals.mean().item(),
                n_std=n_vals.std().item(),
                n_min=n_vals.min().item(),
                n_max=n_vals.max().item(),
                gate_mean=gate_raw.mean().item(),
                gate_on_frac=(gate_raw > 0.5).float().mean().item(),
                kan_grad_norm=_compute_grad_norm(nn),
                lstm_grad_norm=_compute_grad_norm(lstm_nn),
                has_nan=torch.isnan(loss).item(),
                has_inf=torch.isinf(loss).item(),
            )
            metrics.steps.append(step)

            # Clean up batch state
            del streamflow_predictions, spatial_params, dmc_output, daily_runoff
            del loss, filtered_predictions, filtered_observations
            del forcing_data, lstm_params
            routing_model.clear_batch_state()

    return metrics


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""

    @pytest.fixture(scope="class")
    def training_result(
        self,
        integration_config: Config,
        integration_dataset: Merit,
        integration_models: tuple[kan, CudaLSTM, dmc, streamflow, forcings_reader],
    ) -> tuple[TrainingMetrics, dict[str, torch.Tensor], kan]:
        """Run training once, return (metrics, initial_kan_params, trained_kan)."""
        nn, lstm_nn, routing_model, flow, forcings_nn = integration_models

        # Snapshot KAN weights before training
        initial_kan_params = {name: param.detach().clone() for name, param in nn.named_parameters()}

        result = _run_training_loop(
            cfg=integration_config,
            dataset=integration_dataset,
            nn=nn,
            lstm_nn=lstm_nn,
            routing_model=routing_model,
            flow=flow,
            forcings_nn=forcings_nn,
        )
        return result, initial_kan_params, nn

    def test_no_nan_or_inf(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """Loss is finite at every training step."""
        metrics, _, _ = training_result
        for step in metrics.steps:
            assert not step.has_nan, f"NaN loss at epoch {step.epoch}, step {step.step}"
            assert not step.has_inf, f"Inf loss at epoch {step.epoch}, step {step.step}"

    def test_loss_no_blowup(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """Loss does not explode across epochs (epoch 2 avg < 10x epoch 1 avg)."""
        metrics, _, _ = training_result
        avg_1 = metrics.epoch_avg_loss(1)
        avg_2 = metrics.epoch_avg_loss(2)
        assert avg_2 < avg_1 * 10, (
            f"Loss blew up: epoch 1 avg={avg_1:.4f}, epoch 2 avg={avg_2:.4f} (>{avg_1 * 10:.4f})"
        )

    def test_loss_is_finite_and_reasonable(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """All step losses are below a reasonable upper bound."""
        metrics, _, _ = training_result
        for step in metrics.steps:
            assert step.loss < 1e6, (
                f"Unreasonable loss {step.loss:.2f} at epoch {step.epoch}, step {step.step}"
            )

    def test_mannings_n_differentiates(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """Manning's n has spatial variation and stays in physical bounds."""
        metrics, _, _ = training_result
        last_step = metrics.steps[-1]
        assert last_step.n_std > 1e-4, f"No spatial variation in Manning's n (std={last_step.n_std:.6f})"
        assert last_step.n_min >= 0.015, f"Manning's n below lower bound: {last_step.n_min:.4f}"
        assert last_step.n_max <= 0.25, f"Manning's n above upper bound: {last_step.n_max:.4f}"

    def test_gradients_nonzero(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """Both KAN and LSTM receive non-trivial gradients."""
        metrics, _, _ = training_result
        # Check any step has nonzero gradients
        max_kan_grad = max(s.kan_grad_norm for s in metrics.steps)
        max_lstm_grad = max(s.lstm_grad_norm for s in metrics.steps)
        assert max_kan_grad > 1e-8, f"KAN gradient norm too small: {max_kan_grad:.2e}"
        assert max_lstm_grad > 1e-8, f"LSTM gradient norm too small: {max_lstm_grad:.2e}"

    def test_kan_parameters_change(self, training_result: tuple[TrainingMetrics, dict, kan]) -> None:
        """KAN weights differ from their initial values after training."""
        _, initial_params, trained_nn = training_result
        any_changed = False
        for name, param in trained_nn.named_parameters():
            if name in initial_params and not torch.equal(param.detach(), initial_params[name]):
                any_changed = True
                break
        assert any_changed, "No KAN parameters changed during training"
