"""Tests for the CudaLSTM module."""

import pytest
import torch

from ddr.nn.cuda_lstm import CudaLSTM


@pytest.fixture
def model() -> CudaLSTM:
    """Create a CudaLSTM instance for testing."""
    return CudaLSTM(
        input_var_names=["attr1", "attr2", "attr3"],
        forcing_var_names=["P", "PET", "Temp"],
        learnable_parameters=["n", "d_gw"],
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        seed=42,
        device="cpu",
    )


@pytest.fixture
def sample_inputs() -> dict[str, torch.Tensor]:
    """Create sample inputs for the LSTM."""
    T_daily = 10
    N = 5
    return {
        "forcings": torch.rand(T_daily, N, 3),  # 3 forcing vars (P, PET, Temp)
        "attributes": torch.rand(N, 3),  # 3 attrs matching input_var_names
    }


class TestCudaLstmOutput:
    """Test output shape, range, and keys."""

    def test_output_shape(self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test that n and d_gw outputs have shape (T_daily, N)."""
        outputs = model(**sample_inputs)
        T, N, _ = sample_inputs["forcings"].shape
        assert outputs["n"].shape == (T, N), f"n shape {outputs['n'].shape} != ({T}, {N})"
        assert outputs["d_gw"].shape == (T, N), f"d_gw shape {outputs['d_gw'].shape} != ({T}, {N})"

    def test_output_range(self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test that outputs are in [0, 1] (sigmoid)."""
        outputs = model(**sample_inputs)
        for key in ["n", "d_gw"]:
            assert outputs[key].min() >= 0.0, f"{key} min {outputs[key].min()} < 0"
            assert outputs[key].max() <= 1.0, f"{key} max {outputs[key].max()} > 1"

    def test_output_keys(self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test that output dict keys match learnable_parameters."""
        outputs = model(**sample_inputs)
        assert set(outputs.keys()) == {"n", "d_gw"}

    def test_no_nan_or_inf(self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test that outputs contain no NaN or Inf values."""
        outputs = model(**sample_inputs)
        for key, val in outputs.items():
            assert not torch.isnan(val).any(), f"{key} contains NaN"
            assert not torch.isinf(val).any(), f"{key} contains Inf"


class TestCudaLstmGradient:
    """Test gradient flow through the LSTM."""

    def test_gradient_flow(self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]) -> None:
        """Test that gradients flow from loss back through LSTM weights."""
        outputs = model(**sample_inputs)
        loss = sum(v.sum() for v in outputs.values())
        loss.backward()

        # Check that linear_in, lstm, and linear_out all have gradients
        assert model.linear_in.weight.grad is not None, "linear_in should have gradients"
        assert model.linear_out.weight.grad is not None, "linear_out should have gradients"
        for name, param in model.lstm.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"LSTM param {name} should have gradients"


class TestCudaLstmStateManagement:
    """Test cache_states behavior."""

    def test_cache_states_false_no_hidden_carry(
        self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test that with cache_states=False, hidden states stay None between calls."""
        model.cache_states = False
        model(**sample_inputs)
        assert model.hn is None, "hn should be None when cache_states=False"
        assert model.cn is None, "cn should be None when cache_states=False"

    def test_cache_states_true_preserves_hidden(
        self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test that with cache_states=True, hidden states are preserved (detached)."""
        model.cache_states = True
        model(**sample_inputs)
        assert model.hn is not None, "hn should be set when cache_states=True"
        assert model.cn is not None, "cn should be set when cache_states=True"
        assert not model.hn.requires_grad, "cached hn should be detached"
        assert not model.cn.requires_grad, "cached cn should be detached"

    def test_cache_states_true_hidden_changes_between_calls(
        self, model: CudaLSTM, sample_inputs: dict[str, torch.Tensor]
    ) -> None:
        """Test that cached hidden states evolve between forward calls."""
        model.cache_states = True
        model(**sample_inputs)
        assert model.hn is not None
        hn1 = model.hn.clone()

        model(**sample_inputs)
        assert model.hn is not None
        hn2 = model.hn.clone()

        assert not torch.allclose(hn1, hn2), "Hidden state should change between sequential calls"


class TestCudaLstmInit:
    """Test initialization edge cases."""

    def test_input_size_includes_forcings(self) -> None:
        """Test that input_size = len(input_var_names) + len(forcing_var_names)."""
        m = CudaLSTM(
            input_var_names=["a", "b"],
            forcing_var_names=["P", "PET", "Temp"],
            learnable_parameters=["n", "d_gw"],
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            seed=0,
            device="cpu",
        )
        assert m.input_size == 5  # 2 attrs + 3 forcings

    def test_multi_layer_lstm(self) -> None:
        """Test that multi-layer LSTM with dropout works."""
        m = CudaLSTM(
            input_var_names=["a"],
            forcing_var_names=["P", "PET"],
            learnable_parameters=["n", "d_gw"],
            hidden_size=16,
            num_layers=3,
            dropout=0.5,
            seed=0,
            device="cpu",
        )
        # Should run without error
        outputs = m(forcings=torch.rand(5, 4, 2), attributes=torch.rand(4, 1))
        assert outputs["n"].shape == (5, 4)
        assert outputs["d_gw"].shape == (5, 4)

    def test_output_dropout_layer(self) -> None:
        """Test that output dropout layer exists with correct probability."""
        m = CudaLSTM(
            input_var_names=["a"],
            forcing_var_names=["P"],
            learnable_parameters=["n", "d_gw"],
            hidden_size=16,
            num_layers=1,
            dropout=0.5,
            seed=0,
            device="cpu",
        )
        assert isinstance(m.dropout, torch.nn.Dropout)
        assert m.dropout.p == 0.5
        # LSTM's built-in dropout is unused (always 0)
        assert m.lstm.dropout == 0.0
