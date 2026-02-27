import logging
from typing import Any

import torch
import torch.nn.functional as F
from kan import KAN

log = logging.getLogger(__name__)


class kan(torch.nn.Module):
    """A Kolmogorov Arnold Neural Network (KAN)"""

    def __init__(
        self,
        input_var_names: list[str],
        learnable_parameters: list[str],
        hidden_size: int,
        num_hidden_layers: int,
        grid: int,
        k: int,
        seed: int,
        device: int | str = "cpu",
        gate_parameters: list[str] | None = None,
        off_parameters: list[str] | None = None,
        use_graph_context: bool = False,
    ):
        super().__init__()
        self.input_size = len(input_var_names)
        self.hidden_size = hidden_size
        self.learnable_parameters = learnable_parameters
        self.output_size = len(self.learnable_parameters)

        # When graph context is enabled, the input is augmented with neighbor-aggregated
        # attributes (original D + aggregated D), doubling the effective input dimension.
        effective_input_size = self.input_size * 2 if use_graph_context else self.input_size
        self.input = torch.nn.Linear(effective_input_size, self.hidden_size, device=device)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(
                KAN(
                    [self.hidden_size, self.hidden_size],
                    k=k,
                    grid=grid,
                    seed=seed,
                    device=device,
                )
            )
        self.output = torch.nn.Linear(self.hidden_size, self.output_size, bias=True, device=device)

        torch.nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu")
        torch.nn.init.xavier_normal_(self.output.weight, gain=0.1)
        torch.nn.init.zeros_(self.input.bias)
        torch.nn.init.zeros_(self.output.bias)

        # Initialize gate parameter biases to +1.0 so sigmoid(1) ≈ 0.73 → gate starts ON.
        # Starting ON provides an immediate error signal that allows the model to learn
        # where the gated process hurts and turn it off selectively (strong gradient signal).
        # Starting OFF gives no gradient signal to turn ON (chicken-and-egg problem).
        if gate_parameters:
            with torch.no_grad():
                for param_name in gate_parameters:
                    idx = self.learnable_parameters.index(param_name)
                    self.output.bias[idx] = 1.0

        # Initialize off_parameters with negative bias so sigmoid starts near 0 (OFF).
        # Unlike gate_parameters, these remain continuous (no binary STE).
        if off_parameters:
            with torch.no_grad():
                for param_name in off_parameters:
                    idx = self.learnable_parameters.index(param_name)
                    self.output.bias[idx] = -2.0

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass of the neural network"""
        _x: torch.Tensor = kwargs["inputs"]
        outputs = {}
        _x = self.input(_x)
        for layer in self.layers:
            _x = layer(_x)
        _x = self.output(_x)
        _x = F.sigmoid(_x)
        x_transpose = _x.transpose(0, 1)
        for idx, key in enumerate(self.learnable_parameters):
            outputs[key] = x_transpose[idx]
        return outputs
