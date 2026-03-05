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
        output_grid_bounds: bool = False,
        bias_cfg: Any | None = None,
    ):
        super().__init__()
        self.input_size = len(input_var_names)
        self.hidden_size = hidden_size
        self.learnable_parameters = learnable_parameters
        self.output_size = len(self.learnable_parameters)
        self.output_grid_bounds = output_grid_bounds

        self.input = torch.nn.Linear(self.input_size, self.hidden_size, device=device)
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

        if output_grid_bounds:
            bounds_grid = bias_cfg.bounds_grid if bias_cfg else 8
            bounds_k = bias_cfg.bounds_k if bias_cfg else 3
            self.bounds_head = KAN(
                [self.hidden_size, 2],
                grid=bounds_grid,
                k=bounds_k,
                seed=seed,
                device=device,
            )

        torch.nn.init.kaiming_normal_(self.input.weight, nonlinearity="relu")
        torch.nn.init.xavier_normal_(self.output.weight, gain=0.1)
        torch.nn.init.zeros_(self.input.bias)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass of the neural network"""
        _x: torch.Tensor = kwargs["inputs"]
        outputs = {}
        _x = self.input(_x)
        for layer in self.layers:
            _x = layer(_x)

        # Routing parameters
        _params = self.output(_x)
        _params = F.sigmoid(_params)
        x_transpose = _params.transpose(0, 1)
        for idx, key in enumerate(self.learnable_parameters):
            outputs[key] = x_transpose[idx]

        # Grid bounds for φ-KAN normalization
        if self.output_grid_bounds:
            bounds_raw = self.bounds_head(_x)
            bounds = F.softplus(bounds_raw)  # enforce positive
            grid_bounds = torch.stack([bounds[:, 0], bounds[:, 0] + bounds[:, 1]], dim=1)
            outputs["grid_bounds"] = grid_bounds

        return outputs
