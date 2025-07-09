import logging

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
        output_size: int,
        num_hidden_layers: int,
        grid: int,
        k: int,
        seed: int,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_size = len(input_var_names)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learnable_parameters = learnable_parameters

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
        self.output = torch.nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)
        torch.nn.init.kaiming_normal_(self.input.weight)
        torch.nn.init.kaiming_normal_(self.output.weight)
        torch.nn.init.zeros_(self.input.bias)

    def forward(self, *args, **kwargs) -> dict[str, torch.Tensor]:
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
