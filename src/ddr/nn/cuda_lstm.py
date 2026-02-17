import logging
from typing import Any

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


class CudaLSTM(torch.nn.Module):
    """An LSTM for time-varying parameter prediction.

    Follows the CudnnLstmModel/LstmModel architecture from generic_deltamodel:
    Linear(nx, hidden) -> ReLU -> LSTM(hidden, hidden) -> Linear(hidden, output_size) -> Sigmoid

    Concatenates meteorological forcings (P, PET, Temp) with static catchment attributes
    at each timestep, producing time-varying parameters in [0,1].
    """

    def __init__(
        self,
        input_var_names: list[str],
        forcing_var_names: list[str],
        learnable_parameters: list[str],
        hidden_size: int,
        num_layers: int,
        dropout: float,
        seed: int,
        device: int | str = "cpu",
    ):
        super().__init__()
        self.num_forcing_vars = len(forcing_var_names)
        self.input_size = len(input_var_names) + self.num_forcing_vars
        self.hidden_size = hidden_size
        self.learnable_parameters = learnable_parameters
        self.output_size = len(self.learnable_parameters)

        torch.manual_seed(seed)

        # Input encoding (generic_deltamodel CudnnLstmModel pattern)
        self.linear_in = torch.nn.Linear(self.input_size, hidden_size, device=device)

        # LSTM operates on encoded hidden_size features
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            device=device,
        )

        # Output dropout (NeuralHydrology pattern: between LSTM output and head)
        self.dropout = torch.nn.Dropout(p=dropout)

        # Output projection
        self.linear_out = torch.nn.Linear(hidden_size, self.output_size, device=device)

        # Weight initialization (KAN pattern: tight output weights for neutral sigmoid start)
        torch.nn.init.kaiming_normal_(self.linear_in.weight, nonlinearity="relu")
        torch.nn.init.zeros_(self.linear_in.bias)
        torch.nn.init.xavier_normal_(self.linear_out.weight, gain=0.1)
        torch.nn.init.zeros_(self.linear_out.bias)

        # Hidden state management (generic_deltamodel pattern)
        # Training: cache_states=False -> hn/cn stay None, zeros created each batch
        # Inference: cache_states=True -> hn/cn carried over (detached) between batches
        self.cache_states: bool = False
        self.hn: torch.Tensor | None = None
        self.cn: torch.Tensor | None = None

    def forward(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass of the LSTM.

        Parameters
        ----------
        forcings : torch.Tensor
            Meteorological forcings, shape (T_daily, N, num_forcing_vars). Passed via kwargs.
        attributes : torch.Tensor
            Normalized static catchment attributes, shape (N, num_attrs). Passed via kwargs.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with each learnable parameter, shape (T_daily, N) in [0, 1].
        """
        forcings: torch.Tensor = kwargs["forcings"]  # [T_daily, N, num_forcing_vars]
        attributes: torch.Tensor = kwargs["attributes"]
        T, N, _ = forcings.shape

        # Concat forcings + static attrs at each timestep
        attrs_expanded = attributes.unsqueeze(0).expand(T, -1, -1)
        _x = torch.cat([forcings, attrs_expanded], dim=-1)  # [T, N, input_size]

        # Input encoding (CudnnLstmModel pattern: Linear -> ReLU)
        _x = F.relu(self.linear_in(_x))  # [T, N, hidden_size]

        # LSTM forward — hn/cn are None during training (zeros created internally)
        h0 = (
            (self.hn.to(_x.device), self.cn.to(_x.device))
            if self.hn is not None and self.cn is not None
            else None
        )
        lstm_out, (hn, cn) = self.lstm(_x, h0)  # [T, N, hidden_size]

        # State management (generic_deltamodel pattern)
        # Training (cache_states=False): states stay None, reset each batch
        # Inference (cache_states=True): states carried forward (detached)
        if self.cache_states:
            self.hn = hn.detach()
            self.cn = cn.detach()

        # Output dropout + projection + sigmoid
        _x = self.linear_out(self.dropout(lstm_out))  # [T, N, 1]
        _x = torch.sigmoid(_x)

        outputs: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(self.learnable_parameters):
            outputs[key] = _x[:, :, idx]
        return outputs
