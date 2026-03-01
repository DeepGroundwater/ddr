"""Physics-grounded GNN node processor for Muskingum-Cunge routing.

Implements a latent node state that evolves alongside the MC physics solve.
At each routing timestep, the MCNodeProcessor updates per-reach embeddings
using all four Muskingum-Cunge coefficient terms as physics-derived message
channels.  The ParamDecoder then maps the evolving embedding back to physical
routing parameters (Manning's n, channel geometry).

Architecture (encoder-processor-decoder):
  Encoder:  KAN(attrs) → h^0  [N, D_h]
  Processor: for t in 1..T:
    params^{t-1} = ParamDecoder(h^{t-1})
    Q^t = sparse_solve(I - C1·N, b)         # physics unchanged
    h^t = MCNodeProcessor.step(h^{t-1}, ...)  # GNN update
  Decoder: output = Q (already physical state)

The GNN step is grounded in the MC row equation:

  Q_{i,t+1} = C1_i · (N@Q_{t+1})   (implicit upstream)
             + C2_i · (N@Q_t)       (explicit upstream)
             + C3_i · Q_{i,t}       (attenuation / memory)
             + C4_i · q'_{i,t}      (lateral forcing)

All four terms are passed as separate physics channels to the NodeMLP,
making the processor interpretable — the network learns which MC component
most informs how Manning's n should adapt at each reach.
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class MCNodeProcessor(nn.Module):
    """Physics-grounded GNN node embedding processor.

    Runs alongside the MC physics solve at each routing timestep.
    Updates per-reach latent embeddings h^t using the four MC coefficient
    terms as physics message channels plus upstream embedding aggregation.

    Parameters
    ----------
    d_hidden : int
        Latent embedding dimension D_h.  Must match KAN hidden_size.
    """

    def __init__(self, d_hidden: int) -> None:
        super().__init__()
        # NodeMLP: [2*D_h + 5 → D_h → D_h]
        # 2*D_h from (h, upstream_h), 5 from physics channels
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_hidden + 5, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.norm = nn.LayerNorm(d_hidden)

    @staticmethod
    def _signed_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Sign-preserving log transform: sign(x) * log(|x| + eps).

        Works for both positive and negative inputs (C1 and C3 terms can be
        negative).  Provides scale invariance while preserving sign information.
        """
        return x.sign() * torch.log(x.abs() + eps)

    def step(
        self,
        h: torch.Tensor,
        c1_next_upstream: torch.Tensor,
        c2_prev_upstream: torch.Tensor,
        c3_self: torch.Tensor,
        c4_lateral: torch.Tensor,
        q_new: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Single GNN update step using all four MC coefficient terms.

        Parameters
        ----------
        h : torch.Tensor
            Current node embeddings, shape (N, D_h).
        c1_next_upstream : torch.Tensor
            C1_i * (N @ Q_{t+1}) — implicit upstream contribution, shape (N,).
        c2_prev_upstream : torch.Tensor
            C2_i * (N @ Q_t)    — explicit upstream contribution, shape (N,).
        c3_self : torch.Tensor
            C3_i * Q_{i,t}      — attenuation / memory term, shape (N,).
        c4_lateral : torch.Tensor
            C4_i * q'_{i,t}    — lateral forcing contribution, shape (N,).
        q_new : torch.Tensor
            Q_{i,t+1}           — total discharge at t+1, shape (N,).
        adjacency : torch.Tensor
            Sparse CSR adjacency matrix N, shape (N, N).  N[i,j]=1 if j → i.

        Returns
        -------
        torch.Tensor
            Updated node embeddings h^{t+1}, shape (N, D_h).
        """
        # Aggregate upstream embeddings via adjacency — mirrors MC upstream inflow
        upstream_h = torch.sparse.mm(adjacency, h)  # [N, D_h]

        # Five physics channels, sign-preserving log-transformed for scale invariance
        # Channels directly correspond to the four MC equation terms + total Q
        phys = torch.stack(
            [
                self._signed_log(c1_next_upstream),  # C1 · N@Q_{t+1} — implicit upstream
                self._signed_log(c2_prev_upstream),  # C2 · N@Q_t     — explicit upstream
                self._signed_log(c3_self),  # C3 · Q_t       — memory / attenuation
                self._signed_log(c4_lateral),  # C4 · q'        — lateral forcing
                self._signed_log(q_new),  # Q_{t+1}        — total discharge
            ],
            dim=-1,
        )  # [N, 5]

        node_input = torch.cat([h, upstream_h, phys], dim=-1)  # [N, 2*D_h + 5]
        return self.norm(h + self.node_mlp(node_input))  # [N, D_h] residual update


class ParamDecoder(nn.Module):
    """Decode physical routing parameters from a latent node embedding.

    This is the extracted output layer of the KAN — a single Linear layer
    followed by sigmoid, mapping the evolving embedding h^t to per-reach
    physical parameters in [0, 1] (before denormalization to physical bounds).

    Parameters
    ----------
    d_hidden : int
        Input embedding dimension D_h.  Must match MCNodeProcessor d_hidden.
    learnable_parameters : list[str]
        Names of physical parameters to decode.  Must match parameter_ranges
        in the config (e.g. ["q_spatial", "top_width", "side_slope", "n"]).
    gate_parameters : list[str] | None
        Parameter names initialized with bias +1.0 (sigmoid ≈ 0.73 → gate ON).
    off_parameters : list[str] | None
        Parameter names initialized with bias −2.0 (sigmoid ≈ 0.12 → OFF).
    device : int | str
        Computation device for the linear layer.
    """

    def __init__(
        self,
        d_hidden: int,
        learnable_parameters: list[str],
        gate_parameters: list[str] | None = None,
        off_parameters: list[str] | None = None,
        device: int | str = "cpu",
    ) -> None:
        super().__init__()
        self.learnable_parameters = list(learnable_parameters)
        self.linear = nn.Linear(d_hidden, len(learnable_parameters), bias=True, device=device)

        nn.init.xavier_normal_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)

        # Gate parameters: bias +1.0 → sigmoid(1) ≈ 0.73 (starts ON)
        if gate_parameters:
            overlap = set(gate_parameters) & set(off_parameters or [])
            if overlap:
                raise ValueError(f"Parameters {overlap} appear in both gate_parameters and off_parameters")
            with torch.no_grad():
                for name in gate_parameters:
                    idx = self.learnable_parameters.index(name)  # raises ValueError if missing
                    self.linear.bias[idx] = 1.0

        # Off parameters: bias −2.0 → sigmoid(−2) ≈ 0.12 (starts OFF)
        if off_parameters:
            with torch.no_grad():
                for name in off_parameters:
                    idx = self.learnable_parameters.index(name)  # raises ValueError if missing
                    self.linear.bias[idx] = -2.0

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode parameters from node embedding.

        Parameters
        ----------
        h : torch.Tensor
            Node embeddings, shape (N, D_h).

        Returns
        -------
        dict[str, torch.Tensor]
            Per-reach sigmoid outputs in [0, 1], one entry per learnable
            parameter, each shape (N,).  Denormalization to physical bounds
            is handled by MuskingumCunge._denormalize_spatial_parameters().
        """
        raw = torch.sigmoid(self.linear(h))  # [N, n_params]
        x_t = raw.transpose(0, 1)  # [n_params, N]
        return {key: x_t[idx] for idx, key in enumerate(self.learnable_parameters)}
