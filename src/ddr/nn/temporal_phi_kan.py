"""Temporal φ-KAN for bias correction of lateral inflows.

A small KAN that learns an additive correction δ to Q' via a residual connection:

  Q'_corrected = Q' + δ(z, ...)      where z = (Q' - μ) / σ  (z-score)

By Kolmogorov-Arnold theory, δ decomposes as:

  δ(z, sin_m, cos_m) = Σ_q Φ_q( ψ_{q,z}(z) + ψ_{q,sin}(sin_m) + ψ_{q,cos}(cos_m) )

Each ψ is a plottable 1D B-spline curve — fully interpretable.

The residual design ensures:
  - At initialization (KAN ≈ 0), output ≈ Q' (identity pass-through)
  - After training, the KAN learns only the bias correction, not the full signal
"""

import logging
import math

import torch
import torch.nn as nn
from kan import KAN
from torch.utils.checkpoint import checkpoint

from ddr.validation.configs import BiasCorrection
from ddr.validation.enums import PhiInputs

log = logging.getLogger(__name__)

_INPUT_DIM = {
    PhiInputs.STATIC: 1,
    PhiInputs.MONTHLY: 3,
    PhiInputs.FORCING: 2,
    PhiInputs.RANDOM: 3,
}


class TemporalPhiKAN(nn.Module):
    """Small KAN that corrects Q' using flow magnitude and seasonal context.

    Parameters
    ----------
    cfg : BiasCorrection
        Bias correction configuration specifying phi_inputs mode and KAN hyperparameters.
    seed : int
        Random seed for reproducibility (from top-level Config.seed).
    device : int | str
        Compute device (from top-level Config.device).
    """

    def __init__(self, cfg: BiasCorrection, seed: int, device: int | str = "cpu"):
        super().__init__()
        self.phi_inputs = cfg.phi_inputs
        self.input_dim = _INPUT_DIM[cfg.phi_inputs]

        self.phi_kan = KAN(
            [self.input_dim, cfg.phi_hidden_size, 1],
            grid=cfg.phi_grid,
            k=cfg.phi_k,
            seed=seed,
            device=device,
            grid_range=[-4, 4],
        )
        self.phi_kan.save_act = False  # avoid in-place ops + save memory

        self.chunk_size = 8192

    def forward(
        self,
        q_prime: torch.Tensor,
        month: torch.Tensor | None = None,
        forcing: torch.Tensor | None = None,
        q_prime_mean: torch.Tensor | None = None,
        q_prime_std: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bias-correct lateral inflows.

        Parameters
        ----------
        q_prime : (T, N)
            Raw lateral inflow from dHBV2.
        month : (T,), optional
            Month of year as float [1, 12]. Required for MONTHLY mode.
        forcing : (T, N), optional
            Forcing variable values. Required for FORCING mode.
        q_prime_mean : (N,), optional
            Per-basin mean of Q' from pre-computed statistics.
        q_prime_std : (N,), optional
            Per-basin std of Q' from pre-computed statistics.

        Returns
        -------
        q_corrected : (T, N)
            Bias-corrected lateral inflow.
        """
        T, N = q_prime.shape

        # Z-score normalize Q' per-node using pre-computed statistics
        if q_prime_mean is not None and q_prime_std is not None:
            q_norm = (q_prime - q_prime_mean) / (q_prime_std + 1e-8)  # (T, N)
        else:
            q_norm = q_prime

        # Build input tensor based on mode
        if self.phi_inputs == PhiInputs.STATIC:
            phi_input = q_norm.unsqueeze(-1)  # (T, N, 1)

        elif self.phi_inputs == PhiInputs.MONTHLY:
            assert month is not None, "month tensor required for MONTHLY mode"
            two_pi_month = 2.0 * math.pi * month / 12.0  # (T,)
            sin_month = torch.sin(two_pi_month).unsqueeze(1).expand(T, N)  # (T, N)
            cos_month = torch.cos(two_pi_month).unsqueeze(1).expand(T, N)  # (T, N)
            phi_input = torch.stack([q_norm, sin_month, cos_month], dim=-1)  # (T, N, 3)

        elif self.phi_inputs == PhiInputs.FORCING:
            assert forcing is not None, "forcing tensor required for FORCING mode"
            phi_input = torch.stack([q_norm, forcing], dim=-1)  # (T, N, 2)

        elif self.phi_inputs == PhiInputs.RANDOM:
            rand1 = torch.rand(T, N, device=q_prime.device)
            rand2 = torch.rand(T, N, device=q_prime.device)
            phi_input = torch.stack([q_norm, rand1, rand2], dim=-1)  # (T, N, 3)

        else:
            raise ValueError(f"Unknown phi_inputs mode: {self.phi_inputs}")

        # Flatten (T, N, input_dim) → (T*N, input_dim), run KAN in chunks, reshape back.
        # Gradient checkpointing trades compute for memory: intermediates are freed
        # during forward and recomputed during backward (~4 GiB savings).
        phi_input_flat = phi_input.reshape(T * N, self.input_dim)
        if T * N > self.chunk_size:
            chunks = phi_input_flat.split(self.chunk_size, dim=0)
            delta = torch.cat(
                [checkpoint(self.phi_kan, c, use_reentrant=False) for c in chunks], dim=0
            ).reshape(T, N)
        else:
            delta = self.phi_kan(phi_input_flat).reshape(T, N)  # (T, N)

        # Clear pykan internal caches that hold autograd graph references
        self.phi_kan.cache_data = None
        self.phi_kan.acts = []

        # Residual connection: corrected = original + learned correction
        # In z-score space: q_corrected_z = q_norm + δ
        # Denormalized: q_corrected = (q_norm + δ) * σ + μ = q_prime + δ * σ
        if q_prime_mean is not None and q_prime_std is not None:
            q_corrected = (q_norm + delta) * (q_prime_std + 1e-8) + q_prime_mean
        else:
            q_corrected = q_prime + delta

        return torch.clamp(q_corrected, min=1e-6)
