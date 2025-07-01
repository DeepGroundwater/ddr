"""PyTorch Muskingum-Cunge Neural Network Module

This module provides a PyTorch nn.Module wrapper around the core Muskingum-Cunge
routing implementation, enabling training and inference with automatic differentiation.
"""

import logging
from typing import Any

import torch
from omegaconf import DictConfig

from ddr.routing.mmc import MuskingunCunge

log = logging.getLogger(__name__)


class TorchMC(torch.nn.Module):
    """PyTorch nn.Module for differentiable Muskingum-Cunge routing.

    This class wraps the core MuskingunCunge implementation in a PyTorch module,
    providing forward and backward functionality for training neural networks.
    The module is designed to be GPU/CPU compatible and serves as a pluggable
    replacement for the original dmc implementation.
    """

    def __init__(self, cfg: dict[str, Any] | DictConfig, device: str | None = "cpu"):
        """Initialize the PyTorch Muskingum-Cunge module.

        Parameters
        ----------
        cfg : Dict[str, Any] | DictConfig
            Configuration dictionary containing routing parameters
        device : str | None, optional
            Device to use for computations ("cpu", "cuda", etc.), by default "cpu"
        """
        super().__init__()
        self.cfg = cfg
        self.device_num = device if device is not None else "cpu"

        # Initialize the core routing engine
        self.routing_engine = MuskingunCunge(cfg, self.device_num)

        # Store configuration parameters as module attributes for compatibility
        self.t = self.routing_engine.t
        self.parameter_bounds = self.routing_engine.parameter_bounds
        self.p_spatial = self.routing_engine.p_spatial
        self.velocity_lb = self.routing_engine.velocity_lb
        self.depth_lb = self.routing_engine.depth_lb
        self.discharge_lb = self.routing_engine.discharge_lb
        self.bottom_width_lb = self.routing_engine.bottom_width_lb

        # Routing state (for compatibility with original dmc)
        self._discharge_t = None
        self.network = None
        self.n = None
        self.q_spatial = None

        # Progress tracking (for tqdm display compatibility)
        self.epoch = 0
        self.mini_batch = 0

    def to(self, device: torch.device | str) -> "TorchMC":
        """Move the module to the specified device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        TorchMC
            Self for method chaining
        """
        # Call parent to() method
        super().to(device)

        # Update device information
        if isinstance(device, str):
            self.device_num = device
        else:
            self.device_num = str(device)

        # Create new routing engine with updated device
        self.routing_engine = MuskingunCunge(self.cfg, self.device_num)

        # Update tensor attributes
        self.t = self.routing_engine.t
        self.p_spatial = self.routing_engine.p_spatial
        self.velocity_lb = self.routing_engine.velocity_lb
        self.depth_lb = self.routing_engine.depth_lb
        self.discharge_lb = self.routing_engine.discharge_lb
        self.bottom_width_lb = self.routing_engine.bottom_width_lb

        return self

    def cuda(self, device: int | torch.device | None = None) -> "TorchMC":
        """Move the module to CUDA device.

        Parameters
        ----------
        device : int | torch.device | None, optional
            CUDA device index, by default None

        Returns
        -------
        TorchMC
            Self for method chaining
        """
        if device is None:
            cuda_device = "cuda"
        elif isinstance(device, int):
            cuda_device = f"cuda:{device}"
        else:
            cuda_device = str(device)

        return self.to(cuda_device)

    def cpu(self) -> "TorchMC":
        """Move the module to CPU.

        Returns
        -------
        TorchMC
            Self for method chaining
        """
        return self.to("cpu")

    def set_progress_info(self, epoch: int, mini_batch: int) -> None:
        """Set progress information for display purposes.

        Parameters
        ----------
        epoch : int
            Current epoch number
        mini_batch : int
            Current mini batch number
        """
        self.epoch = epoch
        self.mini_batch = mini_batch
        self.routing_engine.set_progress_info(epoch, mini_batch)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass for the Muskingum-Cunge routing model.

        This method performs the complete routing calculation using the core
        MuskingunCunge implementation, maintaining compatibility with the
        original dmc interface.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing:
            - hydrofabric: Hydrofabric object with network and channel properties
            - streamflow: Input streamflow tensor
            - spatial_parameters: Dictionary of spatial parameters (n, q_spatial)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - runoff: Routed discharge at gauge locations
        """
        # Extract inputs
        hydrofabric = kwargs["hydrofabric"]
        q_prime = kwargs["streamflow"].to(self.device_num)
        spatial_parameters = kwargs["spatial_parameters"]

        # Setup routing engine with all inputs
        self.routing_engine.setup_inputs(
            hydrofabric=hydrofabric, streamflow=q_prime, spatial_parameters=spatial_parameters
        )

        # Update compatibility attributes
        self.network = self.routing_engine.network
        self.n = self.routing_engine.n
        self.q_spatial = self.routing_engine.q_spatial
        self._discharge_t = self.routing_engine._discharge_t

        # Perform routing
        output = self.routing_engine.forward()

        # Update discharge state for compatibility
        self._discharge_t = self.routing_engine._discharge_t

        # Return in expected format
        output_dict = {
            "runoff": output,
        }

        return output_dict

    def fill_op(self, data_vector: torch.Tensor) -> torch.Tensor:
        """Fill operation function for sparse matrix (compatibility method).

        This method provides compatibility with the original dmc interface
        by delegating to the routing engine.

        Parameters
        ----------
        data_vector : torch.Tensor
            Data vector to fill the sparse matrix with

        Returns
        -------
        torch.Tensor
            Filled sparse matrix
        """
        return self.routing_engine.fill_op(data_vector)

    def _sparse_eye(self, n: int) -> torch.Tensor:
        """Create sparse identity matrix (compatibility method).

        Parameters
        ----------
        n : int
            Matrix dimension

        Returns
        -------
        torch.Tensor
            Sparse identity matrix
        """
        return self.routing_engine._sparse_eye(n)

    def _sparse_diag(self, data: torch.Tensor) -> torch.Tensor:
        """Create sparse diagonal matrix (compatibility method).

        Parameters
        ----------
        data : torch.Tensor
            Diagonal values

        Returns
        -------
        torch.Tensor
            Sparse diagonal matrix
        """
        return self.routing_engine._sparse_diag(data)

    def route_timestep(
        self,
        q_prime_clamp: torch.Tensor,
        mapper: Any,
    ) -> torch.Tensor:
        """Route flow for a single timestep (compatibility method).

        Parameters
        ----------
        q_prime_clamp : torch.Tensor
            Clamped lateral inflow
        mapper : Any
            Pattern mapper for sparse operations

        Returns
        -------
        torch.Tensor
            Routed discharge
        """
        return self.routing_engine.route_timestep(
            q_prime_clamp=q_prime_clamp,
            mapper=mapper,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return state dictionary for saving/loading.

        Returns
        -------
        Dict[str, Any]
            State dictionary
        """
        state = super().state_dict()
        state["cfg"] = self.cfg
        state["device_num"] = self.device_num
        state["epoch"] = self.epoch
        state["mini_batch"] = self.mini_batch
        return state

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """Load state dictionary.

        Parameters
        ----------
        state_dict : Dict[str, Any]
            State dictionary to load
        strict : bool, optional
            Whether to strictly enforce key matching, by default True
        """
        # Extract custom attributes before calling parent
        cfg = state_dict.pop("cfg", self.cfg)
        device_num = state_dict.pop("device_num", self.device_num)
        epoch = state_dict.pop("epoch", 0)
        mini_batch = state_dict.pop("mini_batch", 0)

        # Load parent state
        super().load_state_dict(state_dict, strict)

        # Restore custom attributes
        self.cfg = cfg
        self.device_num = device_num
        self.epoch = epoch
        self.mini_batch = mini_batch

        # Recreate routing engine
        self.routing_engine = MuskingunCunge(self.cfg, self.device_num)
        self.routing_engine.set_progress_info(self.epoch, self.mini_batch)


# Alias for backward compatibility
dmc = TorchMC
