"""PyTorch Muskingum-Cunge Neural Network Module

This module provides a PyTorch nn.Module wrapper around the core Muskingum-Cunge
routing implementation, enabling training and inference with automatic differentiation.
"""

import logging
from typing import Any

import torch

from ddr.routing.mmc import MuskingumCunge
from ddr.validation.configs import Config

log = logging.getLogger(__name__)


class dmc(torch.nn.Module):
    """PyTorch nn.Module for differentiable Muskingum-Cunge routing.

    This class wraps the core MuskingumCunge implementation in a PyTorch module,
    providing forward and backward functionality for training neural networks.
    The module is designed to be GPU/CPU compatible and serves as a pluggable
    replacement for the original dmc implementation.
    """

    def __init__(self, cfg: Config, device: str | torch.device | None = "cpu") -> None:
        """Initialize the PyTorch Muskingum-Cunge module.

        Parameters
        ----------
        cfg : Config
            Configuration object containing routing parameters
        device : str | torch.device | None, optional
            Device to use for computations ("cpu", "cuda", etc.), by default "cpu"
        """
        super().__init__()
        self.cfg = cfg
        self.device_num: str | torch.device = device if device is not None else "cpu"

        # Initialize the core routing engine
        self.routing_engine = MuskingumCunge(cfg, self.device_num)

        # Store configuration parameters as module attributes for compatibility
        self.t = self.routing_engine.t
        self.parameter_bounds = self.routing_engine.parameter_bounds
        self.p_spatial = self.routing_engine.p_spatial
        self.velocity_lb = self.routing_engine.velocity_lb
        self.depth_lb = self.routing_engine.depth_lb
        self.discharge_lb = self.routing_engine.discharge_lb
        self.bottom_width_lb = self.routing_engine.bottom_width_lb

        self._discharge_t: torch.Tensor = torch.empty(0)
        self.network: torch.Tensor = torch.empty(0)
        self.n: torch.Tensor = torch.empty(0)
        self.q_spatial: torch.Tensor = torch.empty(0)
        self.top_width: torch.Tensor = torch.empty(0)
        self.side_slope: torch.Tensor = torch.empty(0)
        self.K_D: torch.Tensor = torch.empty(0)
        self.d_gw: torch.Tensor = torch.empty(0)
        self.leakance_factor: torch.Tensor = torch.empty(0)

        self.epoch = 0
        self.mini_batch = 0

    def to(self, device: torch.device | str) -> "dmc":
        """Move the module to the specified device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        dmc
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
        self.routing_engine = MuskingumCunge(self.cfg, self.device_num)

        # Update tensor attributes
        self.t = self.routing_engine.t
        self.p_spatial = self.routing_engine.p_spatial
        self.velocity_lb = self.routing_engine.velocity_lb
        self.depth_lb = self.routing_engine.depth_lb
        self.discharge_lb = self.routing_engine.discharge_lb
        self.bottom_width_lb = self.routing_engine.bottom_width_lb

        return self

    def cuda(self, device: int | torch.device | None = None) -> "dmc":
        """Move the module to CUDA device.

        Parameters
        ----------
        device : int | torch.device | None, optional
            CUDA device index, by default None

        Returns
        -------
        dmc
            Self for method chaining
        """
        if device is None:
            cuda_device = "cuda"
        elif isinstance(device, int):
            cuda_device = f"cuda:{device}"
        else:
            cuda_device = str(device)

        return self.to(cuda_device)

    def cpu(self) -> "dmc":
        """Move the module to CPU.

        Returns
        -------
        dmc
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

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Forward pass for the Muskingum-Cunge routing model.

        This method performs the complete routing calculation using the core
        MuskingumCunge implementation, maintaining compatibility with the
        original dmc interface.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments containing:
            - routing_dataclass: routing_dataclass object with network and channel properties
            - streamflow: Input streamflow tensor
            - spatial_parameters: Dictionary of spatial parameters (n, q_spatial)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - runoff: Routed discharge at gauge locations
        """
        # Extract inputs
        routing_dataclass = kwargs["routing_dataclass"]
        q_prime = kwargs["streamflow"].to(self.device_num)
        spatial_parameters = kwargs["spatial_parameters"]

        # Setup routing engine with all inputs
        carry_state = kwargs.get("carry_state", False)
        self.routing_engine.setup_inputs(
            routing_dataclass=routing_dataclass,
            streamflow=q_prime,
            spatial_parameters=spatial_parameters,
            carry_state=carry_state,
        )

        # Setup time-varying leakance params from LSTM (if provided)
        leakance_params = kwargs.get("leakance_params", None)
        if leakance_params is not None:
            self.routing_engine.setup_leakance_params(leakance_params)

        # Update compatibility attributes
        self.network = self.routing_engine.network
        self.n = self.routing_engine.n
        self.q_spatial = self.routing_engine.q_spatial
        self.top_width = self.routing_engine.top_width
        self.side_slope = self.routing_engine.side_slope
        self._discharge_t = self.routing_engine._discharge_t
        if self.routing_engine.use_leakance:
            self.K_D = self.routing_engine.K_D
            self.d_gw = self.routing_engine.d_gw
            self.leakance_factor = self.routing_engine.leakance_factor

        # Perform routing
        output = self.routing_engine.forward()

        # Update discharge state for compatibility
        self._discharge_t = self.routing_engine._discharge_t

        if kwargs.get("retain_grads", False):
            if self.n is not None:
                self.n.retain_grad()
            if self.q_spatial is not None:
                self.q_spatial.retain_grad()
            if self._discharge_t is not None:
                self._discharge_t.retain_grad()
            if self.routing_engine.use_leakance:
                if self.routing_engine.K_D is not None:
                    self.routing_engine.K_D.retain_grad()
                if self.routing_engine.d_gw is not None:
                    self.routing_engine.d_gw.retain_grad()
                if self.routing_engine.leakance_factor is not None:
                    self.routing_engine.leakance_factor.retain_grad()

            # Retain gradients for the original spatial parameters so they can be tested
            spatial_params = self.routing_engine.spatial_parameters
            if spatial_params is not None:
                if "n" in spatial_params:
                    spatial_params["n"].retain_grad()
                if "q_spatial" in spatial_params:
                    spatial_params["q_spatial"].retain_grad()
                if "top_width" in spatial_params:
                    spatial_params["top_width"].retain_grad()
                if "side_slope" in spatial_params:
                    spatial_params["side_slope"].retain_grad()
                if "p_spatial" in spatial_params:
                    spatial_params["p_spatial"].retain_grad()
                if "K_D" in spatial_params:
                    spatial_params["K_D"].retain_grad()
                if "d_gw" in spatial_params:
                    spatial_params["d_gw"].retain_grad()
                if "leakance_factor" in spatial_params:
                    spatial_params["leakance_factor"].retain_grad()

            output.retain_grad()  # Retain gradients for the output tensor

        # Return in expected format
        output_dict: dict[str, torch.Tensor] = {
            "runoff": output,
        }

        if self.routing_engine.use_leakance and self.routing_engine._zeta_sum is not None:
            output_dict["zeta_sum"] = self.routing_engine._zeta_sum
            output_dict["q_prime_sum"] = self.routing_engine._q_prime_sum

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
        state: dict[str, Any] = super().state_dict()
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
        self.routing_engine = MuskingumCunge(self.cfg, self.device_num)
        self.routing_engine.set_progress_info(self.epoch, self.mini_batch)
