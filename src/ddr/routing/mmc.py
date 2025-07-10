"""Muskingum-Cunge routing implementation

This module contains the core mathematical implementation of the Muskingum-Cunge routing
algorithm without PyTorch dependencies, designed to be used by the differentiable
implementation.
"""

import logging
from typing import Any

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ddr.routing.utils import (
    PatternMapper,
    denormalize,
    get_network_idx,
    triangular_sparse_solve,
)

log = logging.getLogger(__name__)


def _log_base_q(x: torch.Tensor, q: float) -> torch.Tensor:
    """Calculate logarithm with base q."""
    return torch.log(x) / torch.log(torch.tensor(q, dtype=x.dtype))


def _get_trapezoid_velocity(
    q_t: torch.Tensor,
    _n: torch.Tensor,
    top_width: torch.Tensor,
    side_slope: torch.Tensor,
    _s0: torch.Tensor,
    p_spatial: torch.Tensor,
    _q_spatial: torch.Tensor,
    velocity_lb: torch.Tensor,
    depth_lb: torch.Tensor,
    _btm_width_lb: torch.Tensor,
) -> torch.Tensor:
    """Calculate flow velocity using Manning's equation for trapezoidal channels.

    Parameters
    ----------
    q_t : torch.Tensor
        Discharge at time t
    _n : torch.Tensor
        Manning's roughness coefficient
    top_width : torch.Tensor
        Top width of channel
    side_slope : torch.Tensor
        Side slope of channel (z:1, z horizontal : 1 vertical)
    _s0 : torch.Tensor
        Channel slope
    p_spatial : torch.Tensor
        Spatial parameter p
    _q_spatial : torch.Tensor
        Spatial parameter q
    velocity_lb : torch.Tensor
        Lower bound for velocity
    depth_lb : torch.Tensor
        Lower bound for depth
    _btm_width_lb : torch.Tensor
        Lower bound for bottom width

    Returns
    -------
    torch.Tensor
        Flow velocity
    """
    numerator = q_t * _n * (_q_spatial + 1)
    denominator = p_spatial * torch.pow(_s0, 0.5)
    depth = torch.clamp(
        torch.pow(
            torch.div(numerator, denominator + 1e-8),
            torch.div(3.0, 5.0 + 3.0 * _q_spatial),
        ),
        min=depth_lb,
    )

    # For z:1 side slopes (z horizontal : 1 vertical)
    _bottom_width = top_width - (2 * side_slope * depth)
    bottom_width = torch.clamp(_bottom_width, min=_btm_width_lb)

    # Area = (top_width + bottom_width)*depth/2
    area = (top_width + bottom_width) * depth / 2

    # Side length = sqrt(1 + z^2) * depth
    # Since for every 1 unit vertical, we go z units horizontal
    wetted_p = bottom_width + 2 * depth * torch.sqrt(1 + side_slope**2)

    # Calculate hydraulic radius
    R = area / wetted_p

    v = torch.div(1, _n) * torch.pow(R, (2 / 3)) * torch.pow(_s0, (1 / 2))
    c_ = torch.clamp(v, min=velocity_lb, max=torch.tensor(15.0, device=v.device))
    c = c_ * 5 / 3
    return c


class MuskingumCunge:
    """Core Muskingum-Cunge routing implementation.

    This class implements the mathematical core of the Muskingum-Cunge routing
    algorithm, managing all hydrofabric data, parameters, and routing calculations.
    """

    def __init__(self, cfg: dict[str, Any] | DictConfig, device: str = "cpu"):
        """Initialize the Muskingum-Cunge router.

        Parameters
        ----------
        cfg : Dict[str, Any] | DictConfig
            Configuration dictionary containing routing parameters
        device : str, optional
            Device to use for computations, by default "cpu"
        """
        self.cfg = cfg
        self.device = device

        # Time step (1 hour in seconds)
        self.t = torch.tensor(3600.0, device=self.device)

        # Routing parameters
        self.n = None
        self.q_spatial = None
        self._discharge_t = None
        self.network = None

        # Parameter bounds and defaults
        self.parameter_bounds = self.cfg.params.parameter_ranges.range
        self.p_spatial = torch.tensor(self.cfg.params.defaults.p, device=self.device)
        self.velocity_lb = torch.tensor(self.cfg.params.attribute_minimums.velocity, device=self.device)
        self.depth_lb = torch.tensor(self.cfg.params.attribute_minimums.depth, device=self.device)
        self.discharge_lb = torch.tensor(self.cfg.params.attribute_minimums.discharge, device=self.device)
        self.bottom_width_lb = torch.tensor(
            self.cfg.params.attribute_minimums.bottom_width, device=self.device
        )

        # Hydrofabric data - managed internally
        self.hydrofabric = None
        self.length = None
        self.slope = None
        self.top_width = None
        self.side_slope = None
        self.x_storage = None
        self.observations = None
        self.gage_indices = None

        # Input data
        self.q_prime = None
        self.spatial_parameters = None

        # Progress tracking attributes (for tqdm display)
        self.epoch = 0
        self.mini_batch = 0

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

    def setup_inputs(
        self, hydrofabric: Any, streamflow: torch.Tensor, spatial_parameters: dict[str, torch.Tensor]
    ) -> None:
        """Setup all inputs for routing including hydrofabric, streamflow, and parameters.

        Parameters
        ----------
        hydrofabric : Any
            Hydrofabric object containing network and channel properties
        streamflow : torch.Tensor
            Input streamflow time series
        spatial_parameters : Dict[str, torch.Tensor]
            Dictionary containing spatial parameters (n, q_spatial)
        """
        # Store hydrofabric and extract spatial attributes
        self.hydrofabric = hydrofabric
        self.observations = hydrofabric.observations.gage_id

        # Setup network
        self.network = hydrofabric.adjacency_matrix

        # Extract and prepare spatial attributes
        self.length = hydrofabric.length.to(self.device).to(torch.float32)
        self.slope = torch.clamp(
            hydrofabric.slope.to(self.device).to(torch.float32),
            min=self.cfg.params.attribute_minimums.slope,
        )
        self.top_width = hydrofabric.top_width.to(self.device).to(torch.float32)
        self.side_slope = hydrofabric.side_slope.to(self.device).to(torch.float32)
        self.x_storage = hydrofabric.x.to(self.device).to(torch.float32)

        # Setup streamflow
        self.q_prime = streamflow.to(self.device)

        # Setup spatial parameters
        self.spatial_parameters = spatial_parameters
        self.n = denormalize(value=spatial_parameters["n"], bounds=self.parameter_bounds["n"])
        self.q_spatial = denormalize(
            value=spatial_parameters["q_spatial"],
            bounds=self.parameter_bounds["q_spatial"],
        )

        # Initialize discharge
        self._discharge_t = self.q_prime[0].to(self.device)

        # TODO: Create dynamic gauge lookup - for now using placeholder
        self.gage_indices = torch.tensor([-1])

    def forward(self) -> torch.Tensor:
        """Perform forward routing calculation.

        Returns
        -------
        torch.Tensor
            Routed discharge at gauge locations
        """
        if self.hydrofabric is None:
            raise ValueError("Hydrofabric not set. Call setup_inputs() first.")

        # Setup output tensor
        output = torch.zeros(
            size=[self.observations.shape[0], self.q_prime.shape[0]],
            device=torch.device(self.device),
        )

        # Create pattern mapper
        mapper, dense_rows, dense_cols = self.create_pattern_mapper()

        # Set initial output values
        if len(self._discharge_t) != 0:
            for i, gage_idx in enumerate(self.gage_indices):
                output[i, 0] = torch.sum(self._discharge_t[gage_idx])
        else:
            for i, gage_idx in enumerate(self.gage_indices):
                output[i, 0] = self.q_prime[0, gage_idx]
        output[:, 0] = torch.clamp(input=output[:, 0], min=self.discharge_lb)

        # Route through time series
        desc = "Running dMC Routing"
        for timestep in tqdm(
            range(1, len(self.q_prime)),
            desc=f"\r{desc} for Epoch: {self.epoch} | Mini Batch: {self.mini_batch} | ",
            ncols=140,
            ascii=True,
        ):
            q_prime_sub = self.q_prime[timestep - 1].clone()
            q_prime_clamp = torch.clamp(q_prime_sub, min=self.cfg.params.attribute_minimums.discharge)

            # Route this timestep
            q_t1 = self.route_timestep(
                q_prime_clamp=q_prime_clamp,
                mapper=mapper,
            )

            # Store output at gauge locations
            for i, gage_idx in enumerate(self.gage_indices):
                output[i, timestep] = torch.sum(q_t1[gage_idx])

            # Update discharge state
            self._discharge_t = q_t1

        return output

    def create_pattern_mapper(self) -> tuple[PatternMapper, torch.Tensor, torch.Tensor]:
        """Create pattern mapper for sparse matrix operations.

        Returns
        -------
        Tuple[PatternMapper, torch.Tensor, torch.Tensor]
            Pattern mapper and dense row/column indices
        """
        matrix_dims = self.network.shape[0]
        mapper = PatternMapper(self.fill_op, matrix_dims, device=self.device)
        dense_rows, dense_cols = get_network_idx(mapper)
        return mapper, dense_rows, dense_cols

    def calculate_muskingum_coefficients(
        self, length: torch.Tensor, velocity: torch.Tensor, x_storage: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Muskingum-Cunge routing coefficients.

        Parameters
        ----------
        length : torch.Tensor
            Channel length
        velocity : torch.Tensor
            Flow velocity
        x_storage : torch.Tensor
            Storage coefficient

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            Routing coefficients c1, c2, c3, c4
        """
        k = torch.div(length, velocity)
        denom = (2.0 * k * (1.0 - x_storage)) + self.t
        c_1 = (self.t - (2.0 * k * x_storage)) / denom
        c_2 = (self.t + (2.0 * k * x_storage)) / denom
        c_3 = ((2.0 * k * (1.0 - x_storage)) - self.t) / denom
        c_4 = (2.0 * self.t) / denom
        return c_1, c_2, c_3, c_4

    def route_timestep(
        self,
        q_prime_clamp: torch.Tensor,
        mapper: PatternMapper,
    ) -> torch.Tensor:
        """Route flow for a single timestep.

        Parameters
        ----------
        q_prime_clamp : torch.Tensor
            Clamped lateral inflow
        mapper : PatternMapper
            Pattern mapper for sparse operations

        Returns
        -------
        torch.Tensor
            Routed discharge
        """
        # Calculate velocity using internal hydrofabric data
        velocity = _get_trapezoid_velocity(
            q_t=self._discharge_t,
            _n=self.n,
            top_width=self.top_width,
            side_slope=self.side_slope,
            _s0=self.slope,
            p_spatial=self.p_spatial,
            _q_spatial=self.q_spatial,
            velocity_lb=self.velocity_lb,
            depth_lb=self.depth_lb,
            _btm_width_lb=self.bottom_width_lb,
        )

        # Calculate routing coefficients
        c_1, c_2, c_3, c_4 = self.calculate_muskingum_coefficients(self.length, velocity, self.x_storage)

        # Calculate inflow from upstream
        i_t = torch.matmul(self.network, self._discharge_t)

        # Calculate right-hand side of equation
        b = (c_2 * i_t) + (c_3 * self._discharge_t) + (c_4 * q_prime_clamp)

        # Setup sparse matrix for solving
        c_1_ = c_1 * -1
        c_1_[0] = 1.0
        A_values = mapper.map(c_1_)

        # Solve the linear system
        solution = triangular_sparse_solve(
            A_values,
            mapper.crow_indices,
            mapper.col_indices,
            b,
            True,  # lower=True
            False,  # unit_diagonal=False
            self.device,
        )

        # Clamp solution to physical bounds
        q_t1 = torch.clamp(solution, min=self.discharge_lb)

        return q_t1

    def fill_op(self, data_vector: torch.Tensor) -> torch.Tensor:
        """Fill operation function for the sparse matrix.

        The equation we want to solve:
        (I - C_1*N) * Q_t+1 = c_2*(N*Q_t_1) + c_3*Q_t + c_4*Q`
        (I - C_1*N) * Q_t+1 = b(t)

        Parameters
        ----------
        data_vector : torch.Tensor
            The data vector to fill the sparse matrix with

        Returns
        -------
        torch.Tensor
            Filled sparse matrix
        """
        identity_matrix = self._sparse_eye(self.network.shape[0])
        vec_diag = self._sparse_diag(data_vector)
        vec_filled = torch.matmul(vec_diag.cpu(), self.network.cpu()).to(self.device)
        A = identity_matrix + vec_filled
        return A

    def _sparse_eye(self, n: int) -> torch.Tensor:
        """Create sparse identity matrix.

        Parameters
        ----------
        n : int
            Matrix dimension

        Returns
        -------
        torch.Tensor
            Sparse identity matrix
        """
        indices = torch.arange(n, dtype=torch.int32, device=self.device)
        values = torch.ones(n, device=self.device)
        identity_coo = torch.sparse_coo_tensor(
            indices=torch.vstack([indices, indices]),
            values=values,
            size=(n, n),
            device=self.device,
        )
        return identity_coo.to_sparse_csr()

    def _sparse_diag(self, data: torch.Tensor) -> torch.Tensor:
        """Create sparse diagonal matrix.

        Parameters
        ----------
        data : torch.Tensor
            Diagonal values

        Returns
        -------
        torch.Tensor
            Sparse diagonal matrix
        """
        n = len(data)
        indices = torch.arange(n, dtype=torch.int32, device=self.device)
        diagonal_coo = torch.sparse_coo_tensor(
            indices=torch.vstack([indices, indices]),
            values=data,
            size=(n, n),
            device=self.device,
        )
        return diagonal_coo.to_sparse_csr()
