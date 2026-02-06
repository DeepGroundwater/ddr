"""Muskingum-Cunge routing implementation

This module contains the core mathematical implementation of the Muskingum-Cunge routing
algorithm without PyTorch dependencies, designed to be used by the differentiable
implementation.
"""

import logging
from typing import Any

import torch
from tqdm import tqdm

from ddr.routing.utils import (
    PatternMapper,
    denormalize,
    get_network_idx,
    triangular_sparse_solve,
)
from ddr.validation.configs import Config

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
    algorithm, managing all routing_dataclass data, parameters, and routing calculations.
    """

    def __init__(self, cfg: Config, device: str | torch.device = "cpu") -> None:
        """Initialize the Muskingum-Cunge router.

        Parameters
        ----------
        cfg : Config
            Configuration object containing routing parameters
        device : str | torch.device, optional
            Device to use for computations, by default "cpu"
        """
        self.cfg = cfg
        self.device = device

        # Time step (1 hour in seconds)
        self.t = torch.tensor(3600.0, device=self.device)

        # Routing parameters
        self.n: torch.Tensor | None = None
        self.q_spatial: torch.Tensor | None = None
        self._discharge_t: torch.Tensor | None = None
        self.network: torch.Tensor | None = None

        # Parameter bounds and defaults
        self.parameter_bounds = self.cfg.params.parameter_ranges
        self.p_spatial = torch.tensor(self.cfg.params.defaults["p_spatial"], device=self.device)
        self.velocity_lb = torch.tensor(self.cfg.params.attribute_minimums["velocity"], device=self.device)
        self.depth_lb = torch.tensor(self.cfg.params.attribute_minimums["depth"], device=self.device)
        self.discharge_lb = torch.tensor(self.cfg.params.attribute_minimums["discharge"], device=self.device)
        self.bottom_width_lb = torch.tensor(
            self.cfg.params.attribute_minimums["bottom_width"], device=self.device
        )

        # routing_dataclass data - managed internally
        self.routing_dataclass: Any = None
        self.length: torch.Tensor | None = None
        self.slope: torch.Tensor | None = None
        self.top_width: torch.Tensor | None = None
        self.side_slope: torch.Tensor | None = None
        self.x_storage: torch.Tensor | None = None
        self.observations: Any = None
        self.output_indices: list[Any] | None = None
        self.gage_catchment: list[str] | None = None

        # Input data
        self.q_prime: torch.Tensor | None = None
        self.spatial_parameters: dict[str, torch.Tensor] | None = None

        # Progress tracking attributes (for tqdm display)
        self.epoch = 0
        self.mini_batch = 0

        # Scatter indices for ragged output (initialized in setup_inputs)
        self._flat_indices: torch.Tensor | None = None
        self._group_ids: torch.Tensor | None = None
        self._num_outputs: int | None = None
        self._scatter_input: torch.Tensor | None = None

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
        self,
        routing_dataclass: Any,
        streamflow: torch.Tensor,
        spatial_parameters: dict[str, torch.Tensor],
        carry_state: bool = False,
    ) -> None:
        """Setup all inputs for routing including routing_dataclass, streamflow, and parameters.

        Parameters
        ----------
        carry_state : bool
            If True, preserve discharge state from the previous batch instead of
            reinitializing from q_prime[0]. Set to True for sequential inference
            (testing/benchmarking) so that batches maintain physical continuity.
        """
        # Store routing_dataclass
        self.routing_dataclass = routing_dataclass
        self.output_indices = routing_dataclass.outflow_idx
        self.gage_catchment = routing_dataclass.gage_catchment

        # Handle observations (only present in gages mode)
        if routing_dataclass.observations is not None:
            self.observations = routing_dataclass.observations.gage_id
        else:
            self.observations = None

        # Setup network
        self.network = routing_dataclass.adjacency_matrix

        # Extract and prepare spatial attributes
        self.length = routing_dataclass.length.to(self.device).to(torch.float32)
        self.slope = torch.clamp(
            routing_dataclass.slope.to(self.device).to(torch.float32),
            min=self.cfg.params.attribute_minimums["slope"],
        )
        self.x_storage = routing_dataclass.x.to(self.device).to(torch.float32)

        # Setup streamflow
        self.q_prime = streamflow.to(self.device)

        # Setup spatial parameters
        self.spatial_parameters = spatial_parameters
        log_space_params = self.cfg.params.log_space_parameters
        self.n = denormalize(
            value=spatial_parameters["n"],
            bounds=self.parameter_bounds["n"],
            log_space="n" in log_space_params,
        )
        self.q_spatial = denormalize(
            value=spatial_parameters["q_spatial"],
            bounds=self.parameter_bounds["q_spatial"],
            log_space="q_spatial" in log_space_params,
        )
        if routing_dataclass.top_width.numel() == 0:
            self.top_width = denormalize(
                value=spatial_parameters["top_width"],
                bounds=self.parameter_bounds["top_width"],
                log_space="top_width" in log_space_params,
            )
        else:
            self.top_width = routing_dataclass.top_width.to(self.device).to(torch.float32)
        if routing_dataclass.side_slope.numel() == 0:
            self.side_slope = denormalize(
                value=spatial_parameters["side_slope"],
                bounds=self.parameter_bounds["side_slope"],
                log_space="side_slope" in log_space_params,
            )
        else:
            self.side_slope = routing_dataclass.side_slope.to(self.device).to(torch.float32)

        # Initialize discharge: carry over from previous batch if requested,
        # otherwise start from the first timestep's lateral inflow.
        # TODO: A better cold-start initialization would use summed Q' (upstream
        # accumulation via the adjacency matrix) instead of local q_prime[0].
        # This would give a physically reasonable initial discharge at each node.
        # Needs to be computed in the dataset to avoid per-batch matrix ops on
        # large networks (~77K nodes for CONUS).
        if carry_state and self._discharge_t is not None:
            pass
        else:
            self._discharge_t = self.q_prime[0].to(self.device)

        # Precompute scatter_add indices for ragged output_indices (gages mode)
        if self.output_indices is not None and len(self.output_indices) != len(self._discharge_t):
            self._flat_indices = torch.cat(
                [torch.as_tensor(idx, device=self.device, dtype=torch.long) for idx in self.output_indices]
            )
            self._group_ids = torch.cat(
                [
                    torch.full((len(idx),), i, device=self.device, dtype=torch.long)
                    for i, idx in enumerate(self.output_indices)
                ]
            )
            self._num_outputs = len(self.output_indices)
            self._scatter_input = torch.zeros(self._num_outputs, device=self.device, dtype=torch.float32)
        else:
            self._flat_indices = None
            self._group_ids = None
            self._num_outputs = None
            self._scatter_input = None

    def forward(self) -> torch.Tensor:
        """Perform forward routing calculation."""
        if self.routing_dataclass is None:
            raise ValueError("routing_dataclass not set. Call setup_inputs() first.")
        if self.q_prime is None or self._discharge_t is None:
            raise ValueError("Streamflow not set. Call setup_inputs() first.")

        num_timesteps = self.q_prime.shape[0]
        num_segments = len(self._discharge_t)
        mapper, _, _ = self.create_pattern_mapper()

        # Check if outputting all segments
        output_all = self.output_indices is None or len(self.output_indices) == num_segments

        if output_all:
            output = torch.zeros(
                (num_segments, num_timesteps),
                device=self.device,
                dtype=torch.float32,
            )
            output[:, 0] = torch.clamp(self._discharge_t, min=self.discharge_lb)
        else:
            if self._flat_indices is None or self._group_ids is None or self._num_outputs is None:
                raise ValueError("Scatter indices not initialized properly")
            if self._scatter_input is None:
                raise ValueError("Scatter input not initialized")

            assert self.output_indices is not None
            max_idx = max(idx.max() for idx in self.output_indices)
            assert max_idx < num_segments, (
                f"Output index {max_idx} out of bounds for discharge tensor of size {num_segments}."
            )

            output = torch.zeros(
                (self._num_outputs, num_timesteps),
                device=self.device,
                dtype=torch.float32,
            )

            # Vectorized initial values
            gathered = self._discharge_t[self._flat_indices]
            output[:, 0] = torch.scatter_add(
                input=self._scatter_input,
                dim=0,
                index=self._group_ids,
                src=gathered,
            )
            output[:, 0] = torch.clamp(output[:, 0], min=self.discharge_lb)

        # Route through time series
        for timestep in tqdm(
            range(1, num_timesteps),
            desc=f"\rRunning dMC Routing for Epoch: {self.epoch} | Mini Batch: {self.mini_batch} | ",
            ncols=140,
            ascii=True,
        ):
            q_prime_clamp = torch.clamp(
                self.q_prime[timestep - 1],
                min=self.cfg.params.attribute_minimums["discharge"],
            )

            q_t1 = self.route_timestep(q_prime_clamp=q_prime_clamp, mapper=mapper)

            if output_all:
                output[:, timestep] = q_t1
            else:
                if self._flat_indices is None or self._group_ids is None or self._scatter_input is None:
                    raise ValueError("Scatter indices not initialized")
                gathered = q_t1[self._flat_indices]
                output[:, timestep] = torch.scatter_add(
                    input=self._scatter_input,
                    dim=0,
                    index=self._group_ids,
                    src=gathered,
                )

            self._discharge_t = q_t1

        return output

    def create_pattern_mapper(self) -> tuple[PatternMapper, torch.Tensor, torch.Tensor]:
        """Create pattern mapper for sparse matrix operations.

        Returns
        -------
        Tuple[PatternMapper, torch.Tensor, torch.Tensor]
            Pattern mapper and dense row/column indices
        """
        if self.network is None:
            raise ValueError("Network not set. Call setup_inputs() first.")
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
        if (
            self._discharge_t is None
            or self.n is None
            or self.top_width is None
            or self.side_slope is None
            or self.slope is None
            or self.q_spatial is None
            or self.length is None
            or self.x_storage is None
            or self.network is None
        ):
            raise ValueError("Required attributes not set. Call setup_inputs() first.")

        # Calculate velocity using internal routing_dataclass data
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
        if self.network is None:
            raise ValueError("Network not set. Call setup_inputs() first.")
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
