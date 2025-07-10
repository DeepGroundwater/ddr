"""Differentiable Muskingum-Cunge"""

import logging
import warnings

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from ddr.routing.utils import (
    PatternMapper,
    denormalize,
    get_network_idx,
    # RiverNetworkMatrix,
    triangular_sparse_solve,
)

log = logging.getLogger(__name__)

warnings.warn(
    "ddr.routing.dmc is deprecated and will be removed in a future version. "
    "Use ddr.routing.torch_mc.TorchMC instead.",
    DeprecationWarning,
    stacklevel=2,
)


def _log_base_q(x, q):
    return torch.log(x) / torch.log(torch.tensor(q, dtype=x.dtype))


def _get_trapezoid_velocity(
    q_t,
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
    """Calculate flow velocity using Manning's equation."""
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


class dmc(torch.nn.Module):
    """
    dMC is a differentiable implementation of the Muskingum Cunge River rouing function

    This class implements the forward pass for the dMC model, including the setup of
    various parameters and the handling of reservoir routing if needed.
    """

    def __init__(self, cfg: dict[str, any] | DictConfig, device: str | None = "cpu"):
        super().__init__()
        self.cfg = cfg

        self.device_num = device

        self.t = torch.tensor(
            3600.0,
            device=self.device_num,
        )

        # Base routing parameters
        self.n = None
        self.q_spatial = None

        # Routing state
        self._discharge_t = None
        self.network = None

        self.parameter_bounds = self.cfg.params.parameter_ranges.range
        self.p_spatial = torch.tensor(self.cfg.params.defaults.p, device=self.device_num)
        self.velocity_lb = torch.tensor(self.cfg.params.attribute_minimums.velocity, device=self.device_num)
        self.depth_lb = torch.tensor(self.cfg.params.attribute_minimums.depth, device=self.device_num)
        self.discharge_lb = torch.tensor(self.cfg.params.attribute_minimums.discharge, device=self.device_num)
        self.bottom_width_lb = torch.tensor(
            self.cfg.params.attribute_minimums.bottom_width, device=self.device_num
        )

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """The forward pass for the dMC model

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing the runoff, zeta, and reservoir storage tensors
        """
        # Setup solver and get hydrofabric data

        hydrofabric = kwargs["hydrofabric"]
        q_prime = kwargs["streamflow"].to(self.device_num)
        observations = hydrofabric.observations.gage_id
        # gage_information = hydrofabric.network.gage_information
        # TODO: create a dynamic gauge look up
        gage_indices = torch.tensor([-1])
        self.network = hydrofabric.adjacency_matrix

        # Set up base parameters
        self.n = denormalize(value=kwargs["spatial_parameters"]["n"], bounds=self.parameter_bounds["n"])
        self.q_spatial = denormalize(
            value=kwargs["spatial_parameters"]["q_spatial"],
            bounds=self.parameter_bounds["q_spatial"],
        )

        # Initialize discharge
        self._discharge_t = q_prime[0].to(self.device_num)

        # Setup output tensors
        output = torch.zeros(
            size=[observations.shape[0], q_prime.shape[0]],
            device=torch.device(self.device_num),
        )

        # Initialize mapper
        matrix_dims = self.network.shape[0]
        mapper = PatternMapper(self.fill_op, matrix_dims, device=self.device_num)
        dense_rows, dense_cols = get_network_idx(mapper)

        # Set initial output values
        if len(self._discharge_t) != 0:
            for i, gage_idx in enumerate(gage_indices):
                output[i, 0] = torch.sum(self._discharge_t[gage_idx])
        else:
            for i, gage_idx in enumerate(gage_indices):
                output[i, 0] = q_prime[0, gage_idx]
        output[:, 0] = torch.clamp(input=output[:, 0], min=self.discharge_lb)

        # Get spatial attributes
        length = hydrofabric.length.to(self.device_num).to(torch.float32)
        slope = torch.clamp(
            hydrofabric.slope.to(self.device_num).to(torch.float32),
            min=self.cfg.params.attribute_minimums.slope,
        )
        top_width = hydrofabric.top_width.to(self.device_num).to(torch.float32)
        side_slope = hydrofabric.side_slope.to(self.device_num).to(torch.float32)
        x_storage = hydrofabric.x.to(self.device_num).to(torch.float32)

        desc = "Running dMC Routing"
        for timestep in tqdm(
            range(1, len(q_prime)),
            desc=f"\r{desc} for Epoch: {self.epoch} | Mini Batch: {self.mini_batch} | ",
            ncols=140,
            ascii=True,
        ):
            q_prime_sub = q_prime[timestep - 1].clone()
            q_prime_clamp = torch.clamp(q_prime_sub, min=self.cfg.params.attribute_minimums.discharge)
            velocity = _get_trapezoid_velocity(
                q_t=self._discharge_t,
                _n=self.n,
                top_width=top_width,
                side_slope=side_slope,
                _s0=slope,
                p_spatial=self.p_spatial,
                _q_spatial=self.q_spatial,
                velocity_lb=self.velocity_lb,
                depth_lb=self.depth_lb,
                _btm_width_lb=self.bottom_width_lb,
            )
            k = torch.div(length, velocity)
            denom = (2.0 * k * (1.0 - x_storage)) + self.t
            c_2 = (self.t + (2.0 * k * x_storage)) / denom
            c_3 = ((2.0 * k * (1.0 - x_storage)) - self.t) / denom
            c_4 = (2.0 * self.t) / denom
            i_t = torch.matmul(self.network, self._discharge_t)
            q_l = q_prime_clamp

            b = (c_2 * i_t) + (c_3 * self._discharge_t) + (c_4 * q_l)
            c_1 = (self.t - (2.0 * k * x_storage)) / denom
            c_1_ = c_1 * -1
            c_1_[0] = 1.0
            A_values = mapper.map(c_1_)
            try:
                solution = triangular_sparse_solve(
                    A_values,
                    mapper.crow_indices,
                    mapper.col_indices,
                    b,
                    True,  # lower=True
                    False,  # unit_diagonal=False
                    self.cfg.device,  # device
                )
                # b_arr = b.unsqueeze(-1)
                # A_dense = self.network.clone()
                # A_dense[dense_rows, dense_cols] = A_values
                # x = solve_triangular(A_dense, b_arr, upper=False)
                # solution_b = x.squeeze()
            except torch.cuda.OutOfMemoryError as e:
                raise torch.cuda.OutOfMemoryError from e

            q_t1 = torch.clamp(solution, min=self.discharge_lb)

            for i, gage_idx in enumerate(gage_indices):
                output[i, timestep] = torch.sum(q_t1[gage_idx])

            self._discharge_t = q_t1

        if kwargs.get("retain_grads", False):
            self.n.retain_grad()
            self.q_spatial.retain_grad()
            self._discharge_t.retain_grad()  # Retain gradients for the discharge tensor
            output.retain_grad()  # Retain gradients for the output tensor

        output_dict = {
            "runoff": output,
        }

        return output_dict

    def fill_op(self, data_vector: torch.Tensor):
        """A fill operation function for the sparse matrix

        The equation we want to solve
        (I - C_1*N) * Q_t+1 = c_2*(N*Q_t_1) + c_3*Q_t + c_4*Q`
        (I - C_1*N) * Q_t+1 = b(t)

        Parameters
        ----------
        data_vector: torch.Tensor
            The data vector to fill the sparse matrix with
        """
        identity_matrix = self._sparse_eye(self.network.shape[0])
        vec_diag = self._sparse_diag(data_vector)
        # vec_filled = bnb.matmul(vec_diag, self.network, threshold=6.0)
        vec_filled = torch.matmul(vec_diag.cpu(), self.network.cpu()).to(self.device_num)
        A = identity_matrix + vec_filled
        return A

    def _sparse_eye(self, n):
        indices = torch.arange(n, dtype=torch.int32, device=self.device_num)
        values = torch.ones(n, device=self.device_num)
        identity_coo = torch.sparse_coo_tensor(
            indices=torch.vstack([indices, indices]),
            values=values,
            size=(n, n),
            device=self.device_num,
        )
        return identity_coo.to_sparse_csr()

    def _sparse_diag(self, data):
        n = len(data)
        indices = torch.arange(n, dtype=torch.int32, device=self.device_num)
        diagonal_coo = torch.sparse_coo_tensor(
            indices=torch.vstack([indices, indices]),
            values=data,
            size=(n, n),
            device=self.device_num,
        )
        return diagonal_coo.to_sparse_csr()
