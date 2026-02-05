"""Integration tests for DDR (Differentiable Muskingum-Cunge) routing.

Tests DDR routing using RAPID Sandbox data with mock streamflow and KAN modules.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

# Reach IDs in the Sandbox network (RAPID2 ordering)
REACH_IDS = [10, 20, 30, 40, 50]


# =============================================================================
# Mock Classes
# =============================================================================


class MockStreamflow(torch.nn.Module):
    """Mock streamflow module that returns pre-interpolated Q' (lateral inflow).

    In real DDR, the streamflow module predicts Q' from forcing data.
    For testing, we directly use interpolated Qext as the Q'.
    """

    def __init__(self, qprime: torch.Tensor) -> None:
        """Initialize with pre-computed Q'.

        Parameters
        ----------
        qprime : torch.Tensor
            Interpolated hourly lateral inflow, shape (timesteps, reaches)
        """
        super().__init__()
        self.qprime = qprime

    def forward(self, **_kwargs) -> torch.Tensor:
        """Return the pre-computed Q'."""
        return self.qprime


class MockKAN(torch.nn.Module):
    """Mock KAN module that returns fixed normalized parameters.

    All parameters are set to 0.5 (normalized midpoint), which maps to
    the geometric mean for log-space parameters and linear mean otherwise.
    """

    def __init__(self, num_reaches: int, learnable_params: list[str]) -> None:
        """Initialize with fixed parameter values.

        Parameters
        ----------
        num_reaches : int
            Number of river reaches
        learnable_params : list[str]
            List of parameter names to output (e.g., ["n", "q_spatial", "top_width", "side_slope"])
        """
        super().__init__()
        self.num_reaches = num_reaches
        self.learnable_params = learnable_params

    def forward(self, **_kwargs) -> dict[str, torch.Tensor]:
        """Return fixed normalized parameters (all 0.5)."""
        return {
            param: torch.full((self.num_reaches,), 0.5, dtype=torch.float32)
            for param in self.learnable_params
        }


# =============================================================================
# Helper Functions
# =============================================================================


def create_ddr_config():
    """Create a minimal Config for DDR routing tests.

    Returns
    -------
    ddr.validation.configs.Config
        Pydantic config with default DDR parameter ranges
    """
    from omegaconf import DictConfig

    from ddr.validation.configs import validate_config

    cfg_dict = {
        "name": "sandbox_test",
        "mode": "testing",
        "geodataset": "merit",
        "data_sources": {
            "geospatial_fabric_gpkg": "mock.gpkg",
            "streamflow": "mock://streamflow",
            "conus_adjacency": "mock.zarr",
        },
        "params": {
            "parameter_ranges": {
                "n": [0.015, 0.25],  # Manning's n (log space)
                "q_spatial": [0.0, 1.0],  # Shape factor (linear)
                "top_width": [1.0, 5000.0],  # Channel width (log space)
                "side_slope": [0.5, 50.0],  # Side slope (log space)
            },
            "log_space_parameters": ["n", "top_width", "side_slope"],
            "defaults": {"p_spatial": 21},
            "attribute_minimums": {
                "velocity": 0.01,
                "depth": 0.01,
                "discharge": 0.0001,
                "bottom_width": 0.1,
                "slope": 0.0001,
            },
            "tau": 3,
        },
        "kan": {
            "input_var_names": ["mock"],
            "learnable_parameters": ["n", "q_spatial", "top_width", "side_slope"],
        },
        "s3_region": "us-east-2",
        "device": "cpu",
    }
    return validate_config(DictConfig(cfg_dict), save_config=False)


def create_routing_dataclass(sandbox_zarr_path: Path, num_reaches: int = 5):
    """Create a RoutingDataclass for DDR routing with sandbox data.

    Parameters
    ----------
    sandbox_zarr_path : Path
        Path to zarr store containing adjacency matrix
    num_reaches : int
        Number of river reaches

    Returns
    -------
    RoutingDataclass
        Routing dataclass configured for sandbox network
    """
    from ddr_engine.merit import coo_from_zarr

    from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass

    # Load adjacency matrix from zarr and convert to torch sparse CSR
    coo, ts_order = coo_from_zarr(sandbox_zarr_path)
    coo_csr = coo.tocsr()

    # Convert scipy CSR to torch sparse CSR tensor
    crow_indices = torch.from_numpy(coo_csr.indptr).to(torch.int32)
    col_indices = torch.from_numpy(coo_csr.indices).to(torch.int32)
    values = torch.from_numpy(coo_csr.data).to(torch.float32)
    adjacency_matrix = torch.sparse_csr_tensor(
        crow_indices, col_indices, values, size=coo_csr.shape, dtype=torch.float32
    )

    # Create Dates object for the sandbox period
    dates = Dates(start_time="1970/01/01", end_time="1970/01/10")

    return RoutingDataclass(
        adjacency_matrix=adjacency_matrix,
        length=torch.full((num_reaches,), 5000.0),  # 5km reaches
        slope=torch.full((num_reaches,), 0.001),  # 1m per km
        x=torch.full((num_reaches,), 0.25),  # Storage coefficient from x_Sandbox.csv
        top_width=torch.empty(0),  # Learned via spatial_params
        side_slope=torch.empty(0),  # Learned via spatial_params
        divide_ids=np.array(ts_order),
        outflow_idx=None,  # Output all segments
        dates=dates,
        observations=None,
        gage_catchment=None,
    )


def run_ddr_routing(sandbox_zarr_path: Path, sandbox_hourly_qprime: torch.Tensor) -> np.ndarray:
    """Run DDR routing and return discharge output.

    Parameters
    ----------
    sandbox_zarr_path : Path
        Path to zarr store containing adjacency matrix
    sandbox_hourly_qprime : torch.Tensor
        Interpolated hourly Q', shape (timesteps, reaches)

    Returns
    -------
    np.ndarray
        Discharge output, shape (reaches, timesteps)
    """
    from ddr import dmc

    num_reaches = 5
    learnable_params = ["n", "q_spatial", "top_width", "side_slope"]

    # Create components
    cfg = create_ddr_config()
    routing_dataclass = create_routing_dataclass(sandbox_zarr_path, num_reaches)

    mock_streamflow = MockStreamflow(sandbox_hourly_qprime)
    mock_kan = MockKAN(num_reaches, learnable_params)
    routing_model = dmc(cfg=cfg, device="cpu")

    # Get Q' from mock streamflow
    qprime = mock_streamflow()

    # Get spatial params from mock KAN
    spatial_params = mock_kan()

    # Run DDR routing
    routing_model.set_progress_info(epoch=0, mini_batch=0)
    dmc_kwargs = {
        "routing_dataclass": routing_dataclass,
        "spatial_parameters": spatial_params,
        "streamflow": qprime,
    }
    ddr_output = routing_model(**dmc_kwargs)
    return ddr_output["runoff"].detach().numpy()


# =============================================================================
# Tests
# =============================================================================


def test_ddr_routing_output_shape(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that DDR routing produces output with correct shape."""
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    num_reaches = 5
    num_timesteps = sandbox_hourly_qprime.shape[0]

    assert ddr_discharge.shape == (num_reaches, num_timesteps), (
        f"Expected shape ({num_reaches}, {num_timesteps}), got {ddr_discharge.shape}"
    )


def test_ddr_routing_non_negative(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that DDR routing produces non-negative discharge values."""
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    assert (ddr_discharge >= 0).all(), "DDR produced negative discharge values"


def test_ddr_routing_no_nan_inf(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that DDR routing produces no NaN or Inf values."""
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    assert not np.isnan(ddr_discharge).any(), "DDR produced NaN values"
    assert not np.isinf(ddr_discharge).any(), "DDR produced Inf values"


def test_ddr_routing_downstream_accumulation(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that discharge accumulates downstream after spin-up.

    Network topology:
    - Reaches 10, 20 flow into reach 30 (confluence)
    - Reaches 30, 40 flow into reach 50 (outlet)
    """
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    # Skip initial spin-up timesteps
    spin_up = 50
    ddr_q30 = ddr_discharge[2, spin_up:].mean()  # Reach 30 (confluence)
    ddr_q40 = ddr_discharge[3, spin_up:].mean()  # Reach 40
    ddr_q50 = ddr_discharge[4, spin_up:].mean()  # Reach 50 (outlet)

    # Outlet should accumulate flow from upstream
    assert ddr_q50 >= ddr_q30, f"Outlet Q50 ({ddr_q50:.2f}) should be >= Q30 ({ddr_q30:.2f})"
    assert ddr_q50 >= ddr_q40, f"Outlet Q50 ({ddr_q50:.2f}) should be >= Q40 ({ddr_q40:.2f})"


def test_ddr_routing_positive_flow(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that DDR produces meaningful positive flow at all reaches."""
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    # After spin-up, all reaches should have positive mean discharge
    spin_up = 50
    for i, reach_id in enumerate(REACH_IDS):
        mean_q = ddr_discharge[i, spin_up:].mean()
        assert mean_q > 0, f"Reach {reach_id} has non-positive mean discharge: {mean_q}"


@pytest.fixture
def ddr_discharge_output(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> np.ndarray:
    """Fixture providing DDR discharge output for use in other tests.

    Returns
    -------
    np.ndarray
        Discharge output, shape (reaches, timesteps)
    """
    return run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)
