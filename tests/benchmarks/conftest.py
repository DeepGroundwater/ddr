"""Pytest configuration and fixtures for benchmarks tests."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
import zarr
from scipy import sparse
from shapely.geometry import LineString

pytest.importorskip("ddr_benchmarks")
ddr_engine = pytest.importorskip("ddr_engine")

from ddr_engine.merit import (  # noqa: E402
    build_merit_adjacency,
    create_adjacency_matrix,
)

# =============================================================================
# Paths
# =============================================================================

TESTS_DIR = Path(__file__).parent.parent
SANDBOX_DIR = TESTS_DIR / "input" / "Sandbox"


# =============================================================================
# Sandbox Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_connections() -> pd.DataFrame:
    """Load raw RAPID Sandbox connectivity CSV."""
    return pd.read_csv(SANDBOX_DIR / "rapid_connect_Sandbox.csv", header=None)


def _sandbox_to_merit_format(sandbox_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert RAPID Sandbox connectivity CSV to MERIT-compatible GeoDataFrame."""
    sandbox_df = sandbox_df.copy()
    sandbox_df.columns = ["COMID", "NextDownID"]

    # Build upstream lookup
    upstream_lookup: dict[int, list[int]] = {}
    for _, row in sandbox_df.iterrows():
        comid = int(row["COMID"])
        next_down = int(row["NextDownID"])
        if next_down != 0:
            upstream_lookup.setdefault(next_down, []).append(comid)

    # Build MERIT-style records
    records = []
    for _, row in sandbox_df.iterrows():
        comid = int(row["COMID"])
        next_down = int(row["NextDownID"])
        upstreams = upstream_lookup.get(comid, [])

        records.append(
            {
                "COMID": comid,
                "NextDownID": next_down,
                "up1": upstreams[0] if len(upstreams) > 0 else 0,
                "up2": upstreams[1] if len(upstreams) > 1 else 0,
                "up3": upstreams[2] if len(upstreams) > 2 else 0,
                "up4": upstreams[3] if len(upstreams) > 3 else 0,
            }
        )

    df = pd.DataFrame(records)
    df["geometry"] = [LineString([(0, i), (1, i)]) for i in range(len(df))]

    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


@pytest.fixture(scope="session")
def mock_merit_fp(sandbox_connections: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert Sandbox connectivity to MERIT-compatible GeoDataFrame."""
    return _sandbox_to_merit_format(sandbox_connections)


# =============================================================================
# Zarr Store Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_zarr_path(tmp_path_factory: pytest.TempPathFactory, mock_merit_fp: gpd.GeoDataFrame) -> Path:
    """Build and persist Sandbox adjacency matrix to zarr store."""
    tmp_dir: Path = tmp_path_factory.mktemp("sandbox_zarr")
    out_path = tmp_dir / "sandbox_adjacency.zarr"
    build_merit_adjacency(mock_merit_fp, out_path)
    return out_path


@pytest.fixture(scope="session")
def sandbox_zarr_root(sandbox_zarr_path: Path) -> zarr.Group:
    """Open the Sandbox zarr store as a zarr Group."""
    return zarr.open_group(store=sandbox_zarr_path, mode="r")


@pytest.fixture(scope="session")
def sandbox_zarr_order(sandbox_zarr_root: zarr.Group) -> Any:
    """Extract topological order from zarr store."""
    order: np.ndarray = sandbox_zarr_root["order"][:]
    return order.tolist()


# =============================================================================
# Matrix Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_matrix(mock_merit_fp: gpd.GeoDataFrame) -> Any:
    """Create adjacency matrix from Sandbox network."""
    return create_adjacency_matrix(mock_merit_fp)


@pytest.fixture(scope="session")
def sandbox_coo_matrix(
    sandbox_matrix: tuple[sparse.coo_matrix, list[int]],
) -> sparse.coo_matrix:
    """Extract just the COO matrix."""
    return sandbox_matrix[0]


@pytest.fixture(scope="session")
def sandbox_ts_order(sandbox_matrix: tuple[sparse.coo_matrix, list[int]]) -> list[int]:
    """Extract just the topological order."""
    return sandbox_matrix[1]


# =============================================================================
# RAPID2 Reference Data Fixtures
# =============================================================================

TESTS_OUTPUT_DIR = TESTS_DIR / "output"


@pytest.fixture(scope="session")
def sandbox_qinit() -> torch.Tensor:
    """Load initial discharge conditions from RAPID Sandbox.

    Returns
    -------
    torch.Tensor
        Initial discharge state: [9, 9, 27, 18, 63] m³/s for reaches [10, 20, 30, 40, 50]
    """
    ds = xr.open_dataset(SANDBOX_DIR / "Qinit_Sandbox_19700101_19700110.nc4")
    qinit = ds["Qout"].values.squeeze()  # (5,)
    ds.close()
    return torch.from_numpy(qinit).float()


@pytest.fixture(scope="session")
def sandbox_expected_qout() -> torch.Tensor:
    """Load RAPID2 reference discharge output.

    Returns
    -------
    torch.Tensor
        Expected discharge time series, shape (80, 5) for (time, reaches)
    """
    ds = xr.open_dataset(TESTS_OUTPUT_DIR / "Sandbox" / "Qout_Sandbox_19700101_19700110.nc4")
    qout = ds["Qout"].values  # (80, 5)
    ds.close()
    return torch.from_numpy(qout).float()


@pytest.fixture(scope="session")
def sandbox_expected_qfinal() -> torch.Tensor:
    """Load RAPID2 final discharge state.

    Returns
    -------
    torch.Tensor
        Final discharge state, shape (5,) for reaches [10, 20, 30, 40, 50]
    """
    ds = xr.open_dataset(TESTS_OUTPUT_DIR / "Sandbox" / "Qfinal_Sandbox_19700101_19700110.nc4")
    qfinal = ds["Qout"].values.squeeze()  # (5,)
    ds.close()
    return torch.from_numpy(qfinal).float()


@pytest.fixture(scope="session")
def sandbox_qext() -> torch.Tensor:
    """Load RAPID Sandbox lateral inflow (Qext).

    Returns
    -------
    torch.Tensor
        Lateral inflow, shape (80, 5) for (time, reaches)
    """
    ds = xr.open_dataset(SANDBOX_DIR / "Qext_Sandbox_19700101_19700110.nc4")
    qext = ds["Qext"].values  # (80, 5)
    ds.close()
    return torch.from_numpy(qext).float()


@pytest.fixture(scope="session")
def sandbox_hourly_qprime(sandbox_qext: torch.Tensor) -> torch.Tensor:
    """Interpolate Qext from 3-hourly (80 timesteps) to hourly.

    Linear interpolation from 80 timesteps to 238 hourly timesteps.
    Original 80 points cover hours [0, 3, 6, ..., 237].
    Interpolated points cover hours [0, 1, 2, ..., 237].

    Returns
    -------
    torch.Tensor
        Interpolated Q', shape (238, 5) for (time, reaches)
    """
    from scipy.interpolate import interp1d

    qext_np = sandbox_qext.numpy()  # (80, 5)
    n_original = qext_np.shape[0]  # 80
    n_reaches = qext_np.shape[1]  # 5

    # Original time points (3-hourly): 0, 3, 6, ..., 237
    t_original = np.arange(n_original) * 3  # [0, 3, 6, ..., 237]

    # Target time points (hourly): 0, 1, 2, ..., 237
    t_hourly = np.arange(t_original[-1] + 1)  # [0, 1, ..., 237] = 238 points

    # Interpolate each reach
    qprime_hourly = np.zeros((len(t_hourly), n_reaches), dtype=np.float32)
    for i in range(n_reaches):
        f = interp1d(t_original, qext_np[:, i], kind="linear")
        qprime_hourly[:, i] = f(t_hourly)

    return torch.from_numpy(qprime_hourly).float()


# =============================================================================
# DiffRoute Network Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_network() -> tuple[nx.DiGraph, pd.DataFrame]:
    """Load RAPID Sandbox network topology as NetworkX DiGraph and params DataFrame.

    Returns
    -------
    tuple[nx.DiGraph, pd.DataFrame]
        Tuple of (graph, param_df) where param_df has Muskingum parameters indexed by reach ID.
    """
    # Read connectivity: columns are [COMID, NextDownID]
    connect_df = pd.read_csv(
        SANDBOX_DIR / "rapid_connect_Sandbox.csv", header=None, names=["comid", "next_down"]
    )

    # Read reach IDs
    reach_ids = pd.read_csv(SANDBOX_DIR / "riv_bas_id_Sandbox.csv", header=None).squeeze().tolist()

    # Muskingum parameters: k=0.1042 days (9000s, RAPID default), x=0.3
    k_days = 0.1042
    x_val = 0.3

    # Build graph (upstream -> downstream)
    G = nx.DiGraph()
    for rid in reach_ids:
        G.add_node(
            rid,
            k=k_days,
            x=x_val,
            tau=k_days,  # tau for linear_storage
            delay=k_days,  # delay for pure_lag
        )

    # Add edges (from COMID to NextDownID, but 0 means outlet)
    for _, row in connect_df.iterrows():
        if row["next_down"] != 0:  # 0 means outlet
            G.add_edge(row["comid"], row["next_down"])

    # Build param_df indexed by reach ID with Muskingum parameters (all in days)
    param_df = pd.DataFrame(
        {
            "k": [k_days] * len(reach_ids),
            "x": [x_val] * len(reach_ids),
            "tau": [k_days] * len(reach_ids),
            "delay": [k_days] * len(reach_ids),
        },
        index=reach_ids,
    )

    return G, param_df


@pytest.fixture(scope="session")
def sandbox_runoff(sandbox_qext: torch.Tensor) -> torch.Tensor:
    """Load RAPID Sandbox runoff data as PyTorch tensor for DiffRoute.

    DiffRoute expects input shape (batch, nodes, time).

    Returns
    -------
    torch.Tensor
        Runoff tensor, shape (1, 5, 80) for (batch, reaches, time)
    """
    # DiffRoute expects (batch, nodes, time)
    # sandbox_qext is (80, 5), so transpose and add batch dimension
    runoff = sandbox_qext.T.unsqueeze(0).float()  # (1, 5, 80)
    return runoff


@pytest.fixture(scope="session")
def sandbox_hourly_runoff(sandbox_hourly_qprime: torch.Tensor) -> torch.Tensor:
    """Interpolated hourly runoff data as PyTorch tensor for DiffRoute.

    DiffRoute expects input shape (batch, nodes, time).

    Returns
    -------
    torch.Tensor
        Runoff tensor, shape (1, 5, 238) for (batch, reaches, time)
    """
    # DiffRoute expects (batch, nodes, time)
    # sandbox_hourly_qprime is (238, 5), so transpose and add batch dimension
    runoff = sandbox_hourly_qprime.T.unsqueeze(0).float()  # (1, 5, 238)
    return runoff


# =============================================================================
# DDR Mock Classes
# =============================================================================

# Reach IDs in the Sandbox network (RAPID2 ordering)
RAPID2_REACH_IDS = [10, 20, 30, 40, 50]


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
# DDR Helper Functions
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
            "forcings": "mock://forcings",
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
            "learnable_parameters": ["q_spatial", "top_width", "side_slope"],
        },
        "cuda_lstm": {
            "input_var_names": ["mock"],
            "learnable_parameters": ["n"],
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
    tuple[RoutingDataclass, list[int]]
        Routing dataclass configured for sandbox network and the topological order
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

    routing_dataclass = RoutingDataclass(
        adjacency_matrix=adjacency_matrix,
        length=torch.full((num_reaches,), 5000.0),  # 5km reaches
        slope=torch.full((num_reaches,), 0.001),  # 1m per km
        x=torch.full((num_reaches,), 0.25),  # Muskingum x weighting factor
        top_width=torch.empty(0),  # Learned via spatial_params
        side_slope=torch.empty(0),  # Learned via spatial_params
        divide_ids=np.array(ts_order),
        outflow_idx=None,  # Output all segments
        dates=dates,
        observations=None,
        gage_catchment=None,
    )

    return routing_dataclass, ts_order


def reorder_ddr_to_rapid2(ddr_output: np.ndarray, ts_order: list[int]) -> np.ndarray:
    """Reorder DDR output from topological order to RAPID2 order [10, 20, 30, 40, 50].

    Parameters
    ----------
    ddr_output : np.ndarray
        DDR discharge output in topological order, shape (reaches, timesteps)
    ts_order : list[int]
        Topological order of reach IDs from zarr store

    Returns
    -------
    np.ndarray
        Discharge reordered to RAPID2 order [10, 20, 30, 40, 50]
    """
    # Build mapping: for each RAPID2 position, find the index in ts_order
    reorder_idx = [ts_order.index(rid) for rid in RAPID2_REACH_IDS]
    return ddr_output[reorder_idx, :]


def run_ddr_routing(sandbox_zarr_path: Path, sandbox_hourly_qprime: torch.Tensor) -> np.ndarray:
    """Run DDR routing and return discharge output in RAPID2 order.

    Parameters
    ----------
    sandbox_zarr_path : Path
        Path to zarr store containing adjacency matrix
    sandbox_hourly_qprime : torch.Tensor
        Interpolated hourly Q', shape (timesteps, reaches) in RAPID2 order

    Returns
    -------
    np.ndarray
        Discharge output in RAPID2 order [10, 20, 30, 40, 50], shape (5, timesteps)
    """
    from ddr import dmc

    num_reaches = 5
    learnable_params = ["q_spatial", "top_width", "side_slope"]

    # Create components
    cfg = create_ddr_config()
    routing_dataclass, ts_order = create_routing_dataclass(sandbox_zarr_path, num_reaches)

    # Reorder Q' from RAPID2 order to topological order for DDR input
    qprime_rapid2 = sandbox_hourly_qprime.numpy()  # (timesteps, 5) in RAPID2 order
    topo_idx = [RAPID2_REACH_IDS.index(rid) for rid in ts_order]
    qprime_topo = qprime_rapid2[:, topo_idx]  # Reorder to topo order
    qprime_tensor = torch.from_numpy(qprime_topo).float()

    mock_streamflow = MockStreamflow(qprime_tensor)
    mock_kan = MockKAN(num_reaches, learnable_params)
    routing_model = dmc(cfg=cfg, device="cpu")

    # Get Q' from mock streamflow
    qprime = mock_streamflow()

    # Get spatial params from mock KAN
    spatial_params = mock_kan()

    # Create mock LSTM params (n from LSTM, time-varying)
    num_timesteps = qprime.shape[0]
    T_daily = max(num_timesteps // 24, 1)
    lstm_params = {
        "n": torch.full((T_daily, num_reaches), 0.5, dtype=torch.float32),
    }

    # Run DDR routing
    routing_model.set_progress_info(epoch=0, mini_batch=0)
    dmc_kwargs = {
        "routing_dataclass": routing_dataclass,
        "spatial_parameters": spatial_params,
        "streamflow": qprime,
        "lstm_params": lstm_params,
    }
    ddr_output = routing_model(**dmc_kwargs)
    ddr_discharge_topo = ddr_output["runoff"].detach().numpy()  # (reaches, timesteps) in topo order

    # Reorder output from topological order to RAPID2 order
    ddr_discharge_rapid2 = reorder_ddr_to_rapid2(ddr_discharge_topo, ts_order)

    return ddr_discharge_rapid2


# =============================================================================
# DDR Routing Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ddr_discharge(sandbox_zarr_path: Path, sandbox_hourly_qprime: torch.Tensor) -> np.ndarray:
    """Run DDR routing and return discharge output in RAPID2 order.

    This fixture provides DDR discharge for use in comparison plots.

    Returns
    -------
    np.ndarray
        Discharge output in RAPID2 order [10, 20, 30, 40, 50], shape (5, 238)
    """
    return run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)
