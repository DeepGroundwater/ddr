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

    # Read Muskingum parameters
    k_vals = pd.read_csv(SANDBOX_DIR / "k_Sandbox.csv", header=None).squeeze().tolist()
    x_vals = pd.read_csv(SANDBOX_DIR / "x_Sandbox.csv", header=None).squeeze().tolist()

    # Build graph (upstream -> downstream)
    G = nx.DiGraph()
    for i, rid in enumerate(reach_ids):
        # Store parameters on nodes for compatibility
        G.add_node(
            rid,
            k=float(k_vals[i]),
            x=float(x_vals[i]),
            tau=float(k_vals[i]),  # tau for linear_storage
            delay=float(k_vals[i]),  # delay for pure_lag
        )

    # Add edges (from COMID to NextDownID, but 0 means outlet)
    for _, row in connect_df.iterrows():
        if row["next_down"] != 0:  # 0 means outlet
            G.add_edge(row["comid"], row["next_down"])

    # Build param_df indexed by reach ID with Muskingum parameters
    # DiffRoute expects k in days (RAPID k is in seconds)
    param_df = pd.DataFrame(
        {
            "k": [k / (3600 * 24) for k in k_vals],  # Convert seconds to days
            "x": x_vals,
            "tau": [k / (3600 * 24) for k in k_vals],  # tau for linear_storage
            "delay": [k / (3600 * 24) for k in k_vals],  # delay for pure_lag
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


# =============================================================================
# DDR Routing Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def ddr_discharge(sandbox_zarr_path: Path, sandbox_hourly_qprime: torch.Tensor) -> np.ndarray:
    """Run DDR routing and return discharge output.

    This fixture provides DDR discharge for use in comparison plots.

    Returns
    -------
    np.ndarray
        Discharge output, shape (5, 238) for (reaches, timesteps)
    """
    from tests.benchmarks.test_ddr import run_ddr_routing

    return run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)
