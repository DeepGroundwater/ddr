"""Fixtures and necessary imports for integration tests"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from ddr_engine.merit import (
    build_graph,
    build_merit_adjacency,
    build_upstream_dict,
    create_adjacency_matrix,
    # coo_from_zarr,
    # subset_upstream,
)
from scipy import sparse
from shapely.geometry import LineString

# =============================================================================
# Sandbox Network Reference
# =============================================================================
# Network topology (from SANDBOX.md):
#
#     10 ──┐
#          ├──► 30 ──┐
#     20 ──┘         │
#                    ├──► 50 ──► (outlet)
#     40 ────────────┘
#
# Network matrix N (row=downstream, col=upstream):
#     [ 0  0  0  0  0 ]   <- 10
#     [ 0  0  0  0  0 ]   <- 20
#     [ 1  1  0  0  0 ]   <- 30 (receives from 10, 20)
#     [ 0  0  0  0  0 ]   <- 40
#     [ 0  0  1  1  0 ]   <- 50 (receives from 30, 40)
#
# Gauging stations at COMIDs 30 and 50
# =============================================================================


# =============================================================================
# Mock Gauge Dataclass (mimics ddr.geodatazoo.dataclasses)
# =============================================================================


@dataclass
class MockGauge:
    """Mock gauge matching MERITGauge interface."""

    STAID: str
    COMID: int


@dataclass
class MockGaugeSet:
    """Mock gauge set matching GaugeSet interface."""

    gauges: list[MockGauge]


# =============================================================================
# Sandbox Data Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_connections() -> pd.DataFrame:
    """Load raw RAPID Sandbox connectivity CSV."""
    cwd = Path(__file__).parent
    return pd.read_csv(cwd / "input/Sandbox/rapid_connect_Sandbox.csv", header=None)


def sandbox_to_merit_format(sandbox_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert RAPID Sandbox connectivity CSV to MERIT-compatible GeoDataFrame.

    Parameters
    ----------
    sandbox_df : pd.DataFrame
        DataFrame with columns [current_comid, next_down_comid].
        NextDownID=0 indicates outlet.

    Returns
    -------
    gpd.GeoDataFrame
        MERIT-compatible GeoDataFrame with COMID, NextDownID, up1-up4, geometry.
    """
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
    return sandbox_to_merit_format(sandbox_connections)


# =============================================================================
# Mock Gauge Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_gauge_set() -> MockGaugeSet:
    """
    Create mock gauge set for Sandbox network.

    From SANDBOX.md, gauging stations are at positions 3 and 5 in the
    selection matrix, corresponding to COMIDs 30 and 50.
    """
    return MockGaugeSet(
        gauges=[
            MockGauge(STAID="00000030", COMID=30),
            MockGauge(STAID="00000050", COMID=50),
        ]
    )


@pytest.fixture(scope="session")
def sandbox_gauge_csv(tmp_path_factory: pytest.TempPathFactory, sandbox_gauge_set: MockGaugeSet) -> Path:
    """Write mock gauge set to CSV file."""
    tmp_dir: Path = tmp_path_factory.mktemp("sandbox_gages")
    csv_path = tmp_dir / "sandbox_gages.csv"

    records = [{"STAID": g.STAID, "COMID": g.COMID} for g in sandbox_gauge_set.gauges]
    pd.DataFrame(records).to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture(scope="session")
def sandbox_zarr_path(tmp_path_factory: pytest.TempPathFactory, mock_merit_fp: gpd.GeoDataFrame) -> Path:
    """
    Build and persist Sandbox adjacency matrix to zarr store.

    This runs the full build_merit_adjacency pipeline and returns
    the path to the zarr store for use in downstream tests.
    """
    tmp_dir: Path = tmp_path_factory.mktemp("sandbox_zarr")
    out_path = tmp_dir / "sandbox_adjacency.zarr"

    build_merit_adjacency(mock_merit_fp, out_path)

    return out_path


@pytest.fixture(scope="session")
def sandbox_zarr_order(sandbox_zarr_root: zarr.Group) -> Any:
    """Extract topological order from zarr store."""
    order: np.ndarray = sandbox_zarr_root["order"][:]
    return order.tolist()


# =============================================================================
# Graph Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_upstream_dict(mock_merit_fp: gpd.GeoDataFrame) -> Any:
    """Build upstream dictionary from Sandbox network."""
    return build_upstream_dict(mock_merit_fp)


@pytest.fixture(scope="session")
def sandbox_graph(sandbox_upstream_dict: dict[int, list[int]]):
    """Build RustWorkX graph from Sandbox network."""
    return build_graph(sandbox_upstream_dict)


@pytest.fixture(scope="session")
def G(sandbox_graph):
    """Extract just the graph object."""
    return sandbox_graph[0]


@pytest.fixture(scope="session")
def sandbox_node_indices(sandbox_graph):
    """Extract just the node indices mapping."""
    return sandbox_graph[1]


# =============================================================================
# Matrix Fixtures (in-memory)
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_matrix(mock_merit_fp: gpd.GeoDataFrame) -> Any:
    """Create adjacency matrix from Sandbox network."""
    return create_adjacency_matrix(mock_merit_fp)


@pytest.fixture(scope="session")
def sandbox_coo_matrix(sandbox_matrix: tuple[sparse.coo_matrix, list[int]]) -> sparse.coo_matrix:
    """Extract just the COO matrix."""
    return sandbox_matrix[0]


@pytest.fixture(scope="session")
def sandbox_ts_order(sandbox_matrix: tuple[sparse.coo_matrix, list[int]]) -> list[int]:
    """Extract just the topological order."""
    return sandbox_matrix[1]


@pytest.fixture(scope="session")
def sandbox_comid_to_idx(sandbox_ts_order: list[int]) -> dict[int, int]:
    """Mapping from COMID to matrix index."""
    return {comid: idx for idx, comid in enumerate(sandbox_ts_order)}


# =============================================================================
# Expected Network Matrix (from SANDBOX.md)
# =============================================================================


@pytest.fixture(scope="session")
def expected_network_matrix() -> np.ndarray:
    """
    Expected network matrix N from SANDBOX.md.

    Rows/cols ordered as [10, 20, 30, 40, 50].
    N[i,j] = 1 means j flows into i.
    """
    return np.array(
        [
            [0, 0, 0, 0, 0],  # 10: no upstream
            [0, 0, 0, 0, 0],  # 20: no upstream
            [1, 1, 0, 0, 0],  # 30: receives from 10, 20
            [0, 0, 0, 0, 0],  # 40: no upstream
            [0, 0, 1, 1, 0],  # 50: receives from 30, 40
        ],
        dtype=np.uint8,
    )


# =============================================================================
# Zarr Store Fixtures (persisted)
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_zarr_root(sandbox_zarr_path: Path) -> zarr.Group:
    """Open the Sandbox zarr store as a zarr Group."""
    return zarr.open_group(store=sandbox_zarr_path, mode="r")


@pytest.fixture(scope="session")
def sandbox_zarr_coo(sandbox_zarr_root: zarr.Group) -> sparse.coo_matrix:
    """Reconstruct COO matrix from zarr store."""
    row = sandbox_zarr_root["indices_0"][:]
    col = sandbox_zarr_root["indices_1"][:]
    data = sandbox_zarr_root["values"][:]
    shape = tuple(sandbox_zarr_root.attrs["shape"])

    return sparse.coo_matrix((data, (row, col)), shape=shape)


@pytest.fixture(scope="session")
def sandbox_zarr_comid_to_idx(sandbox_zarr_order: list[int]) -> dict[int, int]:
    """Mapping from COMID to matrix index (from zarr)."""
    return {comid: idx for idx, comid in enumerate(sandbox_zarr_order)}
