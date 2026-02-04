"""Pytest configuration and fixtures for benchmarks tests."""

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
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
