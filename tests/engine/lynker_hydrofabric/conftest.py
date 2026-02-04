"""Fixtures for Lynker Hydrofabric integration tests."""

import pytest

pytest.importorskip("ddr_engine")

from typing import Any

import polars as pl
from ddr_engine.lynker_hydrofabric import (
    build_graph_from_wb_network,
    create_coo,
    preprocess_river_network,
    subset,
)

# =============================================================================
# Sandbox Network Reference
# =============================================================================
# Network topology (similar to MERIT sandbox):
#
#     wb-10 ──┐
#             ├──► wb-30 ──┐
#     wb-20 ──┘            │
#                          ├──► wb-50 ──► (outlet)
#     wb-40 ───────────────┘
#
# Network connections via nexuses:
# - wb-10 -> nex-10 -> wb-30
# - wb-20 -> nex-20 -> wb-30
# - wb-30 -> nex-30 -> wb-50
# - wb-40 -> nex-40 -> wb-50
# =============================================================================


@pytest.fixture(scope="session")
def sandbox_network() -> pl.LazyFrame:
    """
    Create a sandbox network LazyFrame for testing.

    This simulates the hydrofabric network table structure with
    waterbody (wb-*) and nexus (nex-*) connections.
    """
    data = {
        "id": [
            "wb-10",
            "wb-20",
            "wb-30",
            "wb-40",
            "wb-50",
            "nex-10",
            "nex-20",
            "nex-30",
            "nex-40",
        ],
        "toid": [
            "nex-10",
            "nex-20",
            "nex-30",
            "nex-40",
            None,  # wb-50 is outlet
            "wb-30",
            "wb-30",
            "wb-50",
            "wb-50",
        ],
        "hl_uri": [
            None,
            None,
            "gages-00000030",  # gauge at wb-30
            None,
            "gages-00000050",  # gauge at wb-50
            None,
            None,
            None,
            None,
        ],
    }
    return pl.DataFrame(data).lazy()


@pytest.fixture(scope="session")
def sandbox_wb_network_dict(sandbox_network: pl.LazyFrame) -> Any:
    """Build wb_network_dict from sandbox network."""
    return preprocess_river_network(sandbox_network)


@pytest.fixture(scope="session")
def sandbox_graph(sandbox_wb_network_dict: dict[str, list[str]]) -> Any:
    """Build RustWorkX graph from sandbox network."""
    return build_graph_from_wb_network(sandbox_wb_network_dict)


@pytest.fixture(scope="session")
def sandbox_connections(sandbox_wb_network_dict: dict[str, list[str]]) -> Any:
    """Get connections from wb-50 (outlet)."""
    return subset("wb-50", sandbox_wb_network_dict)


@pytest.fixture(scope="session")
def sandbox_conus_mapping() -> dict[str, int]:
    """Create a mock CONUS mapping for testing."""
    # Sorted order for determinism
    return {
        "wb-10": 0,
        "wb-20": 1,
        "wb-30": 2,
        "wb-40": 3,
        "wb-50": 4,
    }


@pytest.fixture(scope="session")
def sandbox_coo_result(
    sandbox_connections: list[tuple[str, str]],
    sandbox_conus_mapping: dict[str, int],
) -> Any:
    """Create COO matrix from sandbox connections."""
    return create_coo(sandbox_connections, sandbox_conus_mapping)
