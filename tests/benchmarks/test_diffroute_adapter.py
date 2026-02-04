"""Tests for the DiffRoute adapter module.

These tests verify that COO adjacency matrices from DDR Engine can be
correctly converted to NetworkX graphs for use with DiffRoute.

Tests use fixtures from tests/conftest.py to get real
zarr stores created by the Engine pipeline.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
import zarr
from ddr_benchmarks.diffroute_adapter import (
    build_diffroute_inputs,
    coo_to_networkx,
    create_param_df,
    load_rapid_params,
    zarr_group_to_networkx,
    zarr_to_networkx,
)

# Path to RAPID Sandbox test data
SANDBOX_DIR = Path(__file__).parent.parent / "input" / "Sandbox"

# DiffRoute requires CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffRoute requires CUDA")


# =============================================================================
# Sandbox Network Reference (from tests/integration/conftest.py)
# =============================================================================
# Network topology:
#
#     10 ──┐
#          ├──► 30 ──┐
#     20 ──┘         │
#                    ├──► 50 ──► (outlet)
#     40 ────────────┘
#
# COO Matrix (row=downstream, col=upstream):
#     [ 0  0  0  0  0 ]   <- 10
#     [ 0  0  0  0  0 ]   <- 20
#     [ 1  1  0  0  0 ]   <- 30 (receives from 10, 20)
#     [ 0  0  0  0  0 ]   <- 40
#     [ 0  0  1  1  0 ]   <- 50 (receives from 30, 40)
# =============================================================================


# Import fixtures from root conftest (tests/conftest.py provides sandbox fixtures)


# =============================================================================
# Tests for zarr_group_to_networkx (primary function)
# =============================================================================


def test_zarr_group_to_networkx_creates_graph(sandbox_zarr_root: zarr.Group) -> None:
    """Verify zarr_group_to_networkx creates a valid NetworkX graph from zarr Group."""
    G = zarr_group_to_networkx(sandbox_zarr_root)

    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes()) == 5
    assert len(G.edges()) == 4


def test_zarr_group_to_networkx_node_ids(sandbox_zarr_root: zarr.Group) -> None:
    """Verify nodes use segment IDs from order array."""
    G = zarr_group_to_networkx(sandbox_zarr_root)

    expected_nodes = {10, 20, 30, 40, 50}
    assert set(G.nodes()) == expected_nodes


def test_zarr_group_to_networkx_edge_direction(sandbox_zarr_root: zarr.Group) -> None:
    """Verify edges go from upstream to downstream."""
    G = zarr_group_to_networkx(sandbox_zarr_root)

    # Expected edges: upstream → downstream
    expected_edges = {(10, 30), (20, 30), (30, 50), (40, 50)}
    assert set(G.edges()) == expected_edges


def test_zarr_group_to_networkx_topology(sandbox_zarr_root: zarr.Group) -> None:
    """Verify graph topology matches sandbox network."""
    G = zarr_group_to_networkx(sandbox_zarr_root)

    # Check predecessors (upstream of each node)
    assert set(G.predecessors(30)) == {10, 20}
    assert set(G.predecessors(50)) == {30, 40}
    assert set(G.predecessors(10)) == set()
    assert set(G.predecessors(20)) == set()
    assert set(G.predecessors(40)) == set()

    # Check successors (downstream of each node)
    assert set(G.successors(10)) == {30}
    assert set(G.successors(20)) == {30}
    assert set(G.successors(30)) == {50}
    assert set(G.successors(40)) == {50}
    assert set(G.successors(50)) == set()  # outlet


# =============================================================================
# Tests for zarr_to_networkx (path-based convenience function)
# =============================================================================


def test_zarr_to_networkx(sandbox_zarr_path: Path) -> None:
    """Verify zarr_to_networkx loads and converts correctly from path."""
    G = zarr_to_networkx(sandbox_zarr_path)

    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes()) == 5
    assert len(G.edges()) == 4
    assert set(G.nodes()) == {10, 20, 30, 40, 50}


# =============================================================================
# Tests for coo_to_networkx (scipy sparse matrix)
# =============================================================================


def test_coo_to_networkx_creates_graph(
    sandbox_coo_matrix,
    sandbox_ts_order: list[int],
) -> None:
    """Verify coo_to_networkx creates a valid NetworkX graph."""
    order = np.array(sandbox_ts_order, dtype=np.int32)
    G = coo_to_networkx(sandbox_coo_matrix, order)

    assert isinstance(G, nx.DiGraph)
    assert len(G.nodes()) == 5
    assert len(G.edges()) == 4


# =============================================================================
# Tests for create_param_df
# =============================================================================


def test_create_param_df_with_muskingum(sandbox_zarr_root: zarr.Group) -> None:
    """Verify param_df creation with Muskingum parameters."""
    order = sandbox_zarr_root["order"][:]
    k_seconds = [9000.0] * 5  # 9000 seconds = 0.104167 days
    x_vals = [0.25] * 5

    param_df = create_param_df(order, k=k_seconds, x=x_vals, k_units="seconds")

    assert isinstance(param_df, pd.DataFrame)
    assert sorted(param_df.index) == [10, 20, 30, 40, 50]
    assert "k" in param_df.columns
    assert "x" in param_df.columns
    assert "tau" in param_df.columns  # Derived from k
    assert "delay" in param_df.columns  # Derived from k

    # Check k was converted to days
    expected_k_days = 9000.0 / (3600 * 24)
    np.testing.assert_allclose(param_df["k"].values, expected_k_days)
    np.testing.assert_array_equal(param_df["x"].values, 0.25)


def test_create_param_df_derives_tau_from_k(sandbox_zarr_root: zarr.Group) -> None:
    """Verify tau is derived from k when not provided."""
    order = sandbox_zarr_root["order"][:]
    k_days = [0.1] * 5

    param_df = create_param_df(order, k=k_days, x=[0.3] * 5, k_units="days")

    np.testing.assert_array_equal(param_df["tau"].values, param_df["k"].values)
    np.testing.assert_array_equal(param_df["delay"].values, param_df["k"].values)


def test_create_param_df_explicit_tau(sandbox_zarr_root: zarr.Group) -> None:
    """Verify explicit tau overrides derived value."""
    order = sandbox_zarr_root["order"][:]
    k_days = [0.1] * 5
    tau_vals = [0.2] * 5

    param_df = create_param_df(order, k=k_days, x=[0.3] * 5, tau=tau_vals, k_units="days")

    np.testing.assert_array_equal(param_df["tau"].values, 0.2)
    np.testing.assert_array_equal(param_df["k"].values, 0.1)


# =============================================================================
# Tests for load_rapid_params
# =============================================================================


def test_load_rapid_params() -> None:
    """Verify loading parameters from RAPID-style CSV files."""
    reach_ids, k_vals, x_vals = load_rapid_params(
        k_file=SANDBOX_DIR / "k_Sandbox.csv",
        x_file=SANDBOX_DIR / "x_Sandbox.csv",
        reach_id_file=SANDBOX_DIR / "riv_bas_id_Sandbox.csv",
    )

    assert len(reach_ids) == 5
    assert len(k_vals) == 5
    assert len(x_vals) == 5
    assert reach_ids == [10, 20, 30, 40, 50]


# =============================================================================
# Tests for build_diffroute_inputs
# =============================================================================


def test_build_diffroute_inputs_with_rapid_files(sandbox_zarr_path: Path) -> None:
    """Verify build_diffroute_inputs with RAPID parameter files."""
    G, param_df = build_diffroute_inputs(
        zarr_path=sandbox_zarr_path,
        k_file=SANDBOX_DIR / "k_Sandbox.csv",
        x_file=SANDBOX_DIR / "x_Sandbox.csv",
        reach_id_file=SANDBOX_DIR / "riv_bas_id_Sandbox.csv",
        k_units="seconds",
    )

    assert isinstance(G, nx.DiGraph)
    assert isinstance(param_df, pd.DataFrame)
    assert len(G.nodes()) == 5
    assert len(param_df) == 5
    assert "k" in param_df.columns
    assert "x" in param_df.columns


def test_build_diffroute_inputs_with_arrays(sandbox_zarr_path: Path) -> None:
    """Verify build_diffroute_inputs with array parameters."""
    k_vals = [9000.0] * 5
    x_vals = [0.25] * 5

    G, param_df = build_diffroute_inputs(
        zarr_path=sandbox_zarr_path,
        k=k_vals,
        x=x_vals,
        k_units="seconds",
    )

    assert isinstance(G, nx.DiGraph)
    assert isinstance(param_df, pd.DataFrame)
    assert len(G.nodes()) == 5
    assert len(param_df) == 5


# =============================================================================
# Integration test with DiffRoute
# =============================================================================


@requires_cuda
def test_diffroute_integration_with_zarr_group(sandbox_zarr_root: zarr.Group) -> None:
    """Verify the adapter works end-to-end with DiffRoute using zarr Group."""
    import xarray as xr
    from diffroute import LTIRouter, RivTree

    # Build graph directly from zarr Group
    G = zarr_group_to_networkx(sandbox_zarr_root)
    order = sandbox_zarr_root["order"][:]

    # Load RAPID parameters and create param_df
    reach_ids, k_vals, x_vals = load_rapid_params(
        k_file=SANDBOX_DIR / "k_Sandbox.csv",
        x_file=SANDBOX_DIR / "x_Sandbox.csv",
        reach_id_file=SANDBOX_DIR / "riv_bas_id_Sandbox.csv",
    )

    # Reorder parameters to match zarr order
    id_to_idx = {rid: i for i, rid in enumerate(reach_ids)}
    k_ordered = [k_vals[id_to_idx[int(rid)]] for rid in order]
    x_ordered = [x_vals[id_to_idx[int(rid)]] for rid in order]

    param_df = create_param_df(order, k=k_ordered, x=x_ordered, k_units="seconds")

    # Load runoff data
    qext_file = SANDBOX_DIR / "Qext_Sandbox_19700101_19700110.nc4"
    ds = xr.open_dataset(qext_file)
    qext = ds["Qext"].values
    ds.close()
    runoff = torch.from_numpy(qext.T).unsqueeze(0).float().to(DEVICE)

    # Create DiffRoute objects
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=1)

    # Run routing
    discharge = router(runoff, riv)

    # Validate output
    assert discharge.shape == runoff.shape
    assert not torch.isnan(discharge).any()
    assert (discharge >= 0).all()
