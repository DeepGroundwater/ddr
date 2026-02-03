"""Tests to verify DiffRoute installation using RAPID Sandbox data.

NOTE: DiffRoute requires CUDA. These tests will be skipped if CUDA is not available.
"""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch
import xarray as xr

# Path to RAPID Sandbox test data
SANDBOX_DIR = Path(__file__).parent.parent / "input" / "Sandbox"

# DiffRoute requires CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffRoute requires CUDA")

# IRF parameter names for each model (from diffroute.irfs)
IRF_PARAMS = {
    "pure_lag": ["delay"],
    "linear_storage": ["tau"],
    "nash_cascade": ["tau"],
    "muskingum": ["x", "k"],
    "hayami": ["D", "L", "c"],
}


@pytest.fixture
def sandbox_network() -> tuple[nx.DiGraph, pd.DataFrame]:
    """Load RAPID Sandbox network topology as NetworkX DiGraph and params DataFrame.

    Returns
    -------
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


@pytest.fixture
def sandbox_runoff() -> torch.Tensor:
    """Load RAPID Sandbox runoff data as PyTorch tensor."""
    qext_file = SANDBOX_DIR / "Qext_Sandbox_19700101_19700110.nc4"
    ds = xr.open_dataset(qext_file)

    # Qext shape: (time=80, rivid=5)
    qext = ds["Qext"].values
    ds.close()

    # DiffRoute expects (batch, nodes, time)
    # Add batch dimension
    runoff = torch.from_numpy(qext.T).unsqueeze(0).float()  # (1, 5, 80)
    return runoff


def test_diffroute_with_sandbox_network(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
) -> None:
    """Verify DiffRoute can build a router from RAPID Sandbox network."""
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network

    # Create RivTree with graph, IRF function name, and parameters
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df)

    # LTIRouter takes routing configuration, not the RivTree
    router = LTIRouter(max_delay=100, dt=1)

    assert router is not None
    assert len(riv) == 5


@requires_cuda
def test_diffroute_routing_sandbox(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
) -> None:
    """Run DiffRoute on RAPID Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network
    runoff = sandbox_runoff.to(DEVICE)  # (1, 5, 80)

    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=1)

    # Forward pass takes both runoff and the RivTree
    discharge = router(runoff, riv)

    # Output should match input shape
    assert discharge.shape == runoff.shape
    assert not torch.isnan(discharge).any()
    # Discharge should be non-negative (physical constraint)
    assert (discharge >= 0).all()


@requires_cuda
def test_diffroute_gradient_sandbox(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
) -> None:
    """Verify gradients flow through DiffRoute with Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network
    runoff = sandbox_runoff.clone().to(DEVICE).requires_grad_(True)

    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=1)

    discharge = router(runoff, riv)
    loss = discharge.sum()
    loss.backward()

    assert runoff.grad is not None
    assert not torch.isnan(runoff.grad).any()


@requires_cuda
def test_diffroute_multiple_models(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
) -> None:
    """Test different LTI models on Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network
    runoff = sandbox_runoff.to(DEVICE)

    models = ["muskingum", "linear_storage", "pure_lag"]

    for model_name in models:
        # Create RivTree with the appropriate IRF function
        riv = RivTree(G, irf_fn=model_name, param_df=param_df).to(DEVICE)
        router = LTIRouter(max_delay=100, dt=1)

        # Forward pass takes both runoff and the RivTree
        discharge = router(runoff, riv)

        assert discharge.shape == runoff.shape, f"{model_name} failed shape check"
        assert not torch.isnan(discharge).any(), f"{model_name} produced NaN"
