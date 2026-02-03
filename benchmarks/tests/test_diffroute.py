"""Tests to verify DiffRoute installation using RAPID Sandbox data"""

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
import torch
import xarray as xr

# Path to RAPID Sandbox test data
SANDBOX_DIR = Path(__file__).parent.parent.parent / "tests" / "integration" / "input" / "Sandbox"


@pytest.fixture
def sandbox_network() -> nx.DiGraph:
    """Load RAPID Sandbox network topology as NetworkX DiGraph."""
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
        # DiffRoute uses tau (time constant) - convert from Muskingum k
        # k is in seconds, tau is also a time constant
        G.add_node(rid, tau=float(k_vals[i]), k=float(k_vals[i]), x=float(x_vals[i]))

    # Add edges (from COMID to NextDownID, but 0 means outlet)
    for _, row in connect_df.iterrows():
        if row["next_down"] != 0:  # 0 means outlet
            G.add_edge(row["comid"], row["next_down"])

    return G


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


def test_diffroute_with_sandbox_network(sandbox_network: nx.DiGraph) -> None:
    """Verify DiffRoute can build a router from RAPID Sandbox network."""
    from diffroute import LTIRouter, RivTree

    G = sandbox_network

    # Outlet is COMID 50 (the one with NextDownID=0)
    riv = RivTree(G, outlet=50)
    router = LTIRouter(riv, model="muskingum")

    assert router is not None
    assert len(list(G.nodes())) == 5


def test_diffroute_routing_sandbox(sandbox_network: nx.DiGraph, sandbox_runoff: torch.Tensor) -> None:
    """Run DiffRoute on RAPID Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G = sandbox_network
    runoff = sandbox_runoff  # (1, 5, 80)

    riv = RivTree(G, outlet=50)
    router = LTIRouter(riv, model="muskingum")

    discharge = router(runoff)

    # Output should match input shape
    assert discharge.shape == runoff.shape
    assert not torch.isnan(discharge).any()
    # Discharge should be non-negative (physical constraint)
    assert (discharge >= 0).all()


def test_diffroute_gradient_sandbox(sandbox_network: nx.DiGraph, sandbox_runoff: torch.Tensor) -> None:
    """Verify gradients flow through DiffRoute with Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G = sandbox_network
    runoff = sandbox_runoff.clone().requires_grad_(True)

    riv = RivTree(G, outlet=50)
    router = LTIRouter(riv, model="muskingum")

    discharge = router(runoff)
    loss = discharge.sum()
    loss.backward()

    assert runoff.grad is not None
    assert not torch.isnan(runoff.grad).any()


def test_diffroute_multiple_models(sandbox_network: nx.DiGraph, sandbox_runoff: torch.Tensor) -> None:
    """Test different LTI models on Sandbox data."""
    from diffroute import LTIRouter, RivTree

    G = sandbox_network
    runoff = sandbox_runoff

    models = ["muskingum", "linear_storage", "pure_lag"]

    for model_name in models:
        riv = RivTree(G, outlet=50)
        router = LTIRouter(riv, model=model_name)
        discharge = router(runoff)

        assert discharge.shape == runoff.shape, f"{model_name} failed shape check"
        assert not torch.isnan(discharge).any(), f"{model_name} produced NaN"
