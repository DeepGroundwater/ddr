"""Tests to verify DiffRoute installation using RAPID Sandbox data.

NOTE: DiffRoute requires CUDA. These tests will be skipped if CUDA is not available.
"""

import networkx as nx
import pandas as pd
import pytest
import torch

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

# Reach IDs in the Sandbox network (RAPID2 ordering)
REACH_IDS = [10, 20, 30, 40, 50]


def reorder_to_diffroute(data: torch.Tensor, riv) -> torch.Tensor:
    """Reorder data from RAPID2 ordering [10,20,30,40,50] to DiffRoute's DFS ordering.

    Args:
        data: Tensor with node dim at index 1, in RAPID2 order
        riv: RivTree with nodes_idx mapping reach_id -> internal_index

    Returns:
        Tensor reordered for DiffRoute input
    """
    rapid2_to_idx = {rid: i for i, rid in enumerate(REACH_IDS)}
    # For each DFS position, get which RAPID2 index to pull from
    reorder_idx = [rapid2_to_idx[rid] for rid in riv.nodes_idx.index]
    return data[:, reorder_idx, :]


def reorder_to_rapid2(data: torch.Tensor, riv) -> torch.Tensor:
    """Reorder data from DiffRoute's DFS ordering back to RAPID2 ordering [10,20,30,40,50].

    Args:
        data: Tensor with node dim at index 0 or 1, in DiffRoute's DFS order
        riv: RivTree with nodes_idx mapping reach_id -> internal_index

    Returns:
        Tensor reordered to RAPID2 order for comparison with fixtures
    """
    # For each RAPID2 position (0-4), get which DFS index to pull from
    reorder_idx = [int(riv.nodes_idx.loc[rid]) for rid in REACH_IDS]
    if data.dim() == 2:
        return data[reorder_idx, :]
    return data[:, reorder_idx, :]


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


# =============================================================================
# RAPID2 Reference Validation Tests
# =============================================================================


@requires_cuda
def test_compare_with_rapid2_reference(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
    sandbox_expected_qout: torch.Tensor,
) -> None:
    """Compare DiffRoute output against RAPID2 reference after spin-up.

    DiffRoute starts from zero discharge state while RAPID2 starts from Qinit.
    We compare later timesteps where both implementations should converge to
    similar values, since the Qext pattern drives the system toward steady state.
    """
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network

    # RAPID uses dt=900s = 0.25 hours. k=9000s = 0.104 days
    # dt in days = 900 / 86400 = 0.0104167 days
    dt_days = 900 / 86400
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    # Reorder input to DiffRoute's DFS order, run routing, reorder output back to RAPID2 order
    runoff = reorder_to_diffroute(sandbox_runoff, riv).to(DEVICE)
    discharge = router(runoff, riv)  # (1, 5, 80)
    discharge_cpu = reorder_to_rapid2(discharge.squeeze(0).cpu(), riv)  # (5, 80) in RAPID2 order

    # After spin-up period (skip first 20 timesteps), compare the later values
    # The system should approach similar steady-state behavior
    spin_up = 20

    # Compare at outlet (reach 50 = index 4 in RAPID2 order)
    rapid2_outlet = sandbox_expected_qout[spin_up:, 4]
    diffroute_outlet = discharge_cpu[4, spin_up:]

    # Check correlation - should be highly correlated since both follow Qext pattern
    correlation = torch.corrcoef(torch.stack([rapid2_outlet, diffroute_outlet]))[0, 1]

    assert correlation > 0.8, f"Low correlation with RAPID2: {correlation:.3f}"

    # Check that DiffRoute converges toward RAPID2 at the end (last 10 timesteps)
    # when both should be approaching steady state
    end_rapid2 = sandbox_expected_qout[-10:, 4].mean()
    end_diffroute = discharge_cpu[4, -10:].mean()

    # Allow 10% relative tolerance for steady-state convergence
    rel_diff = abs(end_rapid2 - end_diffroute) / end_rapid2
    assert rel_diff < 0.10, f"Final values diverge: RAPID2={end_rapid2:.2f}, DiffRoute={end_diffroute:.2f}"


@requires_cuda
def test_downstream_accumulation(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
) -> None:
    """Verify discharge increases at confluences.

    Network topology:
    - Reaches 10, 20 flow into reach 30 (confluence)
    - Reaches 30, 40 flow into reach 50 (outlet, confluence)

    At each confluence, downstream discharge should be >= sum of upstream
    discharges (water is conserved, not created).
    """
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network

    dt_days = 900 / 86400
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    # Reorder input to DiffRoute's DFS order, run routing, reorder output back to RAPID2 order
    runoff = reorder_to_diffroute(sandbox_runoff, riv).to(DEVICE)
    discharge = router(runoff, riv)  # (1, 5, 80)
    discharge_cpu = reorder_to_rapid2(discharge.squeeze(0).cpu(), riv)  # (5, 80) in RAPID2 order

    # Now indices 0-4 correspond to reaches [10, 20, 30, 40, 50]
    q10 = discharge_cpu[0, :]  # reach 10
    q20 = discharge_cpu[1, :]  # reach 20
    q30 = discharge_cpu[2, :]  # reach 30 (confluence of 10, 20)
    q40 = discharge_cpu[3, :]  # reach 40
    q50 = discharge_cpu[4, :]  # reach 50 (outlet, confluence of 30, 40)

    # After spin-up, check confluence behavior
    # Q30 should be >= individual upstream contributions (considering timing/storage)
    # Q50 should be the largest as it's the outlet
    spin_up = 20

    # At outlet: Q50 should be >= Q30 and Q50 should be >= Q40 (on average)
    assert q50[spin_up:].mean() >= q30[spin_up:].mean(), "Outlet Q50 should be >= Q30"
    assert q50[spin_up:].mean() >= q40[spin_up:].mean(), "Outlet Q50 should be >= Q40"

    # At confluence 30: Q30 should be >= Q10 and Q30 should be >= Q20 (on average)
    assert q30[spin_up:].mean() >= q10[spin_up:].mean(), "Confluence Q30 should be >= Q10"
    assert q30[spin_up:].mean() >= q20[spin_up:].mean(), "Confluence Q30 should be >= Q20"


@requires_cuda
def test_mass_balance(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_hourly_runoff: torch.Tensor,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Verify mass conservation through the network.

    Total water input (sum of Q') should approximately equal total water
    output at the outlet, accounting for storage changes in the network.

    Uses hourly interpolated data (238 timesteps) for consistency with DDR.
    """
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network

    dt_days = 3600 / 86400  # 1 hour in days
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    # Reorder input to DiffRoute's DFS order, run routing, reorder output back to RAPID2 order
    runoff = reorder_to_diffroute(sandbox_hourly_runoff, riv).to(DEVICE)
    discharge = router(runoff, riv)  # (1, 5, 238)
    discharge_cpu = reorder_to_rapid2(discharge.squeeze(0).cpu(), riv)  # (5, 238) in RAPID2 order

    # Total input: sum of all Q' across all reaches and timesteps
    total_input = sandbox_hourly_qprime.sum().item()

    # Total output: sum of discharge at outlet (reach 50 = index 4) over all timesteps
    total_outlet = discharge_cpu[4, :].sum().item()

    # Mass balance check with 5% tolerance
    # (DiffRoute may differ due to initial conditions and numerical differences)
    rel_error = abs(total_outlet - total_input) / total_input
    assert rel_error < 0.05, (
        f"Mass balance error too large: input={total_input:.2f}, "
        f"outlet={total_outlet:.2f}, rel_error={rel_error:.2%}"
    )
    # Print mass balance info for debugging
    print(
        f"\ndiffroute Mass balance: Q' total={total_input:.2f}, Outlet total={total_outlet:.2f}, "
        f"rel_error={rel_error:.2%}"
    )


@requires_cuda
def test_steady_state_convergence(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_qinit: torch.Tensor,
) -> None:
    """Verify constant Qext produces discharge converging toward expected steady-state.

    When Qext is constant, the system should converge to a steady state where
    Q_downstream = sum(Q_upstream) for each node.

    With Qext = [9, 9, 9, 18, 18] (mimicking the end of Sandbox data):
    - Expected steady-state: Q = [9, 9, 27, 18, 63] (matching Qinit)
    """
    from diffroute import LTIRouter, RivTree

    G, param_df = sandbox_network

    dt_days = 900 / 86400
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    # Create constant Qext matching the final values from Sandbox
    # This should drive the system toward Qinit values at steady state
    # RAPID2 ordering: [10, 20, 30, 40, 50] with Qext = [9, 9, 9, 18, 18]
    n_timesteps = 200  # Long enough to reach steady state
    constant_qext = torch.tensor([9.0, 9.0, 9.0, 18.0, 18.0])  # Per-reach lateral inflow in RAPID2 order
    qext_rapid2 = constant_qext.unsqueeze(0).unsqueeze(2).expand(1, 5, n_timesteps)  # (1, 5, 200)

    # Reorder input to DiffRoute's DFS order, run routing, reorder output back to RAPID2 order
    qext = reorder_to_diffroute(qext_rapid2, riv).to(DEVICE)
    discharge = router(qext, riv)  # (1, 5, 200)
    discharge_cpu = reorder_to_rapid2(discharge.squeeze(0).cpu(), riv)  # (5, 200) in RAPID2 order

    # Expected steady-state discharge in RAPID2 order [10, 20, 30, 40, 50]:
    # Reach 10: 9 (just its own Qext)
    # Reach 20: 9 (just its own Qext)
    # Reach 30: 9 + 9 + 9 = 27 (upstream 10 + 20 + own Qext)
    # Reach 40: 18 (just its own Qext)
    # Reach 50: 27 + 18 + 18 = 63 (upstream 30 + 40 + own Qext)
    expected_steady_state = sandbox_qinit  # [9, 9, 27, 18, 63]

    # Check final timesteps (should be near steady state)
    final_discharge = discharge_cpu[:, -10:].mean(dim=1)

    # Allow 15% tolerance for steady-state convergence
    for i, (actual, expected) in enumerate(zip(final_discharge, expected_steady_state, strict=True)):
        rel_diff = abs(actual - expected) / expected
        assert rel_diff < 0.15, (
            f"Reach {REACH_IDS[i]} did not converge to steady state: actual={actual:.2f}, expected={expected:.2f}"
        )
