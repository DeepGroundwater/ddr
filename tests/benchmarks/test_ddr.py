"""Integration tests for DDR (Differentiable Muskingum-Cunge) routing.

Tests DDR routing using RAPID Sandbox data with mock streamflow and KAN modules.
"""

from pathlib import Path

import torch

from tests.benchmarks.conftest import RAPID2_REACH_IDS, run_ddr_routing


def test_ddr_routing_downstream_accumulation(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test that discharge accumulates downstream after spin-up.

    Network topology:
    - Reaches 10, 20 flow into reach 30 (confluence)
    - Reaches 30, 40 flow into reach 50 (outlet)

    DDR output is now in RAPID2 order [10, 20, 30, 40, 50].
    """
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    # Skip initial spin-up timesteps
    spin_up = 50
    # Indices in RAPID2 order: 0=10, 1=20, 2=30, 3=40, 4=50
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
    for i, reach_id in enumerate(RAPID2_REACH_IDS):
        mean_q = ddr_discharge[i, spin_up:].mean()
        assert mean_q > 0, f"Reach {reach_id} has non-positive mean discharge: {mean_q}"


def test_ddr_routing_mass_balance(
    sandbox_zarr_path: Path,
    sandbox_hourly_qprime: torch.Tensor,
) -> None:
    """Test mass conservation: total Q' input should equal total outlet discharge.

    Mass balance check:
    - Total input = sum of Q' (lateral inflow) across all reaches and all timesteps
    - Total output = sum of discharge at outlet (reach 50) across all timesteps

    For a conservative routing scheme, input ≈ output (within tolerance for
    numerical precision and initial/final storage effects).
    """
    ddr_discharge = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)

    # Total input: sum of all Q' across all reaches and timesteps
    # sandbox_hourly_qprime shape: (timesteps, reaches)
    total_qprime = sandbox_hourly_qprime.sum().item()

    # Total output: sum of discharge at outlet (reach 50 = index 4) over all timesteps
    # ddr_discharge shape: (reaches, timesteps) in RAPID2 order
    total_outlet = ddr_discharge[4, :].sum()

    # Calculate relative error
    rel_error = abs(total_outlet - total_qprime) / total_qprime

    # Allow 15% tolerance for mass balance
    # (some error expected due to initial conditions and numerical differences)
    assert rel_error < 0.05, (
        f"Mass balance error too large: "
        f"total Q'={total_qprime:.2f}, outlet={total_outlet:.2f}, "
        f"rel_error={rel_error:.2%}"
    )

    # Print mass balance info for debugging
    print(
        f"\nMass balance: Q' total={total_qprime:.2f}, Outlet total={total_outlet:.2f}, "
        f"rel_error={rel_error:.2%}"
    )
