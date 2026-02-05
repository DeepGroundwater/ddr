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
