"""Tests to generate hydrograph comparison plots using RAPID Sandbox data.

NOTE: DiffRoute requires CUDA. These tests will be skipped if CUDA is not available.
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch

# DiffRoute requires CUDA
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="DiffRoute requires CUDA")

# Reach IDs in the Sandbox network (RAPID2 ordering)
REACH_IDS = [10, 20, 30, 40, 50]


@requires_cuda
def test_generate_diffroute_hydrograph_plot(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
    sandbox_expected_qout: torch.Tensor,
    sandbox_qext: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Generate hydrograph comparison plot of DiffRoute vs RAPID2.

    Creates a multi-panel figure showing:
    - Panel 1: Total Qext (sum across all reaches) vs routed outlet discharge
    - Panel 2: DiffRoute discharge vs RAPID2 for outlet (reach 50)
    - Panel 3: DiffRoute discharge vs RAPID2 for confluence (reach 30)
    - Panel 4: DiffRoute discharge for all reaches

    Plot is saved to tmp_path/diffroute_hydrograph.png
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for CI
    import matplotlib.pyplot as plt
    from diffroute import LTIRouter, RivTree

    from .test_diffroute import reorder_to_diffroute, reorder_to_rapid2

    G, param_df = sandbox_network

    dt_days = 900 / 86400
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    # Reorder input to DiffRoute's DFS order, run routing, reorder output back to RAPID2 order
    runoff = reorder_to_diffroute(sandbox_runoff, riv).to(DEVICE)
    discharge = router(runoff, riv)
    discharge_cpu = reorder_to_rapid2(discharge.squeeze(0).cpu(), riv).numpy()  # (5, 80) in RAPID2 order
    discharge_t = discharge_cpu.T  # (80, 5) to match RAPID2 shape

    rapid2_qout = sandbox_expected_qout.numpy()  # (80, 5) in RAPID2 order [10, 20, 30, 40, 50]
    qext = sandbox_qext.numpy()  # (80, 5)

    # Sum of Qext across all reaches at each timestep
    # This represents the total lateral inflow entering the network
    qext_total = qext.sum(axis=1)  # (80,)

    # Time axis (3-hour intervals over 10 days)
    time_hours = np.arange(80) * 3  # hours since start

    reach_ids = [10, 20, 30, 40, 50]
    reach_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Total Qext vs routed outlet discharge
    # Now indices match RAPID2: 0=10, 1=20, 2=30, 3=40, 4=50
    ax1 = axes[0, 0]
    ax1.plot(time_hours, qext_total, "g-", label="Total Qext (sum all reaches)", linewidth=2)
    ax1.plot(time_hours, discharge_t[:, 4], "r--", label="DiffRoute Outlet (Reach 50)", linewidth=2)
    ax1.plot(time_hours, rapid2_qout[:, 4], "b:", label="RAPID2 Outlet (Reach 50)", linewidth=2)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_title("Total Qext vs Routed Outlet Discharge")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Outlet comparison (reach 50 = index 4)
    ax2 = axes[0, 1]
    ax2.plot(time_hours, rapid2_qout[:, 4], "b-", label="RAPID2", linewidth=2)
    ax2.plot(time_hours, discharge_t[:, 4], "r--", label="DiffRoute", linewidth=2)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Discharge (m³/s)")
    ax2.set_title("Outlet (Reach 50): DiffRoute vs RAPID2")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Confluence comparison (reach 30 = index 2)
    ax3 = axes[1, 0]
    ax3.plot(time_hours, rapid2_qout[:, 2], "b-", label="RAPID2", linewidth=2)
    ax3.plot(time_hours, discharge_t[:, 2], "r--", label="DiffRoute", linewidth=2)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Discharge (m³/s)")
    ax3.set_title("Confluence (Reach 30): DiffRoute vs RAPID2")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: All reaches DiffRoute output
    ax4 = axes[1, 1]
    for i, (rid, color) in enumerate(zip(reach_ids, reach_colors, strict=True)):
        ax4.plot(time_hours, discharge_t[:, i], label=f"Reach {rid}", color=color, linewidth=1.5)
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Discharge (m³/s)")
    ax4.set_title("DiffRoute Discharge: All Reaches")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("RAPID Sandbox: DiffRoute Routing Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save to tmp directory
    plot_path = tmp_path / "diffroute_hydrograph.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Print path for user reference
    print(f"\nHydrograph plot saved to: {plot_path}")

    # Verify plot was created
    assert plot_path.exists(), f"Plot was not created at {plot_path}"
    assert plot_path.stat().st_size > 0, "Plot file is empty"


def test_generate_ddr_hydrograph_plot(
    ddr_discharge: np.ndarray,
    sandbox_expected_qout: torch.Tensor,
    sandbox_qext: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Generate hydrograph comparison plot of DDR vs RAPID2.

    Note: This test does not require CUDA as DDR runs on CPU.

    Creates a multi-panel figure showing:
    - Panel 1: Total Qext vs DDR outlet discharge
    - Panel 2: DDR discharge vs RAPID2 for outlet (reach 50)
    - Panel 3: DDR discharge vs RAPID2 for confluence (reach 30)
    - Panel 4: DDR discharge for all reaches

    Plot is saved to tmp_path/ddr_hydrograph.png
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rapid2_qout = sandbox_expected_qout.numpy()  # (80, 5)
    qext = sandbox_qext.numpy()  # (80, 5)
    qext_total = qext.sum(axis=1)  # (80,)

    # Time axes
    time_hours_3h = np.arange(80) * 3  # 3-hourly for RAPID2
    time_hours_1h = np.arange(ddr_discharge.shape[1])  # hourly for DDR

    reach_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: Total Qext vs DDR outlet discharge
    ax1 = axes[0, 0]
    ax1.plot(time_hours_3h, qext_total, "g-", label="Total Qext (sum all reaches)", linewidth=2)
    ax1.plot(time_hours_1h, ddr_discharge[4, :], "m-", label="DDR Outlet (Reach 50)", linewidth=1.5)
    ax1.plot(time_hours_3h, rapid2_qout[:, 4], "b:", label="RAPID2 Outlet (Reach 50)", linewidth=2)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_title("Total Qext vs Routed Outlet Discharge")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: DDR vs RAPID2 at outlet (reach 50)
    ax2 = axes[0, 1]
    ax2.plot(time_hours_3h, rapid2_qout[:, 4], "b-", label="RAPID2", linewidth=2)
    ax2.plot(time_hours_1h, ddr_discharge[4, :], "m--", label="DDR", linewidth=1.5)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Discharge (m³/s)")
    ax2.set_title("Outlet (Reach 50): DDR vs RAPID2")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: DDR vs RAPID2 at confluence (reach 30)
    ax3 = axes[1, 0]
    ax3.plot(time_hours_3h, rapid2_qout[:, 2], "b-", label="RAPID2", linewidth=2)
    ax3.plot(time_hours_1h, ddr_discharge[2, :], "m--", label="DDR", linewidth=1.5)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Discharge (m³/s)")
    ax3.set_title("Confluence (Reach 30): DDR vs RAPID2")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: All reaches DDR output
    ax4 = axes[1, 1]
    for i, (rid, color) in enumerate(zip(REACH_IDS, reach_colors, strict=True)):
        ax4.plot(time_hours_1h, ddr_discharge[i, :], label=f"Reach {rid}", color=color, linewidth=1.5)
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Discharge (m³/s)")
    ax4.set_title("DDR Discharge: All Reaches")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("RAPID Sandbox: DDR Routing Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save plot
    plot_path = tmp_path / "ddr_hydrograph.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nDDR hydrograph plot saved to: {plot_path}")

    # Verify plot was created
    assert plot_path.exists(), f"Plot was not created at {plot_path}"
    assert plot_path.stat().st_size > 0, "Plot file is empty"


@requires_cuda
def test_generate_comparison_plot(
    sandbox_network: tuple[nx.DiGraph, pd.DataFrame],
    sandbox_runoff: torch.Tensor,
    ddr_discharge: np.ndarray,
    sandbox_expected_qout: torch.Tensor,
    sandbox_qext: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Generate hydrograph comparing DDR vs DiffRoute vs RAPID2.

    Creates a 6-panel figure comparing outputs from all three routing methods.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from diffroute import LTIRouter, RivTree

    from .test_diffroute import reorder_to_diffroute, reorder_to_rapid2

    # Run DiffRoute
    G, param_df = sandbox_network
    dt_days = 900 / 86400
    riv = RivTree(G, irf_fn="muskingum", param_df=param_df).to(DEVICE)
    router = LTIRouter(max_delay=100, dt=dt_days).to(DEVICE)

    runoff = reorder_to_diffroute(sandbox_runoff, riv).to(DEVICE)
    diffroute_discharge = router(runoff, riv)
    diffroute_cpu = reorder_to_rapid2(diffroute_discharge.squeeze(0).cpu(), riv).numpy()
    diffroute_t = diffroute_cpu.T  # (80, 5) in RAPID2 order

    # Reference data
    rapid2_qout = sandbox_expected_qout.numpy()  # (80, 5)
    qext = sandbox_qext.numpy()  # (80, 5)
    qext_total = qext.sum(axis=1)  # (80,)

    # Time axes
    time_hours_3h = np.arange(80) * 3  # 3-hourly for RAPID2/DiffRoute
    time_hours_1h = np.arange(ddr_discharge.shape[1])  # hourly for DDR

    reach_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Total Qext vs DDR outlet vs DiffRoute outlet
    ax1 = axes[0, 0]
    ax1.plot(time_hours_3h, qext_total, "g-", label="Total Qext", linewidth=2)
    ax1.plot(time_hours_1h, ddr_discharge[4, :], "m-", label="DDR Outlet", linewidth=1.5, alpha=0.8)
    ax1.plot(time_hours_3h, diffroute_t[:, 4], "r--", label="DiffRoute Outlet", linewidth=2)
    ax1.plot(time_hours_3h, rapid2_qout[:, 4], "b:", label="RAPID2 Outlet", linewidth=2)
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Discharge (m³/s)")
    ax1.set_title("Total Qext vs Routed Outlet Discharge")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 2: DDR vs RAPID2 at outlet (reach 50)
    ax2 = axes[0, 1]
    ax2.plot(time_hours_3h, rapid2_qout[:, 4], "b-", label="RAPID2", linewidth=2)
    ax2.plot(time_hours_1h, ddr_discharge[4, :], "m--", label="DDR", linewidth=1.5)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Discharge (m³/s)")
    ax2.set_title("Outlet (Reach 50): DDR vs RAPID2")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Panel 3: DiffRoute vs RAPID2 at outlet (reach 50)
    ax3 = axes[0, 2]
    ax3.plot(time_hours_3h, rapid2_qout[:, 4], "b-", label="RAPID2", linewidth=2)
    ax3.plot(time_hours_3h, diffroute_t[:, 4], "r--", label="DiffRoute", linewidth=2)
    ax3.set_xlabel("Time (hours)")
    ax3.set_ylabel("Discharge (m³/s)")
    ax3.set_title("Outlet (Reach 50): DiffRoute vs RAPID2")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    # Panel 4: DDR vs DiffRoute at confluence (reach 30)
    ax4 = axes[1, 0]
    ax4.plot(time_hours_3h, rapid2_qout[:, 2], "b-", label="RAPID2", linewidth=2)
    ax4.plot(time_hours_1h, ddr_discharge[2, :], "m--", label="DDR", linewidth=1.5)
    ax4.plot(time_hours_3h, diffroute_t[:, 2], "r:", label="DiffRoute", linewidth=2)
    ax4.set_xlabel("Time (hours)")
    ax4.set_ylabel("Discharge (m³/s)")
    ax4.set_title("Confluence (Reach 30): DDR vs DiffRoute vs RAPID2")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)

    # Panel 5: All reaches from DDR
    ax5 = axes[1, 1]
    for i, (rid, color) in enumerate(zip(REACH_IDS, reach_colors, strict=True)):
        ax5.plot(time_hours_1h, ddr_discharge[i, :], label=f"Reach {rid}", color=color, linewidth=1.5)
    ax5.set_xlabel("Time (hours)")
    ax5.set_ylabel("Discharge (m³/s)")
    ax5.set_title("DDR Discharge: All Reaches")
    ax5.legend(loc="upper right")
    ax5.grid(True, alpha=0.3)

    # Panel 6: All reaches from DiffRoute
    ax6 = axes[1, 2]
    for i, (rid, color) in enumerate(zip(REACH_IDS, reach_colors, strict=True)):
        ax6.plot(time_hours_3h, diffroute_t[:, i], label=f"Reach {rid}", color=color, linewidth=1.5)
    ax6.set_xlabel("Time (hours)")
    ax6.set_ylabel("Discharge (m³/s)")
    ax6.set_title("DiffRoute Discharge: All Reaches")
    ax6.legend(loc="upper right")
    ax6.grid(True, alpha=0.3)

    fig.suptitle("RAPID Sandbox: DDR vs DiffRoute vs RAPID2 Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save plot
    plot_path = tmp_path / "ddr_diffroute_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nComparison plot saved to: {plot_path}")

    # Verify plot was created
    assert plot_path.exists(), f"Plot was not created at {plot_path}"
    assert plot_path.stat().st_size > 0, "Plot file is empty"
