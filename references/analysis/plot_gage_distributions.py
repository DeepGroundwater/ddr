"""Visualize ABS_DIFF distributions across gage reference datasets.

Produces a single figure (ABS_DIFF_distributions.png) with:
  - Top row: per-dataset ABS_DIFF histograms (GAGES-II, camels_670, gages_3000)
  - Bottom panel: CDF overlay of ABS_DIFF across all datasets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GAGE_FILES = [
    "GAGES-II.csv",
    "camels_670.csv",
    "gages_3000.csv",
]

DATASET_COLORS = ["#1f77b4", "#9467bd", "#ff7f0e"]


def main(gage_dir: Path, output_dir: Path, bins: int = 50) -> None:
    """Build a combined ABS_DIFF figure: histograms + CDF overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets: dict[str, pd.DataFrame] = {}
    for filename in GAGE_FILES:
        filepath = gage_dir / filename
        if not filepath.exists():
            print(f"WARNING: {filepath} not found, skipping.")
            continue
        df = pd.read_csv(filepath)
        if "ABS_DIFF" not in df.columns:
            print(f"WARNING: {filepath.name} missing ABS_DIFF column, skipping.")
            continue
        datasets[filepath.stem] = df

    if not datasets:
        print("No datasets loaded.")
        return

    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 9), height_ratios=[1, 1])
    # Handle case where len(datasets) == 1
    if len(datasets) == 1:
        axes = axes.reshape(2, 1)

    # Top row: per-dataset ABS_DIFF histograms
    for col, ((name, df), color) in enumerate(zip(datasets.items(), DATASET_COLORS, strict=False)):
        ax = axes[0, col]
        data = df["ABS_DIFF"].dropna()
        data = data[data > 0]

        mean = data.mean()
        median = data.median()
        std = data.std()

        bin_edges = np.logspace(np.log10(0.1), np.log10(1000), bins + 1)
        ax.hist(data, bins=bin_edges, edgecolor="black", linewidth=0.3, alpha=0.7, color=color)
        ax.set_xscale("log")
        ax.set_xlim(0.1, 1000)

        ax.axvline(median, color="red", linestyle="--", linewidth=1.2, label=f"median={median:.2f}")
        ax.axvline(50.0, color="orange", linestyle=":", linewidth=1.2, label="50 km²")

        stats_text = f"n={len(data):,}\nmean={mean:.2f}\nmedian={median:.2f}\nstd={std:.2f}"
        ax.text(
            0.97,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

        ax.set_xlabel("ABS_DIFF (km²)")
        ax.set_ylabel("Count")
        ax.set_title(f"{name}  (n={len(df):,})")
        ax.legend(fontsize=7, loc="upper left")

    # Bottom panel: CDF overlay spanning full width
    ax_cdf = fig.add_subplot(2, 1, 2)
    # Hide the individual bottom-row axes created by subplots
    for col in range(len(datasets)):
        axes[1, col].set_visible(False)

    stats_lines: list[str] = []
    for (name, df), color in zip(datasets.items(), DATASET_COLORS, strict=False):
        data = df["ABS_DIFF"].dropna()
        data = data[data > 0]
        data_sorted = np.sort(data.values)
        ecdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        median = np.median(data_sorted)

        ax_cdf.plot(data_sorted, ecdf, color=color, linewidth=1.5, label=f"{name} (n={len(data_sorted):,})")
        ax_cdf.axvline(median, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
        stats_lines.append(f"{name}: {median:.2f}")

    ax_cdf.axvline(50.0, color="gray", linestyle=":", linewidth=1.2, label="50 km²")
    ax_cdf.set_xscale("log")
    ax_cdf.set_xlim(0.1, 1000)
    ax_cdf.set_xlabel("ABS_DIFF (km²)")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_title("CDF — ABS_DIFF")
    ax_cdf.legend(fontsize=8, loc="lower right")
    ax_cdf.grid(True, alpha=0.3)

    stats_text = "Median\n" + "\n".join(stats_lines)
    ax_cdf.text(
        0.03,
        0.97,
        stats_text,
        transform=ax_cdf.transAxes,
        fontsize=7,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    fig.tight_layout()
    out_path = output_dir / "ABS_DIFF_distributions.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ABS_DIFF distributions across gage datasets.")
    parser.add_argument(
        "--gage-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "gage_info",
        help="Directory containing gage CSV files (default: references/gage_info/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for PNG (default: references/analysis/)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Number of histogram bins (default: 50)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path(__file__).resolve().parent
    main(args.gage_dir, output_dir, args.bins)
