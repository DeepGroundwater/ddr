from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_time_series(prediction, observation, time_range, gage_id, name, metrics, path, warmup=3):
    """Plot time series for a single prediction

    Parameters
    ----------
    prediction : np.ndarray
        The predicted values
    observation : np.ndarray
        The observed values
    time_range : np.ndarray
        The time range
    gage_id : str
        The gage ID
    name : str
        The name of the gage
    metrics : dict
        The metrics dictionary
    path : Path
        Path to save the plot
    warmup : int, optional
        Number of warmup timesteps to exclude, by default 3
    """
    fig = plt.figure(figsize=(10, 5))
    prediction_to_plot = prediction[warmup:]
    observation_to_plot = observation[warmup:]
    plt.plot(time_range[warmup:], observation_to_plot, label="Observation")
    plt.plot(time_range[warmup:], prediction_to_plot, label="Routed Streamflow")
    nse = metrics["nse"]
    plt.title(f"Train time period Hydrograph - GAGE ID: {gage_id} - Name: {name}")
    plt.xlabel("Time (hours)")
    plt.ylabel(r"Discharge $m^3/s$")
    plt.legend(title=f"NSE: {nse:.4f}")
    plt.savefig(path)
    plt.close(fig)


def plot_cdf(
    data_list: list[np.ndarray],
    legend_labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple | None = None,
    colors: list[str] | None = None,
    figsize: tuple = (8, 6),
    path: Path | None = None,
) -> tuple:
    """Plot Cumulative Distribution Functions (CDFs) for a list of datasets.

    Parameters
    ----------
    data_list : list[np.ndarray]
        List of datasets to plot CDFs for
    legend_labels : list[str], optional
        Labels for the legend
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    xlim : tuple, optional
        X-axis limits
    colors : list[str], optional
        Colors for each dataset
    figsize : tuple, optional
        Figure size, by default (8, 6)
    path : Path, optional
        Path to save the plot

    Returns
    -------
    tuple
        Figure and axis objects
    """
    if colors is None:
        colors = [
            "darkblue",
            "blue",
            "red",
            "deepskyblue",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
            "silver",
            "orchid",
            "brown",
        ]

    fig, ax = plt.subplots(figsize=figsize)

    for i, data in enumerate(data_list):
        # Remove NaN values and sort
        clean_data = np.sort(data[~np.isnan(data)])
        if len(clean_data) == 0:
            continue

        # Calculate cumulative probabilities
        y_rank = np.arange(len(clean_data)) / float(len(clean_data) - 1)

        # Create legend label
        median_value = np.median(clean_data)
        if legend_labels is not None and i < len(legend_labels):
            label = f"{legend_labels[i]}: Median {median_value:.4f}"
        else:
            label = f"Dataset {i + 1}: Median {median_value:.4f}"

        # Plot CDF
        color = colors[i % len(colors)]
        ax.plot(clean_data, y_rank, color=color, label=label)

    # Add reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="y=x")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)

    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_drainage_area_boxplots(
    gages: pd.DataFrame,
    metrics: list[str],
    model_names: list[str],
    colors: list[str] | None = None,
    bins: np.ndarray | None = None,
    title: str | None = None,
    ylabel: str = "NSE",
    xlabel: str = r"Drainage area (km$^2$)",
    figsize: tuple = (18, 6),
    y_limits: tuple = (0.0, 1.0),
    path: Path | None = None,
) -> plt.Figure:
    """Create box plots of model performance metrics binned by drainage area.

    Parameters
    ----------
    gages : pd.DataFrame
        DataFrame containing gauge information with 'DRAIN_SQKM' column and metric columns
    metrics : list[str]
        List of column names containing the metrics to plot
    model_names : list[str]
        List of model names for the legend (supports LaTeX formatting)
    colors : list[str], optional
        Colors for each model. If None, uses default nature-inspired palette
    bins : np.ndarray, optional
        Bin edges for drainage area. If None, uses default bins
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label, by default "NSE"
    xlabel : str, optional
        X-axis label, by default "Drainage area (km²)"
    figsize : tuple, optional
        Figure size, by default (18, 6)
    y_limits : tuple, optional
        Y-axis limits, by default (0.0, 1.0)
    path : Path, optional
        Path to save the plot

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    # Set default colors if not provided
    if colors is None:
        colors = ["#82C6E2", "#4878D0", "#D65F5F", "#EE854A"]  # Nature-inspired palette

    # Set default bins if not provided
    if bins is None:
        bins = np.array([0, 1000, 5000, 10000, 30000, 50000])

    # Validate inputs
    if len(metrics) != len(model_names):
        raise ValueError("Number of metrics must match number of model names")

    if len(colors) < len(metrics):
        # Repeat colors if not enough provided
        colors = (colors * ((len(metrics) // len(colors)) + 1))[: len(metrics)]

    # Create area bin categories
    gages_copy = gages.copy()
    gages_copy["area_bin"] = pd.cut(gages_copy["DRAIN_SQKM"], bins, labels=False) + 1
    n_bins = len(bins) - 1

    # Update plot style for publication quality
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 22,
            "axes.linewidth": 1.5,
            "axes.edgecolor": "#333333",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "xtick.major.width": 1.5,
            "xtick.minor.width": 1,
            "ytick.major.width": 1.5,
            "ytick.minor.width": 1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True, facecolor="white")

    # Create labels for x-axis
    labels = [f"{int(bins[i])}~{int(bins[i + 1])}" for i in range(len(bins) - 1)]

    # Create equally-sized bins for visualization
    bin_width = 5
    bin_positions = [i * bin_width for i in range(n_bins)]

    # Loop through models and create boxplots
    for j, (metric, color, _model_name) in enumerate(zip(metrics, colors, model_names, strict=False)):
        # Calculate model positioning within each bin
        model_width = bin_width / (len(metrics) + 1)
        model_offset = j - (len(metrics) - 1) / 2  # Center the models within each bin

        data = []
        positions = []
        widths = []

        for i in range(1, n_bins + 1):
            bin_data = gages_copy[gages_copy["area_bin"] == i][metric].dropna().values
            data.append(bin_data)

            # Position the box plot within the bin
            bin_center = bin_positions[i - 1] + bin_width / 2
            positions.append(bin_center + model_offset * model_width)
            widths.append(model_width * 0.8)

        # Create box plot
        _ = ax.boxplot(
            data,
            vert=True,
            showfliers=False,
            positions=positions,
            patch_artist=True,
            widths=widths,
            boxprops={"facecolor": color, "color": "black", "alpha": 0.8, "linewidth": 1.5},
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"color": "black", "linewidth": 1.5},
            capprops={"color": "black", "linewidth": 1.5},
        )

    # Add site count text above each bin
    y_lower, y_upper = y_limits
    for i in range(1, n_bins + 1):
        num_sites = sum(gages_copy["area_bin"] == i)
        bin_center = bin_positions[i - 1] + bin_width / 2
        ax.text(
            bin_center,
            y_upper + 0.05,
            f"{num_sites} sites",
            horizontalalignment="center",
            fontsize=20,
            color="#333333",
        )

    # Customize axes
    ax.set_ylabel(ylabel, fontsize=24, weight="bold", color="#333333")
    ax.set_xlabel(xlabel, fontsize=24, weight="bold", color="#333333")

    # Set y-axis
    ax.set_yticks(np.arange(y_lower, y_upper + 0.1, 0.2))
    ax.set_ylim([y_lower, y_upper + 0.15])  # Extra space for site count text
    ax.tick_params(axis="y", labelsize=16)

    # Set x-axis
    ax.set_xlim([-0.5, n_bins * bin_width + 0.5])

    # Add vertical dashed lines at bin boundaries
    for i in range(n_bins + 1):
        bin_boundary = i * bin_width
        ax.axvline(bin_boundary, color="#333333", linestyle="--", lw=1.5, alpha=0.7)

    # Set x-ticks at bin centers
    bin_centers = [bin_positions[i] + bin_width / 2 for i in range(n_bins)]
    ax.set_xticks(bin_centers)
    ax.set_xticklabels(labels, fontsize=16)

    # Create legend
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#333333",
            lw=2,
            markerfacecolor=color,
            marker="s",
            markersize=10,
            label=name,
            markeredgecolor="black",
            markeredgewidth=1.5,
        )
        for name, color in zip(model_names, colors, strict=False)
    ]

    # Determine legend layout
    ncol = 2 if len(model_names) <= 4 else 3
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        fontsize=16,
        frameon=True,
        framealpha=0.9,
        edgecolor="#E5E5E5",
        ncol=ncol,
    )

    if title:
        ax.set_title(title, fontsize=26, weight="bold", color="#333333", pad=20)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    return fig


def plot_box_fig(
    data_list: list[np.ndarray | list[np.ndarray]],
    labels: list[str] | None = None,
    title: str | None = None,
    ylabel: str | None = None,
    colors: list[str] | None = None,
    figsize: tuple = (10, 8),
    path: Path | None = None,
) -> plt.Figure:
    """Create box plots for multiple datasets.

    Parameters
    ----------
    data_list : list[np.ndarray | list[np.ndarray]]
        List of datasets to create box plots for
    labels : list[str], optional
        Labels for each box plot
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    colors : list[str], optional
        Colors for each box plot
    figsize : tuple, optional
        Figure size, by default (10, 8)
    path : Path, optional
        Path to save the plot

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    if colors is None:
        colors = [
            "darkblue",
            "blue",
            "red",
            "deepskyblue",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
            "silver",
            "orchid",
            "brown",
        ]

    n_plots = len(data_list)
    fig, axes = plt.subplots(ncols=n_plots, nrows=1, sharey=True, figsize=figsize, constrained_layout=True)

    # Ensure axes is always iterable
    if n_plots == 1:
        axes = [axes]

    for i, (ax, data) in enumerate(zip(axes, data_list, strict=False)):
        # Clean the data
        if isinstance(data, list):
            clean_data = []
            for subset in data:
                if subset is not None and len(subset) > 0:
                    clean_subset = np.array(subset)
                    clean_subset = clean_subset[~np.isnan(clean_subset)]
                    clean_data.append(clean_subset)
                else:
                    clean_data.append(np.array([]))
        else:
            clean_data = np.array(data)
            clean_data = clean_data[~np.isnan(clean_data)]

        # Create box plot
        bp = ax.boxplot(clean_data, patch_artist=True, notch=True, showfliers=False, widths=0.6, whis=[5, 95])

        # Set colors
        for j, box in enumerate(bp["boxes"]):
            color = colors[j % len(colors)]
            box.set_facecolor(color)
            box.set_edgecolor("black")
            box.set_linewidth(1)

        # Set labels and formatting
        if labels and i < len(labels):
            ax.set_xlabel(labels[i])

        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

    if ylabel:
        axes[0].set_ylabel(ylabel)

    if title:
        fig.suptitle(title)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
