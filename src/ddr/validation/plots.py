from pathlib import Path

import colormaps as cmaps
import contextily as ctx
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


def flatten_data(x: np.ndarray) -> np.ndarray:
    """Flatten the input data and remove NaN values.

    Parameters
    ----------
    x : np.ndarray
        Input data array

    Returns
    -------
    np.ndarray
        Sorted array with NaN values removed
    """
    return np.sort(x[~np.isnan(x)])


def plot_cdf(
    data_list: list[np.ndarray],
    ax: plt.Axes | None = None,
    title: str | None = None,
    legend_labels: list[str] | None = None,
    figsize: tuple = (8, 6),
    reference_line: str | None = "121",
    color_list: list[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple | None = None,
) -> tuple[plt.Figure | None, plt.Axes]:
    """Plot Cumulative Distribution Functions (CDFs) for a list of datasets.

    Parameters
    ----------
    data_list : list[np.ndarray]
        List of datasets to plot CDFs for
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure and axes.
    title : str, optional
        Plot title
    legend_labels : list[str], optional
        Labels for the legend
    figsize : tuple, optional
        Figure size as (width, height), by default (8, 6)
    reference_line : str | None, optional
        Type of reference line to add ("121" for y=x, "norm" for Gaussian), by default "121"
    color_list : list[str], optional
        List of colors for each dataset. If None, uses default color palette.
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    show_diff : None, optional
        Parameter not used in this version, consider removing if not needed
    xlim : tuple, optional
        X-axis limits as (min, max)
    linespec : None, optional
        Parameter not used in this version, consider removing if not needed

    Returns
    -------
    tuple[plt.Figure | None, plt.Axes]
        Figure object (None if ax was provided) and axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if color_list is None:
        color_list = [
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
            "darkred",
            "orchid",
            "brown",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
            "darkblue",
            "blue",
            "red",
            "deepskyblue",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
        ]

    for i, data in enumerate(data_list):
        sorted_data = flatten_data(data)
        rank_values = np.arange(len(sorted_data)) / float(len(sorted_data) - 1)

        median_value = np.median(sorted_data)
        if legend_labels is not None:
            legend_string = f"{legend_labels[i]}: (NSE={median_value:.4f})"
        else:
            legend_string = f"(NSE={median_value:.4f})"

        ax.plot(sorted_data, rank_values, color=color_list[i % len(color_list)], label=legend_string)

    ax.grid(True)
    if title is not None:
        ax.set_title(title, loc="center")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)

    # Reference lines
    if reference_line == "121":
        ax.plot([0, 1], [0, 1], "k", label="y=x")
    elif reference_line == "norm":
        import scipy.stats

        norm_x = np.linspace(-5, 5, 1000)
        norm_cdf = scipy.stats.norm.cdf(norm_x, 0, 1)
        ax.plot(norm_x, norm_cdf, "k", label="Gaussian")

    ax.legend(loc="best", frameon=False)

    return fig, ax


def plot_box_fig(
    data: list,
    xlabel_list: list[str] | None = None,
    legend_labels: list[str] | None = None,
    color_list: list[str] | None = None,
    title: str | None = None,
    figsize: tuple = (10, 8),
    sharey: bool = True,
    xticklabel: None = None,  # Parameter not used in this version, consider removing if not needed
    edge_color_list: list[str] | None = None,
    legend_font_size: int = 15,
    xlabel_font_size: int = 17,
    tick_font_size: int = 26,
) -> plt.Figure:
    """Create box plots for multiple datasets with customizable styling.

    Parameters
    ----------
    data : list
        List of datasets to create box plots for. Each dataset can be a list or array.
    xlabel_list : list[str], optional
        List of x-axis labels for each subplot
    legend_labels : list[str], optional
        List of labels for the legend
    color_list : list[str], optional
        List of colors for the box faces. If None, uses default color palette.
    title : str, optional
        Overall figure title
    figsize : tuple, optional
        Figure size as (width, height), by default (10, 8)
    sharey : bool, optional
        Whether subplots share the y-axis, by default True
    xticklabel : None, optional
        Parameter not used in this version, consider removing if not needed
    edge_color_list : list[str], optional
        List of colors for box edges
    legend_font_size : int, optional
        Font size for legend text, by default 15
    xlabel_font_size : int, optional
        Font size for x-axis labels, by default 17
    tick_font_size : int, optional
        Font size for tick labels, by default 26

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    if color_list is None:
        color_list = [
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
            "darkred",
            "orchid",
            "brown",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
            "darkblue",
            "blue",
            "red",
            "deepskyblue",
            "black",
            "darkred",
            "pink",
            "gray",
            "lightgray",
        ]

    num_columns = len(data)
    fig, axes = plt.subplots(
        ncols=num_columns, nrows=1, sharey=sharey, figsize=figsize, constrained_layout=True
    )
    if num_columns == 1:  # Ensure axes is iterable
        axes = [axes]

    for i, ax in enumerate(axes):
        temp_data = data[i]
        if isinstance(temp_data, list):
            for j, subset in enumerate(temp_data):
                if subset is not None and len(subset) > 0:  # Check if subset is not None and not empty
                    subset = np.array(subset)  # Ensure subset is a NumPy array
                    subset = subset[~np.isnan(subset)]  # Remove NaN values
                    temp_data[j] = subset
                else:
                    temp_data[j] = np.array([])  # Ensure empty lists are converted to empty arrays
        else:
            temp_data = np.array(temp_data)  # Convert to NumPy array if not already
            temp_data = temp_data[~np.isnan(temp_data)]  # Remove NaN values

        box_plot = ax.boxplot(
            temp_data, patch_artist=True, notch=True, showfliers=False, widths=0.3, whis=[5, 95]
        )

        # Set edge and face colors
        if edge_color_list is not None:
            for j in range(len(box_plot["boxes"])):
                plt.setp(box_plot["boxes"][j], color=edge_color_list[j % len(edge_color_list)], linewidth=4)

        for j in range(len(box_plot["boxes"])):
            plt.setp(box_plot["boxes"][j], facecolor=color_list[j % len(color_list)])

        # Set labels
        if xlabel_list is not None:
            ax.set_xlabel(xlabel_list[i], fontsize=xlabel_font_size)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(tick_font_size)

        # Remove x-tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

    # Add horizontal lines or other customizations here

    if legend_labels is not None:
        # Adjust legend handling based on the number of columns
        fig.legend(
            box_plot["boxes"],
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            frameon=False,
            ncol=len(legend_labels),
            fontsize=legend_font_size,
        )

    if title is not None:
        fig.suptitle(title, fontsize=xlabel_font_size)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

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
    show_plot: bool = False,
) -> plt.Figure:
    """Create custom box plots of model performance metrics binned by drainage area.

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
    show_plot : bool, optional
        Whether to display the plot, by default False

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
    num_bins = len(bins) - 1

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

    # Create exactly equal-sized bins for visualization
    bin_width = 5  # Width of each bin in visualization units
    bin_positions = [i * bin_width for i in range(num_bins)]  # Starting position of each bin

    # Loop through models and create boxplots
    for j, (metric, color, _model_name) in enumerate(zip(metrics, colors, model_names, strict=False)):
        # Better spacing for models - adjust offsets to center the boxes
        model_width = bin_width / 6  # Divide bin into 6 parts (models + spacing)
        model_offset = j - (len(metrics) - 1) / 2  # Center the models within each bin

        # Create data list by slicing the dataframe for each bin
        data = []
        positions = []
        widths = []

        for i in range(1, num_bins + 1):
            bin_data = gages_copy[gages_copy["area_bin"] == i][metric].dropna().values
            data.append(bin_data)

            # Position the box plot within the bin
            bin_center = bin_positions[i - 1] + bin_width / 2
            positions.append(bin_center + model_offset * model_width)
            widths.append(model_width * 0.8)  # Make boxes slightly narrower for better appearance

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

    # Customize appearance and add site count text
    y_lower, y_upper = y_limits

    # Place the site count text above the plot, but slightly lower
    for i in range(1, num_bins + 1):
        num_sites = sum(gages_copy["area_bin"] == i)
        bin_center = bin_positions[i - 1] + bin_width / 2
        # Using figure-relative coordinates to adjust by pixels
        fig_height = fig.get_figheight()
        dpi = fig.dpi if fig.dpi else 100  # Default DPI if not specified
        pixel_offset_in_data = 20 / (dpi * fig_height)  # Convert 20 pixels to data units
        ax.text(
            bin_center,
            y_upper + 0.05 - pixel_offset_in_data,
            f"{num_sites} sites",
            horizontalalignment="center",
            fontsize=20,
            color="#333333",
        )

    # Axis labels, limits, and formatting
    ax.set_ylabel(ylabel, fontsize=24, weight="bold", color="#333333")
    ax.set_xlabel(xlabel, fontsize=24, weight="bold", color="#333333")

    ax.set_yticks(np.arange(y_lower, y_upper + 0.1, 0.2))
    ax.set_ylim([y_lower, y_upper + 0.1])  # Extra space for the site count text
    ax.tick_params(axis="y", labelsize=16)

    # Set x-axis limits
    ax.set_xlim([-0.5, num_bins * bin_width + 0.5])  # Add a bit of padding on both sides

    # Add vertical dashed lines at bin boundaries
    for i in range(num_bins + 1):
        bin_boundary = i * bin_width
        ax.axvline(bin_boundary, color="#333333", linestyle="--", lw=1.5, alpha=0.7)

    # Set x-ticks at bin centers
    bin_centers = [bin_positions[i] + bin_width / 2 for i in range(num_bins)]
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

    # Determine layout - use 2 columns for better spacing
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

    if show_plot:
        plt.show()

    return fig


def plot_gauge_map(
    gages: pd.DataFrame,
    metric_column: str,
    longitude_column: str = "LNG_GAGE",
    latitude_column: str = "LAT_GAGE",
    title: str | None = None,
    colormap: str = "bamako",
    figsize: tuple = (10, 3),
    point_size: int = 25,
    alpha: float = 0.8,
    aspect_ratio: float = 1.7,
    padding: float = 0.5,
    basemap_source: str = "CartoDB.Positron",
    colorbar_label: str = "NSE",
    path: Path | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    """Create a map visualization of gauge locations colored by performance metrics.

    Parameters
    ----------
    gages : pd.DataFrame
        DataFrame containing gauge information with longitude, latitude, and metric columns
    metric_column : str
        Column name containing the metric values to color-code the points
    longitude_column : str, optional
        Column name for longitude values, by default 'LNG_GAGE'
    latitude_column : str, optional
        Column name for latitude values, by default 'LAT_GAGE'
    title : str, optional
        Plot title. If None, auto-generates from metric_column
    colormap : str, optional
        Colormap name for the scatter points, by default 'bamako'
    figsize : tuple, optional
        Figure size as (width, height), by default (10, 3)
    point_size : int, optional
        Size of scatter plot points, by default 25
    alpha : float, optional
        Transparency of scatter points, by default 0.8
    aspect_ratio : float, optional
        Aspect ratio for the map, by default 1.7 (good for US maps)
    padding : float, optional
        Padding around the data extent in degrees, by default 0.5
    basemap_source : str, optional
        Basemap source provider, by default 'CartoDB.Positron'
    colorbar_label : str, optional
        Label for the colorbar, by default 'NSE'
    path : Path, optional
        Path to save the plot
    show_plot : bool, optional
        Whether to display the plot, by default False

    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(colormap, str):
        if hasattr(cmaps, colormap):
            cmap = getattr(cmaps, colormap)
        else:
            cmap = colormap  # Fallback to matplotlib colormap name
    else:
        cmap = colormap

    # Create scatter plot
    scatter = ax.scatter(
        gages[longitude_column],
        gages[latitude_column],
        c=gages[metric_column],
        cmap=cmap,
        s=point_size,
        alpha=alpha,
        edgecolor="none",
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(colorbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    if title is None:
        title = f"{metric_column.replace('_', ' ').title()}"
    ax.set_title(title, fontsize=8)

    ax.set_xlabel("Longitude", fontsize=6)
    ax.set_ylabel("Latitude", fontsize=6)

    try:
        provider = getattr(ctx.providers, basemap_source.split(".")[0])
        if "." in basemap_source:
            for attr in basemap_source.split(".")[1:]:
                provider = getattr(provider, attr)
        ctx.add_basemap(ax, crs="EPSG:4326", source=provider)
    except (AttributeError, Exception):
        # Fallback to default provider if specified one doesn't work
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.CartoDB.Positron)

    ax.set_aspect(aspect_ratio)

    ax.set_xlim(gages[longitude_column].min() - padding, gages[longitude_column].max() + padding)
    ax.set_ylim(gages[latitude_column].min() - padding, gages[latitude_column].max() + padding)

    ax.tick_params(axis="both", labelsize=6)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    if show_plot:
        plt.show()

    return fig
