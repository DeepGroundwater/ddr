"""Adapter to convert DDR COO adjacency matrices to DiffRoute-compatible NetworkX graphs.

This module provides utilities to convert the sparse COO adjacency matrices
created by the DDR Engine into NetworkX DiGraphs that can be used with DiffRoute.

COO Matrix Format (from DDR Engine zarr stores):
    - indices_0: Row indices (downstream segment indices)
    - indices_1: Column indices (upstream segment indices)
    - values: Matrix values (typically 1 for adjacency)
    - order: Topological sort order as segment IDs

DiffRoute expects:
    - NetworkX DiGraph with edges from upstream → downstream
    - Node parameters as a pandas DataFrame indexed by node ID
    - Parameters depend on IRF model (e.g., k/x for muskingum, tau for linear_storage)
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy import sparse


def zarr_group_to_networkx(zarr_group: zarr.Group) -> nx.DiGraph:
    """Convert a zarr Group containing a COO adjacency matrix to a NetworkX DiGraph.

    This is the primary function for converting DDR Engine zarr stores to
    DiffRoute-compatible graphs. It takes a zarr.Group object directly.

    The COO matrix from DDR Engine has semantics:
        matrix[downstream_idx, upstream_idx] = 1

    DiffRoute expects edges from upstream → downstream, so we create
    edges in that direction.

    Parameters
    ----------
    zarr_group : zarr.Group
        Zarr group containing the COO adjacency matrix with arrays:
        - indices_0: Row indices (downstream)
        - indices_1: Column indices (upstream)
        - values: Matrix values
        - order: Topological sort order as segment IDs

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph with edges from upstream to downstream.
        Node IDs are the segment IDs from the order array.

    Examples
    --------
    >>> import zarr
    >>> root = zarr.open_group("path/to/adjacency.zarr", mode="r")
    >>> G = zarr_group_to_networkx(root)
    """
    # Extract COO components from zarr group
    row = zarr_group["indices_0"][:]
    col = zarr_group["indices_1"][:]
    order = zarr_group["order"][:]

    G = nx.DiGraph()

    # Add all nodes (some may have no edges)
    for node_id in order:
        G.add_node(int(node_id))

    # Add edges: upstream → downstream
    # In COO matrix: row = downstream index, col = upstream index
    for row_idx, col_idx in zip(row, col, strict=False):
        upstream_id = int(order[col_idx])
        downstream_id = int(order[row_idx])
        G.add_edge(upstream_id, downstream_id)

    return G


def coo_to_networkx(
    coo_matrix: sparse.coo_matrix,
    order: NDArray[np.int32],
) -> nx.DiGraph:
    """Convert a COO adjacency matrix to a NetworkX DiGraph.

    The COO matrix from DDR Engine has semantics:
        matrix[downstream_idx, upstream_idx] = 1

    DiffRoute expects edges from upstream → downstream, so we create
    edges in that direction.

    Parameters
    ----------
    coo_matrix : sparse.coo_matrix
        Sparse COO adjacency matrix where row=downstream, col=upstream.
    order : NDArray[np.int32]
        Array of segment IDs in topological order. Index i corresponds
        to row/col i in the matrix.

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph with edges from upstream to downstream.
        Node IDs are the segment IDs from the order array.
    """
    G = nx.DiGraph()

    # Add all nodes (some may have no edges)
    for node_id in order:
        G.add_node(int(node_id))

    # Add edges: upstream → downstream
    # In COO matrix: row = downstream index, col = upstream index
    for row_idx, col_idx in zip(coo_matrix.row, coo_matrix.col, strict=False):
        upstream_id = int(order[col_idx])
        downstream_id = int(order[row_idx])
        G.add_edge(upstream_id, downstream_id)

    return G


def zarr_to_networkx(zarr_path: str | Path) -> nx.DiGraph:
    """Load a COO adjacency matrix from zarr path and convert to NetworkX DiGraph.

    This is a convenience wrapper around zarr_group_to_networkx that handles
    opening the zarr store.

    Parameters
    ----------
    zarr_path : Union[str, Path]
        Path to the zarr store containing the COO adjacency matrix.

    Returns
    -------
    nx.DiGraph
        NetworkX directed graph with edges from upstream to downstream.
    """
    root = zarr.open_group(store=zarr_path, mode="r")
    return zarr_group_to_networkx(root)


def create_param_df(
    order: NDArray[np.int32],
    k: NDArray | list[float] | None = None,
    x: NDArray | list[float] | None = None,
    tau: NDArray | list[float] | None = None,
    delay: NDArray | list[float] | None = None,
    k_units: str = "seconds",
) -> pd.DataFrame:
    """Create a parameter DataFrame for DiffRoute.

    DiffRoute expects parameters as a pandas DataFrame indexed by node ID.
    Different IRF models require different parameters:
        - muskingum: k, x
        - linear_storage: tau
        - pure_lag: delay

    Parameters
    ----------
    order : NDArray[np.int32]
        Array of segment IDs in topological order.
    k : array-like, optional
        Muskingum k parameter (storage constant). If k_units="seconds",
        will be converted to days for DiffRoute.
    x : array-like, optional
        Muskingum x parameter (weighting factor, typically 0-0.5).
    tau : array-like, optional
        Time constant for linear_storage model. If None and k is provided,
        will be derived from k.
    delay : array-like, optional
        Delay parameter for pure_lag model. If None and k is provided,
        will be derived from k.
    k_units : str, default "seconds"
        Units of the k parameter. If "seconds", converts to days.
        Use "days" if k is already in days.

    Returns
    -------
    pd.DataFrame
        Parameter DataFrame indexed by segment ID with columns for
        available parameters.
    """
    n = len(order)
    data = {}

    # Convert k to days if needed
    k_in_days = None
    if k is not None:
        k_array = np.asarray(k, dtype=np.float64)
        if k_units == "seconds":
            k_in_days = k_array / (3600 * 24)
        else:
            k_in_days = k_array
        data["k"] = k_in_days

    if x is not None:
        data["x"] = np.asarray(x, dtype=np.float64)

    # Derive tau from k if not provided
    if tau is not None:
        data["tau"] = np.asarray(tau, dtype=np.float64)
    elif k_in_days is not None:
        data["tau"] = k_in_days

    # Derive delay from k if not provided
    if delay is not None:
        data["delay"] = np.asarray(delay, dtype=np.float64)
    elif k_in_days is not None:
        data["delay"] = k_in_days

    # Ensure we have at least empty columns for common parameters
    if not data:
        # Create placeholder with zeros if no parameters provided
        data["k"] = np.zeros(n)
        data["x"] = np.full(n, 0.3)  # Default x value
        data["tau"] = np.zeros(n)
        data["delay"] = np.zeros(n)

    return pd.DataFrame(data, index=order.tolist())


def load_rapid_params(
    k_file: str | Path,
    x_file: str | Path,
    reach_id_file: str | Path,
) -> tuple[list[int], list[float], list[float]]:
    """Load Muskingum parameters from RAPID-style CSV files.

    RAPID stores parameters in separate CSV files without headers,
    one value per line, in the same order as the reach ID file.

    Parameters
    ----------
    k_file : Union[str, Path]
        Path to k parameter CSV file.
    x_file : Union[str, Path]
        Path to x parameter CSV file.
    reach_id_file : Union[str, Path]
        Path to reach ID CSV file.

    Returns
    -------
    tuple[list[int], list[float], list[float]]
        Tuple of (reach_ids, k_values, x_values).
    """
    reach_ids = pd.read_csv(reach_id_file, header=None).squeeze().tolist()
    k_vals = pd.read_csv(k_file, header=None).squeeze().tolist()
    x_vals = pd.read_csv(x_file, header=None).squeeze().tolist()

    # Handle single-value case where squeeze returns a scalar
    if not isinstance(reach_ids, list):
        reach_ids = [reach_ids]
    if not isinstance(k_vals, list):
        k_vals = [k_vals]
    if not isinstance(x_vals, list):
        x_vals = [x_vals]

    return reach_ids, k_vals, x_vals


def build_diffroute_inputs(
    zarr_path: str | Path,
    k_file: str | Path | None = None,
    x_file: str | Path | None = None,
    reach_id_file: str | Path | None = None,
    k: NDArray | list[float] | None = None,
    x: NDArray | list[float] | None = None,
    k_units: str = "seconds",
) -> tuple[nx.DiGraph, pd.DataFrame]:
    """Build DiffRoute inputs from a zarr COO adjacency matrix.

    This is a convenience function that loads the graph and creates
    the parameter DataFrame in one call.

    Parameters
    ----------
    zarr_path : Union[str, Path]
        Path to the zarr store containing the COO adjacency matrix.
    k_file : Union[str, Path], optional
        Path to RAPID-style k parameter CSV file.
    x_file : Union[str, Path], optional
        Path to RAPID-style x parameter CSV file.
    reach_id_file : Union[str, Path], optional
        Path to RAPID-style reach ID CSV file.
    k : array-like, optional
        Muskingum k values (alternative to k_file).
    x : array-like, optional
        Muskingum x values (alternative to x_file).
    k_units : str, default "seconds"
        Units of the k parameter.

    Returns
    -------
    tuple[nx.DiGraph, pd.DataFrame]
        Tuple of (graph, param_df) ready for DiffRoute.
    """
    # Load graph
    G = zarr_to_networkx(zarr_path)

    # Load parameters
    root = zarr.open_group(store=zarr_path, mode="r")
    order = root["order"][:]

    if k_file is not None and x_file is not None and reach_id_file is not None:
        reach_ids, k_vals, x_vals = load_rapid_params(k_file, x_file, reach_id_file)
        # Reorder parameters to match zarr order
        id_to_idx = {rid: i for i, rid in enumerate(reach_ids)}
        k_ordered = [k_vals[id_to_idx[int(rid)]] for rid in order]
        x_ordered = [x_vals[id_to_idx[int(rid)]] for rid in order]
        param_df = create_param_df(order, k=k_ordered, x=x_ordered, k_units=k_units)
    elif k is not None and x is not None:
        param_df = create_param_df(order, k=k, x=x, k_units=k_units)
    else:
        # Create default parameters
        param_df = create_param_df(order)

    return G, param_df
