"""Graph construction and traversal utilities for Lynker Hydrofabric flowpaths."""

from typing import Any

import polars as pl
import rustworkx as rx

from ddr.geodatazoo.dataclasses import Gauge


def find_origin(gauge: Gauge, fp: pl.LazyFrame, network: pl.LazyFrame) -> Any:
    """
    Find the origin flowpath ID for a gauge by querying the network.

    Parameters
    ----------
    gauge : Gauge
        A pydantic object containing gauge information (STAID, DRAIN_SQKM).
    fp : pl.LazyFrame
        The hydrofabric flowpaths table with 'id' and 'tot_drainage_areasqkm'.
    network : pl.LazyFrame
        The hydrofabric network table with 'id', 'toid', and 'hl_uri'.

    Returns
    -------
    str
        The flowpath ID (e.g., "wb-123") associated with the gauge.

    Raises
    ------
    ValueError
        If no flowpath is found for the gauge.

    Notes
    -----
    If multiple flowpaths match the gauge, the one with drainage area closest
    to the gauge's DRAIN_SQKM is selected.
    """
    try:
        flowpaths = (
            network.filter(
                pl.col("hl_uri") == f"gages-{gauge.STAID}"  # Finding the matching gauge
            )
            .select(
                pl.col("id")  # Select the `wb` values
            )
            .collect()
            .to_numpy()
            .squeeze()
        )
        if flowpaths.size > 1:
            return (
                fp.filter(
                    pl.col("id").is_in(flowpaths)  # finds the rows with matching IDs
                )
                .with_columns(
                    (pl.col("tot_drainage_areasqkm") - gauge.DRAIN_SQKM)
                    .abs()
                    .alias("diff")  # creates a new column with the DA diference from the USGS Gauge
                )
                .sort("diff")
                .head(1)
                .select("id")
                .collect()
                .item()
            )  # Selects the flowpath with the smallest difference
        else:
            return flowpaths.item()
    except ValueError as e:
        raise ValueError from e


def subset(origin: str, wb_network_dict: dict[str, list[str]]) -> list[tuple[str, str]]:
    """
    Find all upstream watershed boundary connections from an origin flowpath.

    Parameters
    ----------
    origin : str
        The starting flowpath ID from which to find upstream connections.
    wb_network_dict : dict[str, list[str]]
        Dictionary mapping downstream ID (toid) to list of upstream IDs.
        The upstream lists should be pre-sorted for deterministic output.

    Returns
    -------
    list[tuple[str, str]]
        List of (downstream_id, upstream_id) connection tuples.
        tuple[0] is the downstream (toid), tuple[1] is the upstream (from_id).

    Notes
    -----
    Relies on wb_network_dict having sorted upstream_ids lists (from preprocess_river_network).
    The recursive traversal processes upstream IDs in sorted order for determinism.
    """
    upstream_segments = set()
    connections = []

    def trace_upstream_recursive(current_id: str) -> None:
        """Recursively trace upstream from current_id."""
        if current_id in upstream_segments:
            return
        upstream_segments.add(current_id)

        # Find all segments that flow into current_id
        if current_id in wb_network_dict:
            for upstream_id in wb_network_dict[current_id]:
                connections.append(
                    (current_id, upstream_id)
                )  # Row is where the flow is going, col is where the flow is coming from
                if upstream_id not in upstream_segments:
                    trace_upstream_recursive(upstream_id)

    trace_upstream_recursive(origin)
    return connections


def preprocess_river_network(network: pl.LazyFrame) -> dict[str, list[str]]:
    """
    Preprocess the network to build a downstream-to-upstream mapping dictionary.

    Connections are ordered by the key being the toid (downstream segment),
    and the values being ids (upstream segments) sorted alphabetically for determinism.

    Parameters
    ----------
    network : pl.LazyFrame
        Network dataframe with 'id' and 'toid' columns.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping downstream segment to sorted list of upstream segments.

    Notes
    -----
    The upstream_ids list is sorted for deterministic output ordering.
    """
    network_dict = (
        network.filter(pl.col("toid").is_not_null())
        .group_by("toid")
        .agg(pl.col("id").sort().alias("upstream_ids"))
        .collect()
    )

    # Create a lookup for nexus -> downstream wb connections
    nexus_downstream = (
        network.filter(pl.col("id").str.starts_with("nex-"))
        .filter(pl.col("toid").str.starts_with("wb-"))
        .select(["id", "toid"])
        .rename({"id": "nexus_id", "toid": "downstream_wb"})
    ).collect()

    # Explode the upstream_ids to get one row per connection
    connections = network_dict.with_row_index().explode("upstream_ids")

    # Separate wb-to-wb connections (keep as-is)
    wb_to_wb = (
        connections.filter(pl.col("upstream_ids").str.starts_with("wb-"))
        .filter(pl.col("toid").str.starts_with("wb-"))
        .select(["toid", "upstream_ids"])
    )

    # Handle nexus connections: wb -> nex -> wb becomes wb -> wb
    wb_to_nexus = (
        connections.filter(pl.col("upstream_ids").str.starts_with("wb-"))
        .filter(pl.col("toid").str.starts_with("nex-"))
        .join(nexus_downstream, left_on="toid", right_on="nexus_id", how="inner")
        .select(["downstream_wb", "upstream_ids"])
        .rename({"downstream_wb": "toid"})
    )

    # Combine both types of connections
    wb_connections = pl.concat([wb_to_wb, wb_to_nexus]).unique()

    # Group back to dictionary format with sorting for determinism
    wb_network_result = (
        wb_connections.group_by("toid").agg(pl.col("upstream_ids").sort().alias("upstream_ids")).unique()
    )
    wb_network_dict = {row["toid"]: row["upstream_ids"] for row in wb_network_result.iter_rows(named=True)}
    return wb_network_dict


def build_graph_from_wb_network(
    wb_network_dict: dict[str, list[str]],
) -> tuple[rx.PyDiGraph, dict[str, int]]:
    """
    Build a RustWorkX directed graph from wb_network_dict.

    Parameters
    ----------
    wb_network_dict : dict[str, list[str]]
        Dictionary mapping downstream wb ID to list of upstream wb IDs.

    Returns
    -------
    tuple[rx.PyDiGraph, dict[str, int]]
        Graph and mapping of wb ID to graph node index.

    Notes
    -----
    Node indices are assigned in sorted order of wb IDs for deterministic behavior.
    """
    graph = rx.PyDiGraph(check_cycle=False)
    node_indices: dict[str, int] = {}

    # Collect all IDs and sort for deterministic node index assignment
    all_ids_set = set(wb_network_dict.keys())
    for upstream_list in wb_network_dict.values():
        all_ids_set.update(upstream_list)

    all_ids = sorted(all_ids_set)

    for wb_id in all_ids:
        node_indices[wb_id] = graph.add_node(wb_id)

    # Add edges (upstream -> downstream) - iterate sorted keys for determinism
    for to_id in sorted(wb_network_dict.keys()):
        from_ids = wb_network_dict[to_id]
        for from_id in from_ids:
            graph.add_edge(node_indices[from_id], node_indices[to_id], None)

    return graph, node_indices


def subset_upstream(
    origin: str,
    graph: rx.PyDiGraph,
    node_indices: dict[str, int],
) -> list[str]:
    """
    Find all upstream catchment IDs from the origin.

    Parameters
    ----------
    origin : str
        The catchment ID to start from
    graph : rx.PyDiGraph
        The river network graph
    node_indices : dict[str, int]
        Mapping of catchment ID to graph node index

    Returns
    -------
    list[str]
        List of all catchment IDs in the subset (including origin)
    """
    if origin not in node_indices:
        return [origin]

    origin_idx = node_indices[origin]
    ancestor_indices = rx.ancestors(graph, origin_idx)
    ancestor_ids = [graph.get_node_data(idx) for idx in ancestor_indices]

    return ancestor_ids + [origin]
