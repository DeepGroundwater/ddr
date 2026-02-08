"""Graph construction and traversal utilities for MERIT flowpaths"""

import geopandas as gpd
import polars as pl
import rustworkx as rx
from tqdm import tqdm


def build_upstream_dict(
    fp: gpd.GeoDataFrame,
) -> dict[int, list[int]]:
    """
    Build upstream connectivity dictionary from MERIT flowpaths.

    Parameters
    ----------
    fp : gpd.GeoDataFrame
        Flowpaths with COMID, NextDownID, and up1-up4 columns.

    Returns
    -------
    dict[int, list[int]]
        Dictionary mapping downstream COMID to list of upstream COMIDs.
    """
    df = pl.DataFrame(fp.drop(columns="geometry"))

    connections = []
    for up_col in ["up1", "up2", "up3", "up4"]:
        conn = df.select(
            [
                pl.col("COMID").cast(pl.Int32).alias("dn_comid"),
                pl.col(up_col).cast(pl.Int32).alias("up_comid"),
            ]
        ).filter(pl.col("up_comid") > 0)
        connections.append(conn)

    if not connections:
        return {}

    all_connections = pl.concat(connections)

    upstream_dict_df = all_connections.group_by("dn_comid").agg(
        pl.col("up_comid").sort().alias("upstream_list")
    )

    return dict(
        zip(
            upstream_dict_df["dn_comid"].to_list(),
            upstream_dict_df["upstream_list"].to_list(),
            strict=False,
        )
    )


def build_graph(
    upstream_network: dict[int, list[int]],
) -> tuple[rx.PyDiGraph, dict[int, int]]:
    """
    Build a RustWorkX directed graph from upstream network dictionary.

    Parameters
    ----------
    upstream_network : dict[int, list[int]]
        Dictionary mapping downstream COMID to list of upstream COMIDs.

    Returns
    -------
    tuple[rx.PyDiGraph, dict[int, int]]
        Graph and mapping of COMID to graph node index.
    """
    graph = rx.PyDiGraph(check_cycle=False)
    node_indices: dict[int, int] = {}

    for to_comid in tqdm(sorted(upstream_network.keys()), desc="Adding nodes", ncols=140, ascii=True):
        from_comids = upstream_network[to_comid]
        if to_comid not in node_indices:
            node_indices[to_comid] = graph.add_node(to_comid)
        for from_comid in from_comids:
            if from_comid not in node_indices:
                node_indices[from_comid] = graph.add_node(from_comid)

    for to_comid, from_comids in tqdm(upstream_network.items(), desc="Adding edges", ncols=140, ascii=True):
        for from_comid in from_comids:
            graph.add_edge(node_indices[from_comid], node_indices[to_comid], None)

    return graph, node_indices


def subset_upstream(
    origin_comid: int,
    graph: rx.PyDiGraph,
    node_indices: dict[int, int],
) -> list[int]:
    """
    Find all upstream COMIDs from the origin using graph ancestors.

    Parameters
    ----------
    origin_comid : int
        The COMID to start from
    graph : rx.PyDiGraph
        The river network graph
    node_indices : dict[int, int]
        Mapping of COMID to graph node index

    Returns
    -------
    list[int]
        List of all COMIDs in the subset (including origin)
    """
    if origin_comid not in node_indices:
        return [origin_comid]

    origin_node_idx = node_indices[origin_comid]
    ancestor_indices = rx.ancestors(graph, origin_node_idx)
    ancestor_comids = [graph.get_node_data(idx) for idx in ancestor_indices]

    return ancestor_comids + [origin_comid]
