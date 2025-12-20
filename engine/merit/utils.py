"""A utils file for building network matrices for MERIT vectorized flowlines"""

import geopandas as gpd
import polars as pl
import rustworkx as rx
from tqdm import tqdm


def _build_upstream_dict_from_merit(
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
    # Convert to polars (drop geometry)
    df = pl.DataFrame(fp.drop(columns="geometry"))

    # Build connections from up1, up2, up3, up4
    # Create a long-form dataframe with all upstream connections
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

    # Concatenate all connections
    all_connections = pl.concat(connections)

    # Group by downstream COMID and aggregate upstream COMIDs
    upstream_dict_df = all_connections.group_by("dn_comid").agg(
        pl.col("up_comid").sort().alias("upstream_list")
    )

    # Convert to dictionary
    return dict(
        zip(
            upstream_dict_df["dn_comid"].to_list(),
            upstream_dict_df["upstream_list"].to_list(),
            strict=False,
        )
    )


def _build_rustworkx_object(
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

    # Add all nodes first
    for to_comid in tqdm(sorted(upstream_network.keys()), desc="Adding nodes"):
        from_comids = upstream_network[to_comid]
        if to_comid not in node_indices:
            node_indices[to_comid] = graph.add_node(to_comid)
        for from_comid in from_comids:
            if from_comid not in node_indices:
                node_indices[from_comid] = graph.add_node(from_comid)

    # Add edges
    for to_comid, from_comids in tqdm(upstream_network.items(), desc="Adding edges"):
        for from_comid in from_comids:
            graph.add_edge(node_indices[from_comid], node_indices[to_comid], None)

    return graph, node_indices
