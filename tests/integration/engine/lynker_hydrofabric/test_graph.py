"""Tests for graph.py - graph construction and traversal utilities."""

import polars as pl
import rustworkx as rx
from ddr_engine.lynker_hydrofabric import (
    build_graph_from_wb_network,
    preprocess_river_network,
    subset,
)


class TestPreprocessRiverNetwork:
    """Tests for preprocess_river_network function."""

    def test_returns_dict(self, sandbox_network: pl.LazyFrame) -> None:
        """Result should be a dictionary."""
        result = preprocess_river_network(sandbox_network)
        assert isinstance(result, dict)

    def test_correct_keys(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Only downstream nodes with upstream connections should be keys."""
        # wb-30 receives from wb-10, wb-20
        # wb-50 receives from wb-30, wb-40
        assert set(sandbox_wb_network_dict.keys()) == {"wb-30", "wb-50"}

    def test_wb_30_upstreams(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """wb-30 receives from wb-10 and wb-20."""
        assert set(sandbox_wb_network_dict["wb-30"]) == {"wb-10", "wb-20"}

    def test_wb_50_upstreams(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """wb-50 receives from wb-30 and wb-40."""
        assert set(sandbox_wb_network_dict["wb-50"]) == {"wb-30", "wb-40"}

    def test_headwaters_not_keys(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Headwater nodes have no upstream, so not in dict."""
        for wb_id in ["wb-10", "wb-20", "wb-40"]:
            assert wb_id not in sandbox_wb_network_dict


class TestBuildGraphFromWbNetwork:
    """Tests for build_graph_from_wb_network function."""

    def test_returns_tuple(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Result should be a tuple of (graph, node_indices)."""
        result = build_graph_from_wb_network(sandbox_wb_network_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_graph_type(self, sandbox_graph) -> None:
        """First element should be PyDiGraph."""
        graph, _ = sandbox_graph
        assert isinstance(graph, rx.PyDiGraph)

    def test_node_indices_type(self, sandbox_graph) -> None:
        """Second element should be dict."""
        _, node_indices = sandbox_graph
        assert isinstance(node_indices, dict)

    def test_node_count(self, sandbox_graph) -> None:
        """Graph should have 5 nodes."""
        graph, _ = sandbox_graph
        assert graph.num_nodes() == 5

    def test_edge_count(self, sandbox_graph) -> None:
        """Graph should have 4 edges."""
        graph, _ = sandbox_graph
        assert graph.num_edges() == 4

    def test_all_wb_ids_indexed(self, sandbox_graph) -> None:
        """All 5 watershed boundary IDs should have node indices."""
        _, node_indices = sandbox_graph
        expected_ids = {"wb-10", "wb-20", "wb-30", "wb-40", "wb-50"}
        assert set(node_indices.keys()) == expected_ids

    def test_predecessors_of_wb_30(self, sandbox_graph) -> None:
        """wb-30 should have predecessors wb-10 and wb-20."""
        graph, node_indices = sandbox_graph
        ancestor_indices = rx.ancestors(graph, node_indices["wb-30"])
        ancestor_values = {graph.get_node_data(idx) for idx in ancestor_indices}
        assert ancestor_values == {"wb-10", "wb-20"}

    def test_outlet_has_no_successors(self, sandbox_graph) -> None:
        """wb-50 (outlet) should have no successors."""
        graph, node_indices = sandbox_graph
        successors = list(graph.successors(node_indices["wb-50"]))
        assert len(successors) == 0


class TestSubset:
    """Tests for subset function."""

    def test_outlet_returns_all_connections(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Subsetting from outlet (wb-50) should return all connections."""
        connections = subset("wb-50", sandbox_wb_network_dict)
        # Should have 4 connections: wb-50<-wb-30, wb-50<-wb-40, wb-30<-wb-10, wb-30<-wb-20
        assert len(connections) == 4

    def test_intermediate_node(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Subsetting from wb-30 should return connections to wb-10, wb-20."""
        connections = subset("wb-30", sandbox_wb_network_dict)
        # Should have 2 connections: wb-30<-wb-10, wb-30<-wb-20
        assert len(connections) == 2

    def test_headwater_returns_empty(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Subsetting from headwater should return empty list."""
        for wb_id in ["wb-10", "wb-20", "wb-40"]:
            connections = subset(wb_id, sandbox_wb_network_dict)
            assert len(connections) == 0

    def test_connections_format(self, sandbox_connections: list[tuple[str, str]]) -> None:
        """Connections should be list of tuples (downstream, upstream)."""
        assert isinstance(sandbox_connections, list)
        for conn in sandbox_connections:
            assert isinstance(conn, tuple)
            assert len(conn) == 2
