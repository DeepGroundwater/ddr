"""Tests for graph.py - graph construction and traversal utilities."""

import rustworkx as rx
from ddr_engine.merit import build_graph, build_upstream_dict, subset_upstream


class TestBuildUpstreamDict:
    """Tests for build_upstream_dict function."""

    def test_returns_dict(self, mock_merit_fp):
        result = build_upstream_dict(mock_merit_fp)
        assert isinstance(result, dict)

    def test_correct_keys(self, sandbox_upstream_dict):
        """Only nodes with upstream connections should be keys."""
        assert set(sandbox_upstream_dict.keys()) == {30, 50}

    def test_node_30_upstreams(self, sandbox_upstream_dict):
        """Node 30 receives from 10 and 20."""
        assert set(sandbox_upstream_dict[30]) == {10, 20}

    def test_node_50_upstreams(self, sandbox_upstream_dict):
        """Node 50 receives from 30 and 40."""
        assert set(sandbox_upstream_dict[50]) == {30, 40}

    def test_headwaters_not_keys(self, sandbox_upstream_dict):
        """Headwater nodes (10, 20, 40) have no upstream, not in dict."""
        for comid in [10, 20, 40]:
            assert comid not in sandbox_upstream_dict


class TestBuildGraph:
    """Tests for build_graph function."""

    def test_returns_tuple(self, sandbox_upstream_dict):
        result = build_graph(sandbox_upstream_dict)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_graph_type(self, sandbox_graph_obj):
        assert isinstance(sandbox_graph_obj, rx.PyDiGraph)

    def test_node_indices_type(self, sandbox_node_indices):
        assert isinstance(sandbox_node_indices, dict)

    def test_node_count(self, sandbox_graph_obj):
        """Graph should have 5 nodes."""
        assert sandbox_graph_obj.num_nodes() == 5

    def test_edge_count(self, sandbox_graph_obj):
        """Graph should have 4 edges: 10→30, 20→30, 30→50, 40→50."""
        assert sandbox_graph_obj.num_edges() == 4

    def test_all_comids_indexed(self, sandbox_node_indices):
        """All 5 COMIDs should have node indices."""
        assert set(sandbox_node_indices.keys()) == {10, 20, 30, 40, 50}

    def test_predecessors_of_30(self, sandbox_graph_obj, sandbox_node_indices):
        """Node 30 should have predecessors 10 and 20."""
        predecessors = {
            sandbox_graph_obj.get_node_data(n)
            for n in sandbox_graph_obj.predecessors(sandbox_node_indices[30])
        }
        assert predecessors == {10, 20}

    def test_predecessors_of_50(self, sandbox_graph_obj, sandbox_node_indices):
        """Node 50 should have predecessors 30 and 40."""
        predecessors = {
            sandbox_graph_obj.get_node_data(n)
            for n in sandbox_graph_obj.predecessors(sandbox_node_indices[50])
        }
        assert predecessors == {30, 40}

    def test_successors_of_10(self, sandbox_graph_obj, sandbox_node_indices):
        """Node 10 should have successor 30."""
        successors = [
            sandbox_graph_obj.get_node_data(n) for n in sandbox_graph_obj.successors(sandbox_node_indices[10])
        ]
        assert successors == [30]

    def test_outlet_has_no_successors(self, sandbox_graph_obj, sandbox_node_indices):
        """Node 50 (outlet) should have no successors."""
        successors = list(sandbox_graph_obj.successors(sandbox_node_indices[50]))
        assert len(successors) == 0

    def test_headwaters_have_no_predecessors(self, sandbox_graph_obj, sandbox_node_indices):
        """Headwater nodes should have no predecessors."""
        for comid in [10, 20, 40]:
            predecessors = list(sandbox_graph_obj.predecessors(sandbox_node_indices[comid]))
            assert len(predecessors) == 0, f"Node {comid} should have no predecessors"


class TestSubsetUpstream:
    """Tests for subset_upstream function."""

    def test_outlet_returns_all_nodes(self, sandbox_graph_obj, sandbox_node_indices):
        """Subsetting from outlet (50) should return all nodes."""
        subset = subset_upstream(50, sandbox_graph_obj, sandbox_node_indices)
        assert set(subset) == {10, 20, 30, 40, 50}

    def test_intermediate_node(self, sandbox_graph_obj, sandbox_node_indices):
        """Subsetting from node 30 should return 10, 20, 30."""
        subset = subset_upstream(30, sandbox_graph_obj, sandbox_node_indices)
        assert set(subset) == {10, 20, 30}

    def test_headwater_returns_self(self, sandbox_graph_obj, sandbox_node_indices):
        """Subsetting from headwater should return just itself."""
        for comid in [10, 20, 40]:
            subset = subset_upstream(comid, sandbox_graph_obj, sandbox_node_indices)
            assert subset == [comid]

    def test_unknown_comid_returns_self(self, sandbox_graph_obj, sandbox_node_indices):
        """Unknown COMID should return just itself."""
        subset = subset_upstream(99999, sandbox_graph_obj, sandbox_node_indices)
        assert subset == [99999]

    def test_subset_includes_origin(self, sandbox_graph_obj, sandbox_node_indices):
        """Subset should always include the origin COMID."""
        for comid in [10, 20, 30, 40, 50]:
            subset = subset_upstream(comid, sandbox_graph_obj, sandbox_node_indices)
            assert comid in subset
