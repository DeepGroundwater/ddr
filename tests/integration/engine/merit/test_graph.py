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

    def test_node_indices_type(self, sandbox_node_indices):
        assert isinstance(sandbox_node_indices, dict)

    def test_node_count(self, G):
        """Graph should have 5 nodes."""
        assert G.num_nodes() == 5

    def test_edge_count(self, G):
        """Graph should have 4 edges: 10→30, 20→30, 30→50, 40→50."""
        assert G.num_edges() == 4

    def test_all_comids_indexed(self, sandbox_node_indices):
        """All 5 COMIDs should have node indices."""
        assert set(sandbox_node_indices.keys()) == {10, 20, 30, 40, 50}

    def test_predecessors_of_30(self, G, sandbox_node_indices):
        """Node 30 should have predecessors 10 and 20."""
        ancestor_indices = rx.ancestors(G, sandbox_node_indices[30])
        ancestor_values = {G.get_node_data(idx) for idx in ancestor_indices}
        assert ancestor_values == {10, 20}

    def test_successors_of_10(self, G, sandbox_node_indices):
        """Node 10 should have successor 30."""
        descendants_indices = rx.descendants(G, sandbox_node_indices[10])
        descendants_values = {G.get_node_data(idx) for idx in descendants_indices}
        assert descendants_values == {30, 50}

    def test_outlet_has_no_successors(self, G, sandbox_node_indices):
        """Node 50 (outlet) should have no successors."""
        successors = list(G.successors(sandbox_node_indices[50]))
        assert len(successors) == 0

    def test_headwaters_have_no_predecessors(self, G, sandbox_node_indices):
        """Headwater nodes should have no predecessors."""
        for comid in [10, 20, 40]:
            predecessors = list(G.predecessors(sandbox_node_indices[comid]))
            assert len(predecessors) == 0, f"Node {comid} should have no predecessors"


class TestSubsetUpstream:
    """Tests for subset_upstream function."""

    def test_outlet_returns_all_nodes(self, G, sandbox_node_indices):
        """Subsetting from outlet (50) should return all nodes."""
        subset = subset_upstream(50, G, sandbox_node_indices)
        assert set(subset) == {10, 20, 30, 40, 50}

    def test_intermediate_node(self, G, sandbox_node_indices):
        """Subsetting from node 30 should return 10, 20, 30."""
        subset = subset_upstream(30, G, sandbox_node_indices)
        assert set(subset) == {10, 20, 30}

    def test_headwater_returns_self(self, G, sandbox_node_indices):
        """Subsetting from headwater should return just itself."""
        for comid in [10, 20, 40]:
            subset = subset_upstream(comid, G, sandbox_node_indices)
            assert subset == [comid]

    def test_unknown_comid_returns_self(self, G, sandbox_node_indices):
        """Unknown COMID should return just itself."""
        subset = subset_upstream(99999, G, sandbox_node_indices)
        assert subset == [99999]

    def test_subset_includes_origin(self, G, sandbox_node_indices):
        """Subset should always include the origin COMID."""
        for comid in [10, 20, 30, 40, 50]:
            subset = subset_upstream(comid, G, sandbox_node_indices)
            assert comid in subset
