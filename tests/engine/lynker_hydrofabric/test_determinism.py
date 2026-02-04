"""Tests for deterministic output of lynker_hydrofabric adjacency matrix creation.

These tests verify that running the same operations multiple times produces
identical results, which is critical for reproducible scientific workflows.
"""

import polars as pl
from ddr_engine.lynker_hydrofabric import (
    build_graph_from_wb_network,
    create_coo,
    preprocess_river_network,
    subset,
)


class TestDeterministicPreprocessing:
    """Tests for deterministic preprocessing."""

    def test_preprocess_river_network_deterministic(self, sandbox_network: pl.LazyFrame) -> None:
        """Verify preprocess_river_network produces identical output across runs."""
        results = []
        for _ in range(5):
            result = preprocess_river_network(sandbox_network)
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0].keys() == results[i].keys()
            for key in results[0]:
                assert results[0][key] == results[i][key], f"Non-deterministic output for key {key}"

    def test_upstream_lists_are_sorted(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Verify upstream ID lists are sorted."""
        for key, upstream_list in sandbox_wb_network_dict.items():
            assert upstream_list == sorted(upstream_list), (
                f"Upstream list for {key} is not sorted: {upstream_list}"
            )


class TestDeterministicGraphBuilding:
    """Tests for deterministic graph building."""

    def test_build_graph_deterministic(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Verify build_graph_from_wb_network produces identical output."""
        results = []
        for _ in range(5):
            graph, node_indices = build_graph_from_wb_network(sandbox_wb_network_dict)
            results.append((graph.num_nodes(), graph.num_edges(), dict(node_indices)))

        for i in range(1, len(results)):
            assert results[0] == results[i], f"Non-deterministic graph building: run 0 != run {i}"

    def test_node_indices_are_deterministic(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Verify node indices are assigned in deterministic order."""
        _, node_indices = build_graph_from_wb_network(sandbox_wb_network_dict)

        # Node indices should be assigned in sorted order of IDs
        sorted_ids = sorted(node_indices.keys())
        for i, wb_id in enumerate(sorted_ids):
            assert node_indices[wb_id] == i, f"Node {wb_id} has index {node_indices[wb_id]}, expected {i}"


class TestDeterministicSubset:
    """Tests for deterministic subsetting."""

    def test_subset_deterministic(self, sandbox_wb_network_dict: dict[str, list[str]]) -> None:
        """Verify subset produces identical connections across runs."""
        results = []
        for _ in range(5):
            connections = subset("wb-50", sandbox_wb_network_dict)
            results.append(connections)

        for i in range(1, len(results)):
            assert results[0] == results[i], f"Non-deterministic subset: run 0 != run {i}"


class TestDeterministicCOO:
    """Tests for deterministic COO matrix creation."""

    def test_create_coo_deterministic(
        self,
        sandbox_connections: list[tuple[str, str]],
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Verify create_coo produces identical output across runs."""
        results = []
        for _ in range(5):
            coo, flowpaths = create_coo(sandbox_connections, sandbox_conus_mapping)
            results.append((coo.row.tolist(), coo.col.tolist(), flowpaths))

        for i in range(1, len(results)):
            assert results[0] == results[i], f"Non-deterministic COO creation: run 0 != run {i}"

    def test_create_coo_returns_list_not_set(
        self,
        sandbox_connections: list[tuple[str, str]],
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Verify create_coo returns list, not set."""
        _, flowpaths = create_coo(sandbox_connections, sandbox_conus_mapping)
        assert isinstance(flowpaths, list), f"Expected list, got {type(flowpaths)}"

    def test_create_coo_flowpaths_sorted_by_conus_index(
        self,
        sandbox_connections: list[tuple[str, str]],
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Verify flowpaths are sorted by CONUS mapping index."""
        _, flowpaths = create_coo(sandbox_connections, sandbox_conus_mapping)

        # Verify sorting
        indices = [sandbox_conus_mapping[fp] for fp in flowpaths]
        assert indices == sorted(indices), f"Flowpaths not sorted by CONUS index: {flowpaths}"
