"""Tests for build.py - adjacency matrix creation functions."""

import numpy as np
import pytest
import zarr
from ddr_engine.merit import build_gauge_adjacencies, build_merit_adjacency, create_adjacency_matrix
from scipy import sparse


class TestCreateAdjacencyMatrix:
    """Tests for create_adjacency_matrix function."""

    def test_returns_tuple(self, mock_merit_fp):
        result = create_adjacency_matrix(mock_merit_fp)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_is_coo(self, sandbox_coo_matrix):
        assert isinstance(sandbox_coo_matrix, sparse.coo_matrix)

    def test_matrix_shape(self, sandbox_coo_matrix):
        """Matrix should be 5x5 for 5 nodes."""
        assert sandbox_coo_matrix.shape == (5, 5)

    def test_matrix_nnz(self, sandbox_coo_matrix):
        """Matrix should have 4 non-zero entries (4 edges)."""
        assert sandbox_coo_matrix.nnz == 4

    def test_matrix_dtype(self, sandbox_coo_matrix):
        """Matrix should be uint8."""
        assert sandbox_coo_matrix.dtype == np.uint8

    def test_matrix_is_lower_triangular(self, sandbox_coo_matrix):
        """Matrix should be lower triangular (row >= col)."""
        assert np.all(sandbox_coo_matrix.row >= sandbox_coo_matrix.col)

    def test_topological_order_length(self, sandbox_ts_order):
        """Topological order should contain all 5 COMIDs."""
        assert len(sandbox_ts_order) == 5

    def test_topological_order_complete(self, sandbox_ts_order):
        """Topological order should contain exactly the 5 COMIDs."""
        assert set(sandbox_ts_order) == {10, 20, 30, 40, 50}

    def test_topological_order_valid(self, sandbox_comid_to_idx):
        """Upstream nodes must appear before downstream nodes."""
        idx = sandbox_comid_to_idx

        # 10, 20 must come before 30
        assert idx[10] < idx[30]
        assert idx[20] < idx[30]

        # 30, 40 must come before 50
        assert idx[30] < idx[50]
        assert idx[40] < idx[50]

    def test_matrix_matches_expected_network(
        self,
        sandbox_coo_matrix,
        sandbox_ts_order,
        expected_network_matrix,
    ):
        """
        Verify matrix matches expected network matrix from SANDBOX.md.

        Need to reorder expected matrix to match topological order.
        """
        # Expected order in SANDBOX.md is [10, 20, 30, 40, 50]
        # expected_order = [10, 20, 30, 40, 50]

        # Create permutation from expected order to actual topological order
        actual_dense = sandbox_coo_matrix.toarray()

        # Verify connectivity matches
        idx = {comid: i for i, comid in enumerate(sandbox_ts_order)}

        # Check each expected edge
        expected_edges = [
            (30, 10),  # 10 → 30
            (30, 20),  # 20 → 30
            (50, 30),  # 30 → 50
            (50, 40),  # 40 → 50
        ]

        for downstream, upstream in expected_edges:
            row, col = idx[downstream], idx[upstream]
            assert actual_dense[row, col] == 1, f"Missing edge {upstream}→{downstream}"

        # Verify no extra edges
        assert actual_dense.sum() == len(expected_edges)

    def test_matrix_encodes_correct_edges(self, sandbox_coo_matrix, sandbox_comid_to_idx):
        """Verify matrix[downstream_idx, upstream_idx] = 1 for each edge."""
        idx = sandbox_comid_to_idx
        dense = sandbox_coo_matrix.toarray()

        expected_edges = [
            (idx[30], idx[10]),
            (idx[30], idx[20]),
            (idx[50], idx[30]),
            (idx[50], idx[40]),
        ]

        for row, col in expected_edges:
            assert dense[row, col] == 1, f"Missing edge at ({row}, {col})"

        assert dense.sum() == len(expected_edges)

    def test_outlet_has_no_outgoing_edges(self, sandbox_coo_matrix, sandbox_comid_to_idx):
        """Node 50 (outlet) should not appear as a source (in col array)."""
        idx = sandbox_comid_to_idx
        assert idx[50] not in sandbox_coo_matrix.col


class TestBuildMeritAdjacency:
    """Tests for build_merit_adjacency function."""

    def test_creates_zarr_store(self, sandbox_zarr_path):
        """Zarr store should be created."""
        assert sandbox_zarr_path.exists()

    def test_raises_if_exists(self, tmp_path, mock_merit_fp):
        """Should raise FileExistsError if store already exists."""
        out_path = tmp_path / "test.zarr"

        # Create first time
        build_merit_adjacency(mock_merit_fp, out_path)

        # Should fail second time
        with pytest.raises(FileExistsError):
            build_merit_adjacency(mock_merit_fp, out_path)

    def test_returns_path(self, tmp_path, mock_merit_fp):
        """Should return the output path."""
        out_path = tmp_path / "test_return.zarr"
        result = build_merit_adjacency(mock_merit_fp, out_path)
        assert result == out_path


class TestBuildGaugeAdjacencies:
    """Tests for build_gauge_adjacencies function."""

    def test_creates_per_gauge_groups(self, tmp_path, mock_merit_fp, sandbox_zarr_path, sandbox_gauge_set):
        """Given GaugeSet with 2 gauges, zarr has 2 subgroups."""
        out_path = tmp_path / "gages.zarr"
        build_gauge_adjacencies(mock_merit_fp, sandbox_zarr_path, sandbox_gauge_set, out_path)
        root = zarr.open_group(store=out_path, mode="r")
        assert "00000030" in root
        assert "00000050" in root

    def test_headwater_gauge(self, tmp_path, mock_merit_fp, sandbox_zarr_path):
        """Headwater COMID gauge (10) → subgroup with 0 edges."""
        from tests.engine.merit.conftest import make_gauge_set

        headwater_gauges = make_gauge_set([{"STAID": "00000010", "COMID": 10}])
        out_path = tmp_path / "hw_gages.zarr"
        build_gauge_adjacencies(mock_merit_fp, sandbox_zarr_path, headwater_gauges, out_path)
        root = zarr.open_group(store=out_path, mode="r")
        assert "00000010" in root
        # Headwater has no edges (0 nnz in COO)
        indices_0 = root["00000010"]["indices_0"][:]
        assert len(indices_0) == 0

    def test_raises_if_exists(self, tmp_path, mock_merit_fp, sandbox_zarr_path, sandbox_gauge_set):
        """Second call to same path → FileExistsError."""
        out_path = tmp_path / "gages_dup.zarr"
        build_gauge_adjacencies(mock_merit_fp, sandbox_zarr_path, sandbox_gauge_set, out_path)
        with pytest.raises(FileExistsError):
            build_gauge_adjacencies(mock_merit_fp, sandbox_zarr_path, sandbox_gauge_set, out_path)
