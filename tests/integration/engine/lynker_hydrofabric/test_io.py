"""Tests for io.py - COO matrix creation utilities."""

import numpy as np
from ddr_engine.lynker_hydrofabric import create_coo
from scipy import sparse


class TestCreateCoo:
    """Tests for create_coo function."""

    def test_returns_tuple(
        self,
        sandbox_connections: list[tuple[str, str]],
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Result should be a tuple."""
        result = create_coo(sandbox_connections, sandbox_conus_mapping)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_matrix_is_coo(self, sandbox_coo_result) -> None:
        """First element should be COO matrix."""
        coo, _ = sandbox_coo_result
        assert isinstance(coo, sparse.coo_matrix)

    def test_flowpaths_is_list(self, sandbox_coo_result) -> None:
        """Second element should be list."""
        _, flowpaths = sandbox_coo_result
        assert isinstance(flowpaths, list)

    def test_matrix_is_lower_triangular(self, sandbox_coo_result) -> None:
        """Matrix should be lower triangular (row >= col)."""
        coo, _ = sandbox_coo_result
        assert np.all(coo.row >= coo.col)

    def test_matrix_nnz(self, sandbox_coo_result) -> None:
        """Matrix should have 4 non-zero entries (4 edges)."""
        coo, _ = sandbox_coo_result
        assert coo.nnz == 4

    def test_flowpaths_count(self, sandbox_coo_result) -> None:
        """Should have 5 unique flowpaths."""
        _, flowpaths = sandbox_coo_result
        assert len(flowpaths) == 5

    def test_flowpaths_complete(self, sandbox_coo_result) -> None:
        """Should contain all 5 watershed boundary IDs."""
        _, flowpaths = sandbox_coo_result
        expected = {"wb-10", "wb-20", "wb-30", "wb-40", "wb-50"}
        assert set(flowpaths) == expected

    def test_flowpaths_sorted(
        self,
        sandbox_coo_result,
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Flowpaths should be sorted by CONUS index."""
        _, flowpaths = sandbox_coo_result
        indices = [sandbox_conus_mapping[fp] for fp in flowpaths]
        assert indices == sorted(indices)

    def test_matrix_encodes_correct_edges(
        self,
        sandbox_coo_result,
        sandbox_conus_mapping: dict[str, int],
    ) -> None:
        """Verify matrix[downstream_idx, upstream_idx] = 1 for each edge."""
        coo, _ = sandbox_coo_result
        dense = coo.toarray()

        # Expected edges: (downstream, upstream)
        expected_edges = [
            ("wb-30", "wb-10"),
            ("wb-30", "wb-20"),
            ("wb-50", "wb-30"),
            ("wb-50", "wb-40"),
        ]

        for downstream, upstream in expected_edges:
            row = sandbox_conus_mapping[downstream]
            col = sandbox_conus_mapping[upstream]
            assert dense[row, col] == 1, f"Missing edge {upstream}->{downstream}"
