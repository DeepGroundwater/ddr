"""Tests for io.py - zarr read/write utilities."""

import numpy as np
from ddr_engine.merit import coo_from_zarr
from scipy import sparse


class TestCooToZarr:
    """Tests for coo_to_zarr function."""

    def test_creates_zarr_store(self, sandbox_zarr_path):
        """Zarr store should be created."""
        assert sandbox_zarr_path.exists()
        assert sandbox_zarr_path.is_dir()

    def test_zarr_has_required_arrays(self, sandbox_zarr_root):
        """Zarr should have indices_0, indices_1, values, order arrays."""
        required = ["indices_0", "indices_1", "values", "order"]
        for name in required:
            assert name in sandbox_zarr_root, f"Missing array: {name}"

    def test_zarr_format_attr(self, sandbox_zarr_root):
        """Zarr should have format='COO' attribute."""
        assert sandbox_zarr_root.attrs["format"] == "COO"

    def test_zarr_shape_attr(self, sandbox_zarr_root):
        """Zarr should have correct shape attribute."""
        assert sandbox_zarr_root.attrs["shape"] == [5, 5]

    def test_zarr_data_types_attr(self, sandbox_zarr_root):
        """Zarr should have data_types attribute."""
        assert "data_types" in sandbox_zarr_root.attrs
        assert "indices_0" in sandbox_zarr_root.attrs["data_types"]
        assert "indices_1" in sandbox_zarr_root.attrs["data_types"]
        assert "values" in sandbox_zarr_root.attrs["data_types"]

    def test_zarr_order_correct(self, sandbox_zarr_order, sandbox_ts_order):
        """Zarr order should match in-memory topological order."""
        assert sandbox_zarr_order == sandbox_ts_order

    def test_zarr_indices_shape(self, sandbox_zarr_root):
        """Indices arrays should have same length."""
        assert len(sandbox_zarr_root["indices_0"]) == len(sandbox_zarr_root["indices_1"])
        assert len(sandbox_zarr_root["indices_0"]) == len(sandbox_zarr_root["values"])

    def test_zarr_values_all_ones(self, sandbox_zarr_root):
        """Values array should be all ones (unweighted adjacency)."""
        values = sandbox_zarr_root["values"][:]
        assert np.all(values == 1)


class TestCooFromZarr:
    """Tests for coo_from_zarr function."""

    def test_returns_tuple(self, sandbox_zarr_path):
        """Should return (coo_matrix, ts_order) tuple."""
        result = coo_from_zarr(sandbox_zarr_path)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_coo_matrix(self, sandbox_zarr_path):
        """First element should be COO matrix."""
        coo, _ = coo_from_zarr(sandbox_zarr_path)
        assert isinstance(coo, sparse.coo_matrix)

    def test_returns_order_list(self, sandbox_zarr_path):
        """Second element should be list of ints."""
        _, ts_order = coo_from_zarr(sandbox_zarr_path)
        assert isinstance(ts_order, list)
        assert all(isinstance(x, (int, np.integer)) for x in ts_order)

    def test_roundtrip_matrix(self, sandbox_coo_matrix, sandbox_zarr_path):
        """Matrix should survive roundtrip through zarr."""
        loaded_coo, _ = coo_from_zarr(sandbox_zarr_path)
        np.testing.assert_array_equal(
            sandbox_coo_matrix.toarray(),
            loaded_coo.toarray(),
        )

    def test_roundtrip_order(self, sandbox_ts_order, sandbox_zarr_path):
        """Topological order should survive roundtrip through zarr."""
        _, loaded_order = coo_from_zarr(sandbox_zarr_path)
        assert sandbox_ts_order == loaded_order


class TestZarrMatrixIntegrity:
    """Tests verifying zarr-loaded matrix matches in-memory matrix."""

    def test_zarr_coo_shape(self, sandbox_zarr_coo):
        """Zarr COO should have correct shape."""
        assert sandbox_zarr_coo.shape == (5, 5)

    def test_zarr_coo_nnz(self, sandbox_zarr_coo):
        """Zarr COO should have correct number of non-zeros."""
        assert sandbox_zarr_coo.nnz == 4

    def test_zarr_coo_is_lower_triangular(self, sandbox_zarr_coo):
        """Zarr COO should be lower triangular."""
        assert np.all(sandbox_zarr_coo.row >= sandbox_zarr_coo.col)

    def test_zarr_matches_memory(self, sandbox_coo_matrix, sandbox_zarr_coo):
        """Zarr COO should match in-memory COO."""
        np.testing.assert_array_equal(
            sandbox_coo_matrix.toarray(),
            sandbox_zarr_coo.toarray(),
        )

    def test_zarr_order_matches_memory(self, sandbox_ts_order, sandbox_zarr_order):
        """Zarr order should match in-memory order."""
        assert sandbox_ts_order == sandbox_zarr_order

    def test_zarr_comid_mapping_matches(self, sandbox_comid_to_idx, sandbox_zarr_comid_to_idx):
        """COMID to index mapping should match."""
        assert sandbox_comid_to_idx == sandbox_zarr_comid_to_idx
