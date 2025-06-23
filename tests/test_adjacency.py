"""
@author Nels Frazier

@date June 12, 2025

Tests for functionality of the adjacency module.
"""

import os
import sys
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).parents[1] / "engine"))

from adjacency import create_matrix, index_matrix, matrix_to_zarr

# TODO consider generating random flowpaths and networks for more robust testing
# then figure out a way to reporduce the same random flowpaths and networks
# when a failure occurs so it can be debugged easily


class TestIndexMatrix:
    """Test cases for the index_matrix function."""

    @pytest.mark.parametrize(
        "fp, network, ghost",
        [
            ("simple_flowpaths", "simple_network", True),
            ("complex_flowpaths", "complex_network", True),
            ("simple_flowpaths", "simple_network", False),
            ("complex_flowpaths", "complex_network", False),
        ],
    )
    def test_index_matrix(self, fp, network, ghost, request):
        """Test basic functionality of index_matrix."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)

        result = index_matrix(matrix, fp)

        # Check that the result is a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
        # Check that the index matches the flowpaths
        assert result.index.equals(fp.index)
        # Check that the columns are named correctly
        assert result.columns.equals(fp.index)
        # Check that the data matches the matrix
        assert np.array_equal(result.values, matrix)
        # Check that the index names are present
        assert result.index.name == "to"
        assert result.columns.name == "from"

        # Check that terminal nodes are found and unique
        if ghost:
            t_rows = result.index[result.index.str.startswith("ghost-")]
            t_cols = result.columns[result.columns.str.startswith("ghost-")]
            assert len(t_rows) > 0
            assert len(t_cols) > 0
        else:
            t_rows = result.index
            t_cols = result.columns
        assert len(t_rows) == len(t_cols)
        assert len(t_rows.unique() == len(t_rows))
        assert len(t_cols.unique() == len(t_cols))


class TestAdjanceyMatrix:
    """Test cases for the create_matrix function."""

    @pytest.mark.parametrize(
        "fp, network, ghost",
        [
            ("simple_flowpaths", "simple_network", True),
            ("complex_flowpaths", "complex_network", True),
            ("simple_flowpaths", "simple_network", False),
            ("complex_flowpaths", "complex_network", False),
        ],
    )
    def test_create_matrix(self, fp, network, ghost, request):
        """Test basic functionality of create_matrix."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)

        # Validate matrix properties
        # Check that matrix is square and has correct dimensions
        assert matrix.shape[0] == matrix.shape[1]
        # Check that matrix contains only 0s and 1s
        assert np.all((matrix == 0) | (matrix == 1))
        # Check that diagonal is all zeros (no self-loops)
        assert np.all(np.diag(matrix) == 0)
        # Check that matrix is lower triangular
        assert np.allclose(matrix, np.tril(matrix))

        paths = set(fp.index)
        # NOTE this works becuase crete_matrix will modify the
        # flowpath and network index when ghost is True
        num_paths = len(paths)
        assert matrix.shape[0] == num_paths
        # Check that ts_order contains all flowpath IDs
        assert len(ts_order) == num_paths
        if not ghost:
            ts_order = filter(lambda s: not s.startswith("ghost-"), ts_order)
        assert set(ts_order) == paths

    def test_empty_dataframes(self):
        """Test behavior with empty dataframes."""
        empty_fp = gpd.GeoDataFrame({"toid": []}, index=[])
        empty_network = gpd.GeoDataFrame({"toid": []}, index=[])
        empty_fp.index.name = "id"
        empty_network.index.name = "id"

        matrix, ts_order = create_matrix(empty_fp, empty_network)

        assert matrix.shape == (0, 0)
        assert len(ts_order) == 0

    @pytest.mark.parametrize("ghost", [True, False])
    def test_single_flowpath(self, ghost):
        """Test with a single flowpath (terminal)."""
        single_fp = gpd.GeoDataFrame({"toid": ["nex-1"]}, index=["wb-1"])
        single_network = gpd.GeoDataFrame({"toid": [np.nan]}, index=["nex-1"], dtype="object")
        single_fp.index.name = "id"
        single_network.index.name = "id"

        matrix, ts_order = create_matrix(single_fp, single_network, ghost)

        if ghost:
            assert matrix.shape == (2, 2)
            assert ts_order == ["wb-1", "ghost-0"]
        else:
            assert matrix.shape == (1, 1)
            assert ts_order == ["wb-1"]

    @pytest.mark.parametrize(
        "fp, network, ghost",
        [
            ("simple_flowpaths", "simple_network", True),
            ("complex_flowpaths", "complex_network", True),
            ("simple_flowpaths", "simple_network", False),
            ("complex_flowpaths", "complex_network", False),
        ],
    )
    def test_topology(self, fp, network, ghost, request):
        """Test matrix topology."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        # NJF an interesting result of the adjancency matrix is that
        # columns that are all zeros are tailwater/terminal segments
        # and the rows that are all zeros are headwater segments
        matrix, ts_order = create_matrix(fp, network, ghost)

        fp = fp.reindex(ts_order)
        # Check matrix properties
        # NOTE this works becuase crete_matrix will modify the
        # flowpath and network index when ghost is True

        # paths = set(fp.index)
        # num_paths = len( paths )

        # These are checked in test_create_matrix
        # I don't think we need to check them again
        # unless a new network/fixture is used for this function
        # which isn't in create_matrix, so for now I'll leave these
        # commented out here...
        # assert matrix.shape[0] == num_paths
        # assert np.allclose(matrix, np.tril(matrix))
        # assert len(ts_order) == num_paths
        # assert set(ts_order) == paths

        # Test the values in the matrix
        idx = fp.index
        for i in range(len(fp)):
            nex = fp.loc[idx[i]]["toid"]
            if isinstance(nex, float) and np.isnan(nex):
                continue
            ds = network.loc[nex]["toid"]
            if isinstance(ds, float) and np.isnan(ds):
                continue
            j = idx.get_loc(ds)
            assert matrix[j, i] == 1, f"Expected 1 at ({i}, {j}), got {matrix[i, j]}"

    def test_missing_nexus_handled(self, capsys):
        """Test that missing nexus IDs are handled gracefully."""
        fp = gpd.GeoDataFrame({"toid": ["nex-missing"]}, index=["wb-1"])
        network = gpd.GeoDataFrame({"toid": []}, index=[])
        fp.index.name = "id"
        network.index.name = "id"

        matrix, ts_order = create_matrix(fp, network)

        # Should capture the print statement about terminal nex
        captured = capsys.readouterr()
        assert "Terminal nex???" in captured.out


@pytest.mark.parametrize(
    "fp, network, ghost",
    [
        ("simple_flowpaths", "simple_network", False),
        ("complex_flowpaths", "complex_network", True),
    ],
)
class TestMatrixToZarr:
    """Test cases for the matrix_to_zarr function."""

    def test_matrix_to_zarr_basic(self, fp, network, ghost, request):
        """Test basic zarr export functionality."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_adjacency.zarr"
            matrix_to_zarr(matrix, ts_order, "test_matrix", out_path)

            # Check that zarr file was created
            assert out_path.exists()

            # Check that zarr structure is correct
            store = zarr.storage.LocalStore(root=out_path)
            root = zarr.open_group(store=store)
            assert "test_matrix" in root

            gauge_root = root["test_matrix"]
            assert "indices_0" in gauge_root
            assert "indices_1" in gauge_root
            assert "values" in gauge_root
            assert "order" in gauge_root

    def test_matrix_to_zarr_default_path(self, fp, network, ghost, request):
        """Test zarr export with default path."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)

        # Change to temp directory to avoid cluttering
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                os.chdir(temp_dir)
                matrix_to_zarr(matrix, ts_order, "test_matrix")

                expected_path = Path(temp_dir) / "test_matrix_adjacency.zarr"
                assert expected_path.exists()
            finally:
                os.chdir(original_cwd)

    def test_zarr_attributes(self, fp, network, ghost, request):
        """Test that zarr attributes are set correctly."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        matrix, ts_order = create_matrix(fp, network, ghost)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_adjacency.zarr"
            matrix_to_zarr(matrix, ts_order, "test_matrix", out_path)

            store = zarr.storage.LocalStore(root=out_path)
            root = zarr.open_group(store=store)
            gauge_root = root["test_matrix"]

            assert gauge_root.attrs["format"] == "COO"
            assert "shape" in gauge_root.attrs
            assert "data_types" in gauge_root.attrs
