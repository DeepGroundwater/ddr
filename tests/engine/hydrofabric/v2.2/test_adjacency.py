"""
@author Nels Frazier
@author Tadd Bindas

@date June 19, 2025
@version 1.1

Tests for functionality of the adjacency module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import zarr

from engine.hydrofabric.v2_2.adjacency import coo_to_zarr, create_matrix, index_matrix

# TODO consider generating random flowpaths and networks for more robust testing
# then figure out a way to reproduce the same random flowpaths and networks
# when a failure occurs so it can be debugged easily

_table_schema = {"id": pl.String, "toid": pl.String}


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
        coo, ts_order = create_matrix(fp, network, ghost)
        matrix = coo.toarray()

        # Convert original flowpaths for comparison, but only include original flowpaths (not ghost nodes)
        fp_pandas = fp.collect().to_pandas().set_index("id")

        # Ghost nodes will be added to this new frame
        # if they exist in ts_order, and the will have NaN
        # for all columns
        # For non-ghost mode, reindex to match ts_order
        fp_reindexed = fp_pandas.reindex(ts_order)
        result = index_matrix(matrix, fp_reindexed)

        # Check that the result is a DataFrame
        assert isinstance(result, pd.DataFrame)
        # Check that the matrix dimensions match
        assert result.shape == matrix.shape
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

        # Check uniqueness
        assert len(result.index.unique()) == len(result.index)
        assert len(result.columns.unique()) == len(result.columns)


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
        coo, ts_order = create_matrix(fp, network, ghost)
        matrix = coo.toarray()

        # Convert to pandas for easier inspection
        fp_pandas = fp.collect().to_pandas().set_index("id")

        # Validate matrix properties
        # Check that matrix is square and has correct dimensions
        assert matrix.shape[0] == matrix.shape[1], (
            "matrix is not square and does not have the correct dimensions"
        )
        # Check that matrix contains only 0s and 1s
        assert np.all((matrix == 0) | (matrix == 1)), "matrix contains numbers other than 0s and 1s"
        # Check that diagonal is all zeros (no self-loops)
        assert np.all(np.diag(matrix) == 0)
        # Check that matrix is lower triangular
        assert np.allclose(matrix, np.tril(matrix))

        # Check dimensions match expected (note: ghost nodes may be added)
        original_fp_count = len(fp_pandas)
        if ghost:
            # With ghost nodes, matrix should be larger than original flowpaths
            assert matrix.shape[0] >= original_fp_count
        else:
            # Without ghost nodes, should match original flowpaths
            assert matrix.shape[0] == original_fp_count

        # Check that ts_order length matches matrix dimensions
        assert len(ts_order) == matrix.shape[0]

        # Check that original flowpath IDs are in ts_order
        original_ids = set(fp_pandas.index)
        ts_order_set = set(ts_order)
        if ghost:
            # All original IDs should be present (plus ghost nodes)
            assert original_ids.issubset(ts_order_set)
        else:
            # Should exactly match original IDs
            assert original_ids == ts_order_set

    def test_empty_dataframes(self):
        """Test behavior with empty dataframes."""
        empty_fp = pl.LazyFrame({"id": [], "toid": []}, schema=_table_schema)
        empty_network = pl.LazyFrame({"id": [], "toid": []}, schema=_table_schema)

        coo, ts_order = create_matrix(empty_fp, empty_network)
        matrix = coo.toarray()
        assert matrix.shape == (0, 0)
        assert len(ts_order) == 0

    @pytest.mark.parametrize("ghost", [True, False])
    def test_single_flowpath(self, ghost):
        """Test with a single flowpath (terminal)."""
        single_fp = pl.LazyFrame({"id": ["wb-1"], "toid": ["nex-1"]}, schema=_table_schema)
        # Create network DataFrame properly
        single_network = pl.LazyFrame({"id": ["nex-1"], "toid": [None]}, schema=_table_schema)

        coo, ts_order = create_matrix(single_fp, single_network, ghost)
        matrix = coo.toarray()
        if ghost:
            assert matrix.shape == (2, 2)
            assert len(ts_order) == 2
            assert "wb-1" in ts_order
            assert any(id.startswith("ghost-") for id in ts_order)
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
        coo, ts_order = create_matrix(fp, network, ghost)
        matrix = coo.toarray()

        # Convert to pandas and reindex according to ts_order
        fp_pandas = fp.collect().to_pandas().set_index("id").reindex(ts_order)
        network_pandas = network.collect().to_pandas().set_index("id")

        # Test the values in the matrix
        idx = fp_pandas.index
        for i in range(len(fp_pandas)):
            if i >= len(idx):
                break
            nex = fp_pandas.iloc[i]["toid"]
            # Skip if nexus is null
            if pd.isna(nex):
                continue
            # Check if nexus exists in network
            assert nex in network_pandas.index
            ds = network_pandas.loc[nex]["toid"]
            # Skip if downstream is null
            if pd.isna(ds):
                continue
            # Check if downstream segment exists in the index
            assert ds in idx
            j = idx.get_loc(ds)
            assert matrix[j, i] == 1, f"Expected 1 at ({j}, {i}), got {matrix[j, i]}"


@pytest.mark.parametrize(
    "fp, network, ghost",
    [
        ("simple_flowpaths", "simple_network", False),
        ("complex_flowpaths", "complex_network", True),
    ],
)
class TestMatrixToZarr:
    """Test cases for the coo_to_zarr function."""

    def test_coo_to_zarr_basic(self, fp, network, ghost, request):
        """Test basic zarr export functionality."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        coo, ts_order = create_matrix(fp, network, ghost)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_adjacency.zarr"
            coo_to_zarr(coo, ts_order, out_path)

            # Check that zarr file was created
            assert out_path.exists()

            # Check that zarr structure is correct
            store = zarr.storage.LocalStore(root=out_path)
            root = zarr.open_group(store=store)

            assert "indices_0" in root
            assert "indices_1" in root
            assert "values" in root
            assert "order" in root

    def test_zarr_attributes(self, fp, network, ghost, request):
        """Test that zarr attributes are set correctly."""
        fp = request.getfixturevalue(fp)
        network = request.getfixturevalue(network)
        coo, ts_order = create_matrix(fp, network, ghost)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test_adjacency.zarr"
            coo_to_zarr(coo, ts_order, out_path)

            store = zarr.storage.LocalStore(root=out_path)
            root = zarr.open_group(store=store)
            assert root.attrs["format"] == "COO"
            assert "shape" in root.attrs
            assert "data_types" in root.attrs
