"""
@author Nels Frazier
@author Tadd Bindas

@date June 19, 2025
@version 1.1

Tests for functionality of the adjacency module.
"""

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import polars as pl
import zarr
from scipy import sparse

from ..adjacency import create_matrix, index_matrix, coo_to_zarr  # noqa: TID252

# TODO consider generating random flowpaths and networks for more robust testing
# then figure out a way to reporduce the same random flowpaths and networks
# when a failure occurs so it can be debugged easily


@pytest.fixture
def simple_flowpaths() -> pl.LazyFrame:
    """Create a simple flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-1", "wb-2"],
        "toid": ["nex-1", "nex-1"],
    }
    fp = pl.LazyFrame(data, schema={"id": pl.String, "toid": pl.String})
    return fp


@pytest.fixture
def simple_network() -> pl.LazyFrame:
    """Create a simple network LazyFrame for testing."""
    data = {
        "id": ["nex-1", "wb-1", "wb-2"],
        "toid": [None, "nex-1", "nex-1"],  # Use None for null values
    }
    network = pl.LazyFrame(data, schema={"id": pl.String, "toid": pl.String})
    return network


@pytest.fixture
def complex_flowpaths() -> pl.LazyFrame:
    """Create a more complex flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-10", "wb-11", "wb-12", "wb-13", "wb-14", "wb-15"],
        "toid": ["nex-10", "nex-10", "nex-10", "nex-11", "nex-12", "nex-12"]
    }
    fp = pl.LazyFrame(data, schema={"id": pl.String, "toid": pl.String})
    return fp


@pytest.fixture
def complex_network(complex_flowpaths: pl.LazyFrame) -> pl.LazyFrame:
    """Create a more complex network LazyFrame for testing."""
    flowpath_ids = complex_flowpaths.select(pl.col("id")).collect().to_series().to_list()
    flowpath_toids = complex_flowpaths.select(pl.col("toid")).collect().to_series().to_list()

    data = {
        "id": ["nex-10", "nex-11", "nex-12"] + flowpath_ids,
        "toid": ["wb-13", "wb-14", None] + flowpath_toids,
    }
    network = pl.LazyFrame(data, schema={"id": pl.String, "toid": pl.String})
    return network


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
        
        # Create a DataFrame that matches the matrix dimensions
        if ghost:
            # For ghost mode, we need to create a dataframe that includes ghost nodes
            full_fp_data = []
            for node_id in ts_order:
                if node_id in fp_pandas.index:
                    full_fp_data.append({"id": node_id, "toid": fp_pandas.loc[node_id, "toid"]})
                else:
                    # This is a ghost node
                    full_fp_data.append({"id": node_id, "toid": np.nan})
            
            full_fp = pd.DataFrame(full_fp_data).set_index("id")
            result = index_matrix(matrix, full_fp)
        else:
            # For non-ghost mode, reindex to match ts_order
            fp_reindexed = fp_pandas.reindex(ts_order)
            result = index_matrix(matrix, fp_reindexed)

        # Check that the result is a GeoDataFrame
        assert isinstance(result, gpd.GeoDataFrame)
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
        assert matrix.shape[0] == matrix.shape[1], "matrix is not square and does not have the correct dimensions"
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
        empty_fp = pl.LazyFrame({"id": [], "toid": []}, schema={"id": pl.String, "toid": pl.String})
        empty_network = pl.LazyFrame({"id": [], "toid": []}, schema={"id": pl.String, "toid": pl.String})

        coo, ts_order = create_matrix(empty_fp, empty_network)
        matrix = coo.toarray()
        assert matrix.shape == (0, 0)
        assert len(ts_order) == 0

    @pytest.mark.parametrize("ghost", [True, False])
    def test_single_flowpath(self, ghost):
        """Test with a single flowpath (terminal)."""
        single_fp = pl.LazyFrame({"id": ["wb-1"], "toid": ["nex-1"]}, 
                                schema={"id": pl.String, "toid": pl.String})
        # Create network DataFrame properly
        single_network = pl.LazyFrame({"id": ["nex-1"], "toid": [None]}, 
                                     schema={"id": pl.String, "toid": pl.String})

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
            nex = fp_pandas.loc[idx[i]]["toid"]
            # Skip if nexus is null
            if pd.isna(nex):
                continue
            # Check if nexus exists in network
            if nex not in network_pandas.index:
                continue
            ds = network_pandas.loc[nex]["toid"]
            # Skip if downstream is null
            if pd.isna(ds):
                continue
            # Check if downstream segment exists in the index
            if ds not in idx:
                continue
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