"""Tests for Lynker Hydrofabric adjacency matrix build functions."""

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("ddr_engine")

from ddr_engine.lynker_hydrofabric import create_coo, create_matrix


class TestBuildLynkerHydrofabricAdjacency:
    """Tests for create_matrix (the core of build_lynker_hydrofabric_adjacency)."""

    def test_creates_coo_matrix(self, sandbox_network) -> None:
        """create_matrix returns a COO matrix from sandbox flowpath/network frames."""
        import polars as pl

        # Build minimal flowpaths and network LazyFrames from sandbox data
        fp_data = {
            "id": ["wb-10", "wb-20", "wb-30", "wb-40", "wb-50"],
            "toid": ["nex-10", "nex-20", "nex-30", "nex-40", None],
            "tot_drainage_areasqkm": [10.0, 10.0, 30.0, 10.0, 50.0],
        }
        fp = pl.DataFrame(fp_data).lazy()

        matrix, ts_order = create_matrix(fp, sandbox_network)
        assert isinstance(matrix, sparse.coo_matrix)
        assert len(ts_order) == 5

    def test_matrix_lower_triangular(self, sandbox_network) -> None:
        """Output matrix should be lower triangular (row >= col)."""
        import polars as pl

        fp_data = {
            "id": ["wb-10", "wb-20", "wb-30", "wb-40", "wb-50"],
            "toid": ["nex-10", "nex-20", "nex-30", "nex-40", None],
            "tot_drainage_areasqkm": [10.0, 10.0, 30.0, 10.0, 50.0],
        }
        fp = pl.DataFrame(fp_data).lazy()

        matrix, _ = create_matrix(fp, sandbox_network)
        assert np.all(matrix.row >= matrix.col)


class TestCreateCoo:
    """Tests for create_coo (per-gauge COO subset creation)."""

    def test_coo_matrix_from_connections(self, sandbox_connections, sandbox_conus_mapping) -> None:
        """Connections for wb-50 → valid COO matrix."""
        coo, subset_flowpaths = create_coo(sandbox_connections, sandbox_conus_mapping)
        assert isinstance(coo, sparse.coo_matrix)
        assert len(subset_flowpaths) > 0

    def test_coo_matrix_has_edges(self, sandbox_connections, sandbox_conus_mapping) -> None:
        """wb-50 has upstream connections → non-empty COO."""
        coo, _ = create_coo(sandbox_connections, sandbox_conus_mapping)
        assert coo.nnz > 0
