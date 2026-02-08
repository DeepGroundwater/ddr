"""End-to-end integration tests for the MERIT adjacency module."""

import numpy as np
import pytest
import zarr
from ddr_engine.merit import (
    build_gauge_adjacencies,
    build_graph,
    build_merit_adjacency,
    build_upstream_dict,
    coo_from_zarr,
    create_adjacency_matrix,
    subset_upstream,
)


class TestFullPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_pipeline_from_geodataframe_to_zarr(self, tmp_path, mock_merit_fp):
        """Test complete pipeline: GeoDataFrame → matrix → zarr → load."""
        out_path = tmp_path / "pipeline_test.zarr"

        # Build and save
        result_path = build_merit_adjacency(mock_merit_fp, out_path)
        assert result_path.exists()

        # Load and verify
        coo, ts_order = coo_from_zarr(result_path)

        assert coo.shape == (5, 5)
        assert coo.nnz == 4
        assert set(ts_order) == {10, 20, 30, 40, 50}

    def test_pipeline_preserves_connectivity(self, tmp_path, mock_merit_fp):
        """Verify connectivity is preserved through pipeline."""
        out_path = tmp_path / "connectivity_test.zarr"

        # Create in-memory
        matrix_mem, order_mem = create_adjacency_matrix(mock_merit_fp)

        # Save and reload
        build_merit_adjacency(mock_merit_fp, out_path)
        matrix_disk, order_disk = coo_from_zarr(out_path)

        # Should be identical
        np.testing.assert_array_equal(matrix_mem.toarray(), matrix_disk.toarray())
        assert order_mem == order_disk

    def test_subset(self, mock_merit_fp):
        """Test subsetting followed by routing."""
        upstream_dict = build_upstream_dict(mock_merit_fp)
        graph, node_indices = build_graph(upstream_dict)
        # matrix, ts_order = create_adjacency_matrix(mock_merit_fp)

        # idx = {comid: i for i, comid in enumerate(ts_order)}

        # Subset to gauge at node 30
        subset_30 = subset_upstream(30, graph, node_indices)
        assert set(subset_30) == {10, 20, 30}


class TestGaugeIntegration:
    """Tests for gauge-related integration."""

    def test_gauge_subset_sizes(self, G, sandbox_node_indices, sandbox_gauge_set):
        """Verify subset sizes for each gauge."""
        expected_sizes = {
            30: 3,  # 10, 20, 30
            50: 5,  # 10, 20, 30, 40, 50
        }

        for gauge in sandbox_gauge_set.gauges:
            subset = subset_upstream(gauge.COMID, G, sandbox_node_indices)
            assert len(subset) == expected_sizes[gauge.COMID], (
                f"Gauge {gauge.STAID} at COMID {gauge.COMID} has wrong subset size"
            )

    def test_gauge_subsets_are_nested(self, G, sandbox_node_indices):
        """Gauge 30's subset should be contained in gauge 50's subset."""
        subset_30 = set(subset_upstream(30, G, sandbox_node_indices))
        subset_50 = set(subset_upstream(50, G, sandbox_node_indices))

        assert subset_30.issubset(subset_50)

    def test_headwater_gauge_subset(self, G, sandbox_node_indices):
        """A gauge at a headwater should have subset of size 1."""
        for comid in [10, 20, 40]:
            subset = subset_upstream(comid, G, sandbox_node_indices)
            assert len(subset) == 1
            assert subset[0] == comid


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_geodataframe(self, tmp_path):
        """Empty GeoDataFrame should raise error."""
        import geopandas as gpd

        empty_gdf = gpd.GeoDataFrame(
            {"COMID": [], "NextDownID": [], "up1": [], "up2": [], "up3": [], "up4": []},
            geometry=[],
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError, match="No upstream connections"):
            create_adjacency_matrix(empty_gdf)

    def test_single_node_network(self, tmp_path):
        """Single node network (no connections) should raise error."""
        import geopandas as gpd
        from shapely.geometry import LineString

        single_node = gpd.GeoDataFrame(
            {
                "COMID": [1],
                "NextDownID": [0],
                "up1": [0],
                "up2": [0],
                "up3": [0],
                "up4": [0],
            },
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:4326",
        )

        with pytest.raises(ValueError, match="No upstream connections"):
            create_adjacency_matrix(single_node)

    def test_zarr_path_parent_created(self, tmp_path, mock_merit_fp):
        """Parent directories should be created if they don't exist."""
        out_path = tmp_path / "nested" / "dirs" / "test.zarr"

        assert not out_path.parent.exists()

        build_merit_adjacency(mock_merit_fp, out_path)

        assert out_path.exists()


class TestIsolatedCOMIDs:
    """Tests for isolated COMIDs (single-reach basins with no connections)."""

    @pytest.fixture
    def fp_with_isolated(self, mock_merit_fp):
        """Sandbox network plus an isolated COMID 99 with no connections."""
        import geopandas as gpd
        from shapely.geometry import LineString

        isolated_row = gpd.GeoDataFrame(
            {
                "COMID": [99],
                "NextDownID": [0],
                "up1": [0],
                "up2": [0],
                "up3": [0],
                "up4": [0],
            },
            geometry=[LineString([(0, 99), (1, 99)])],
            crs="EPSG:4326",
        )
        return gpd.GeoDataFrame(
            data=mock_merit_fp._append(isolated_row, ignore_index=True),
            crs="EPSG:4326",
        )

    def test_isolated_comid_in_order(self, fp_with_isolated):
        """Isolated COMID should appear in topological order."""
        matrix, order = create_adjacency_matrix(fp_with_isolated)
        assert 99 in order

    def test_isolated_comid_matrix_shape(self, fp_with_isolated):
        """Matrix should be 6x6 (5 connected + 1 isolated)."""
        matrix, order = create_adjacency_matrix(fp_with_isolated)
        assert matrix.shape == (6, 6)
        assert len(order) == 6

    def test_isolated_comid_no_edges(self, fp_with_isolated):
        """Isolated COMID should have no edges — same 4 edges as sandbox."""
        matrix, order = create_adjacency_matrix(fp_with_isolated)
        assert matrix.nnz == 4

    def test_isolated_comid_lower_triangular(self, fp_with_isolated):
        """Matrix should still be lower triangular."""
        matrix, _ = create_adjacency_matrix(fp_with_isolated)
        assert np.all(matrix.row >= matrix.col)

    def test_connected_comids_unchanged(self, fp_with_isolated):
        """Connected COMIDs should still have correct edges."""
        matrix, order = create_adjacency_matrix(fp_with_isolated)
        idx = {comid: i for i, comid in enumerate(order)}
        dense = matrix.toarray()

        expected_edges = [
            (30, 10),
            (30, 20),
            (50, 30),
            (50, 40),
        ]
        for downstream, upstream in expected_edges:
            assert dense[idx[downstream], idx[upstream]] == 1

    def test_isolated_comid_zarr_roundtrip(self, tmp_path, fp_with_isolated):
        """Isolated COMID should survive zarr roundtrip."""
        out_path = tmp_path / "isolated_test.zarr"
        build_merit_adjacency(fp_with_isolated, out_path)
        coo, ts_order = coo_from_zarr(out_path)

        assert 99 in ts_order
        assert coo.shape == (6, 6)
        assert coo.nnz == 4


class TestHeadwaterGaugeAdjacency:
    """Tests for build_gauge_adjacencies with headwater gages."""

    @pytest.fixture
    def fp_with_isolated(self, mock_merit_fp):
        """Sandbox network plus isolated COMID 99."""
        import geopandas as gpd
        from shapely.geometry import LineString

        isolated_row = gpd.GeoDataFrame(
            {
                "COMID": [99],
                "NextDownID": [0],
                "up1": [0],
                "up2": [0],
                "up3": [0],
                "up4": [0],
            },
            geometry=[LineString([(0, 99), (1, 99)])],
            crs="EPSG:4326",
        )
        return gpd.GeoDataFrame(
            data=mock_merit_fp._append(isolated_row, ignore_index=True),
            crs="EPSG:4326",
        )

    @pytest.fixture
    def merit_zarr_path(self, tmp_path, fp_with_isolated):
        """Build MERIT adjacency zarr with isolated COMID included."""
        out_path = tmp_path / "merit_adjacency.zarr"
        build_merit_adjacency(fp_with_isolated, out_path)
        return out_path

    def test_headwater_gauge_gets_zarr_group(self, tmp_path, fp_with_isolated, merit_zarr_path):
        """A gauge at a connected headwater (COMID 10) should produce a zarr subgroup."""
        from tests.engine.merit.conftest import MockGauge, MockGaugeSet

        gauge_set = MockGaugeSet(gauges=[MockGauge(STAID="00000010", COMID=10)])
        out_path = tmp_path / "gauge_adj.zarr"

        build_gauge_adjacencies(fp_with_isolated, merit_zarr_path, gauge_set, out_path)

        root = zarr.open_group(store=out_path, mode="r")
        assert "00000010" in list(root.group_keys())

    def test_headwater_gauge_has_empty_coo(self, tmp_path, fp_with_isolated, merit_zarr_path):
        """Headwater gauge zarr subgroup should have empty indices (no edges)."""
        from tests.engine.merit.conftest import MockGauge, MockGaugeSet

        gauge_set = MockGaugeSet(gauges=[MockGauge(STAID="00000010", COMID=10)])
        out_path = tmp_path / "gauge_adj_empty.zarr"

        build_gauge_adjacencies(fp_with_isolated, merit_zarr_path, gauge_set, out_path)

        root = zarr.open_group(store=out_path, mode="r")
        gauge_group = root["00000010"]
        assert gauge_group["indices_0"][:].shape[0] == 0
        assert gauge_group["indices_1"][:].shape[0] == 0
        assert len(gauge_group["order"][:]) == 1

    def test_isolated_gauge_gets_zarr_group(self, tmp_path, fp_with_isolated, merit_zarr_path):
        """A gauge at an isolated COMID (99) should produce a zarr subgroup."""
        from tests.engine.merit.conftest import MockGauge, MockGaugeSet

        gauge_set = MockGaugeSet(gauges=[MockGauge(STAID="00000099", COMID=99)])
        out_path = tmp_path / "gauge_adj_isolated.zarr"

        build_gauge_adjacencies(fp_with_isolated, merit_zarr_path, gauge_set, out_path)

        root = zarr.open_group(store=out_path, mode="r")
        assert "00000099" in list(root.group_keys())

        gauge_group = root["00000099"]
        assert gauge_group["indices_0"][:].shape[0] == 0
        assert gauge_group["indices_1"][:].shape[0] == 0
        assert len(gauge_group["order"][:]) == 1
        assert gauge_group.attrs["gage_catchment"] == 99

    def test_mixed_gauges_all_present(self, tmp_path, fp_with_isolated, merit_zarr_path):
        """A batch with connected gauge (50), headwater (10), and isolated (99) should all produce zarr groups."""
        from tests.engine.merit.conftest import MockGauge, MockGaugeSet

        gauge_set = MockGaugeSet(
            gauges=[
                MockGauge(STAID="00000050", COMID=50),
                MockGauge(STAID="00000010", COMID=10),
                MockGauge(STAID="00000099", COMID=99),
            ]
        )
        out_path = tmp_path / "gauge_adj_mixed.zarr"

        build_gauge_adjacencies(fp_with_isolated, merit_zarr_path, gauge_set, out_path)

        root = zarr.open_group(store=out_path, mode="r")
        group_keys = list(root.group_keys())
        assert "00000050" in group_keys
        assert "00000010" in group_keys
        assert "00000099" in group_keys

        # Connected gauge should have edges
        assert root["00000050"]["indices_0"][:].shape[0] > 0
        # Headwater/isolated should have no edges
        assert root["00000010"]["indices_0"][:].shape[0] == 0
        assert root["00000099"]["indices_0"][:].shape[0] == 0
