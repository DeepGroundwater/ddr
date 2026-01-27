"""End-to-end integration tests for the MERIT adjacency module."""

import numpy as np
import pytest
from ddr_engine.merit import (
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
