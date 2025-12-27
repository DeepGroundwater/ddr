"""A test for all merit flowpath related engine functions"""

import numpy as np
import zarr
from ddr_engine.merit import (
    _build_rustworkx_object,
    _build_upstream_dict_from_merit,
    coo_to_zarr,
    coo_to_zarr_group,
    create_coo,
    create_matrix,
    subset,
)
from scipy import sparse


class TestBuildUpstreamDictFromMerit:
    """Integration tests with sample data for merit upstream dictionary creation"""

    def test_dict_values_are_upstream_comids(self, simple_merit_flowpaths):
        """Test that dictionary values contain upstream COMIDs."""
        upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
        assert 71029036 in upstream_dict[71028858]
        assert 71029426 in upstream_dict[71028858]


class TestBuildRustworkxObject:
    """Integration tests with sample data for merit graph object creation"""

    def test_graph_node_count(self, simple_merit_flowpaths):
        """Test that graph has correct number of nodes."""
        upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
        graph, node_indices = _build_rustworkx_object(upstream_dict)
        assert graph.num_nodes() == 25

    def test_graph_edges_match_upstream_connections(self, simple_merit_flowpaths):
        """Test that graph edges represent upstream connections."""
        upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
        graph, node_indices = _build_rustworkx_object(upstream_dict)
        upstream_idx = node_indices[71029036]
        successors = graph.successors(upstream_idx)
        assert 71028858 in successors

    def test_empty_dict_returns_empty_graph(self):
        """Test that empty upstream dict returns empty graph."""
        graph, node_indices = _build_rustworkx_object({})
        assert graph.num_nodes() == 0
        assert node_indices == {}

    def test_outlet_has_no_successors(self, simple_merit_flowpaths):
        """Test that outlet node has no successors (out_degree=0)."""
        upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
        graph, node_indices = _build_rustworkx_object(upstream_dict)
        outlet_idx = node_indices[71028858]
        assert graph.out_degree(outlet_idx) == 0

    def test_headwaters_have_no_predecessors(self, simple_merit_flowpaths):
        """Test that headwater nodes have no predecessors (in_degree=0)."""
        upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
        graph, node_indices = _build_rustworkx_object(upstream_dict)
        headwater_comid = 71032437
        headwater_idx = node_indices[headwater_comid]
        assert graph.in_degree(headwater_idx) == 0


class TestCreateMatrix:
    def test_create_matrix_returns_coo_and_order(self, simple_merit_flowpaths):
        """Test that create_matrix returns a sparse COO matrix and topological order."""
        matrix, id_order = create_matrix(simple_merit_flowpaths)
        assert isinstance(matrix, sparse.coo_matrix)
        assert isinstance(id_order, list)
        assert len(id_order) == len(simple_merit_flowpaths)

    def test_matrix_is_lower_triangular(self, simple_merit_flowpaths):
        """Test that the resulting matrix is lower triangular."""
        matrix, _ = create_matrix(simple_merit_flowpaths)
        assert np.all(matrix.row >= matrix.col)

    def test_id_order_contains_all_comids(self, simple_merit_flowpaths):
        """Test that topological order contains all COMIDs."""
        _, id_order = create_matrix(simple_merit_flowpaths)
        original_comids = set(simple_merit_flowpaths["COMID"].tolist())
        order_comids = set(id_order)
        assert original_comids == order_comids


class TestCreateMatrixWithCycles:
    def test_create_matrix_removes_cycles(self, merit_flowpaths_with_cycles):
        """Test that create_matrix successfully removes cyclic flowpaths."""
        matrix, id_order = create_matrix(merit_flowpaths_with_cycles)
        assert len(id_order) == 5
        assert matrix.shape == (5, 5)

    def test_valid_flowpaths_retained(self, merit_flowpaths_with_cycles):
        """Test that valid (non-cyclic) flowpaths are retained."""
        _, id_order = create_matrix(merit_flowpaths_with_cycles)
        valid_comids = {78025040, 78025154, 78025845, 78025880, 78025914}
        result_comids = set(id_order)
        assert valid_comids == result_comids


class TestCooToZarr:
    def test_coo_to_zarr_contains_expected_arrays(self, simple_merit_flowpaths, tmp_path):
        """Test that zarr store contains all expected arrays."""
        matrix, id_order = create_matrix(simple_merit_flowpaths)
        out_path = tmp_path / "test_matrix.zarr"
        coo_to_zarr(matrix, id_order, out_path)
        root = zarr.open_group(out_path, mode="r")
        assert "indices_0" in root
        assert "indices_1" in root
        assert "values" in root
        assert "order" in root

    def test_coo_to_zarr_data_integrity(self, simple_merit_flowpaths, tmp_path):
        """Test that data stored in zarr matches original."""
        matrix, id_order = create_matrix(simple_merit_flowpaths)
        out_path = tmp_path / "test_matrix.zarr"
        coo_to_zarr(matrix, id_order, out_path)
        root = zarr.open_group(out_path, mode="r")
        np.testing.assert_array_equal(root["indices_0"][:], matrix.row)
        np.testing.assert_array_equal(root["indices_1"][:], matrix.col)
        np.testing.assert_array_equal(root["values"][:], matrix.data)
        np.testing.assert_array_equal(root["order"][:], np.array(id_order, dtype=np.int32))


class TestSubset:
    def test_subset_from_outlet_returns_all_comids(self, graph_and_indices, simple_merit_flowpaths):
        """Test that subsetting from the outlet returns all COMIDs."""
        graph, node_indices = graph_and_indices
        outlet_comid = 71028858
        subset_comids = subset(outlet_comid, graph, node_indices)
        all_comids = set(simple_merit_flowpaths["COMID"].tolist())
        assert set(subset_comids) == all_comids

    def test_subset_from_mid_network_gauge(self, graph_and_indices):
        """Test subsetting from gauge at COMID 71029768 (mid-network)."""
        graph, node_indices = graph_and_indices
        gauge_comid = 71029768
        subset_comids = subset(gauge_comid, graph, node_indices)
        assert gauge_comid in subset_comids
        assert 71030224 in subset_comids
        assert 71030355 in subset_comids
        assert 71032605 in subset_comids
        assert 71032623 in subset_comids
        assert 71032681 in subset_comids
        assert 71032740 in subset_comids
        assert 71029426 not in subset_comids
        assert 71028858 not in subset_comids

    def test_subset_from_headwater(self, graph_and_indices):
        """Test subsetting from a headwater (order 1) returns only itself."""
        graph, node_indices = graph_and_indices
        headwater_comid = 71032437
        subset_comids = subset(headwater_comid, graph, node_indices)
        assert subset_comids == [headwater_comid]

    def test_subset_comid_not_in_graph(self, graph_and_indices):
        """Test subsetting with a COMID not in the graph."""
        graph, node_indices = graph_and_indices
        unknown_comid = 99999999
        subset_comids = subset(unknown_comid, graph, node_indices)
        assert subset_comids == [unknown_comid]

    def test_subset_includes_origin(self, graph_and_indices):
        """Test that the origin COMID is always included in the subset."""
        graph, node_indices = graph_and_indices
        origin_comid = 71029190
        subset_comids = subset(origin_comid, graph, node_indices)
        assert origin_comid in subset_comids


class TestCreateCoo:
    def test_create_coo_returns_sparse_matrix(self, graph_and_indices, merit_mapping):
        """Test that create_coo returns a sparse COO matrix."""
        graph, node_indices = graph_and_indices
        subset_comids = subset(71028858, graph, node_indices)
        coo, returned_comids = create_coo(subset_comids, merit_mapping, graph, node_indices)
        assert isinstance(coo, sparse.coo_matrix)
        assert isinstance(returned_comids, list)

    def test_create_coo_matrix_is_lower_triangular(self, graph_and_indices, merit_mapping):
        """Test that the COO matrix is lower triangular."""
        graph, node_indices = graph_and_indices
        subset_comids = subset(71028858, graph, node_indices)
        coo, _ = create_coo(subset_comids, merit_mapping, graph, node_indices)
        if coo.nnz > 0:
            assert np.all(coo.row >= coo.col)

    def test_create_coo_headwater_returns_empty_matrix(self, graph_and_indices, merit_mapping):
        """Test that a headwater returns an empty COO matrix."""
        graph, node_indices = graph_and_indices
        headwater_comid = 71032437
        subset_comids = subset(headwater_comid, graph, node_indices)
        coo, _ = create_coo(subset_comids, merit_mapping, graph, node_indices)
        assert coo.nnz == 0


class TestCooToZarrGroup:
    def test_coo_to_zarr_group_creates_arrays(self, graph_and_indices, merit_mapping, tmp_path):
        """Test that coo_to_zarr_group creates expected arrays."""
        graph, node_indices = graph_and_indices
        origin_comid = 71028858
        subset_comids = subset(origin_comid, graph, node_indices)
        coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
        zarr_path = tmp_path / "test_gauge.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000001")
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping)
        assert "indices_0" in gauge_root
        assert "indices_1" in gauge_root
        assert "values" in gauge_root
        assert "order" in gauge_root

    def test_coo_to_zarr_group_data_integrity(self, graph_and_indices, merit_mapping, tmp_path):
        """Test that data stored in zarr matches original."""
        graph, node_indices = graph_and_indices
        origin_comid = 71028858
        subset_comids = subset(origin_comid, graph, node_indices)
        coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
        zarr_path = tmp_path / "test_gauge.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000001")
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping)
        np.testing.assert_array_equal(gauge_root["indices_0"][:], coo.row)
        np.testing.assert_array_equal(gauge_root["indices_1"][:], coo.col)
        np.testing.assert_array_equal(gauge_root["values"][:], coo.data)
        np.testing.assert_array_equal(gauge_root["order"][:], np.array(ts_order, dtype=np.int32))

    def test_coo_to_zarr_group_attrs(self, graph_and_indices, merit_mapping, tmp_path):
        """Test that zarr group has correct attributes."""
        graph, node_indices = graph_and_indices
        origin_comid = 71028858
        subset_comids = subset(origin_comid, graph, node_indices)
        coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
        zarr_path = tmp_path / "test_gauge.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000001")
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping)
        assert gauge_root.attrs["format"] == "COO"
        assert gauge_root.attrs["gage_comid"] == origin_comid
        assert gauge_root.attrs["gage_idx"] == merit_mapping[origin_comid]
        assert "shape" in gauge_root.attrs
        assert "data_types" in gauge_root.attrs


class TestIntegration:
    """Integration tests simulating the full workflow for multiple gauges."""

    def test_full_workflow_outlet_gauge(self, graph_and_indices, merit_mapping, tmp_path):
        """Test full workflow for outlet gauge (STAID='1', COMID=71028858, DRAIN_SQKM=1075)."""
        graph, node_indices = graph_and_indices
        staid = "00000001"
        origin_comid = 71028858
        subset_comids = subset(origin_comid, graph, node_indices)
        assert len(subset_comids) == 25
        coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
        assert coo.nnz == 24
        zarr_path = tmp_path / "gauges.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group(staid)
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping)
        assert gauge_root.attrs["gage_comid"] == 71028858

    def test_full_workflow_mid_network_gauge(self, graph_and_indices, merit_mapping, tmp_path):
        """Test full workflow for mid-network gauge (STAID='2', COMID=71029768, DRAIN_SQKM=300)."""
        graph, node_indices = graph_and_indices
        staid = "00000002"
        origin_comid = 71029768
        subset_comids = subset(origin_comid, graph, node_indices)
        assert origin_comid in subset_comids
        assert len(subset_comids) > 1
        assert len(subset_comids) < 25
        coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
        zarr_path = tmp_path / "gauges.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group(staid)
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping)
        assert gauge_root.attrs["gage_comid"] == 71029768

    def test_full_workflow_cycles_network_gauge(
        self, graph_and_indices_with_cycle, merit_mapping_cycles, tmp_path
    ):
        """Test full workflow for cycles network gauge (STAID='3', COMID=78025040, DRAIN_SQKM=230)."""
        graph, node_indices = graph_and_indices_with_cycle
        staid = "00000003"
        origin_comid = 78025040
        subset_comids = subset(origin_comid, graph, node_indices)
        assert len(subset_comids) == 5
        coo, ts_order = create_coo(subset_comids, merit_mapping_cycles, graph, node_indices)
        assert coo.nnz == 4
        zarr_path = tmp_path / "gauges.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        gauge_root = root.create_group(staid)
        coo_to_zarr_group(coo, ts_order, origin_comid, gauge_root, merit_mapping_cycles)
        assert gauge_root.attrs["gage_comid"] == 78025040

    def test_multiple_gauges_same_zarr_store(self, graph_and_indices, merit_mapping, tmp_path):
        """Test saving multiple gauges to the same zarr store."""
        graph, node_indices = graph_and_indices
        gauges = [
            {"staid": "00000001", "comid": 71028858},
            {"staid": "00000002", "comid": 71029768},
        ]
        zarr_path = tmp_path / "multi_gauges.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)
        for gauge in gauges:
            subset_comids = subset(gauge["comid"], graph, node_indices)
            coo, ts_order = create_coo(subset_comids, merit_mapping, graph, node_indices)
            gauge_root = root.create_group(gauge["staid"])
            coo_to_zarr_group(coo, ts_order, gauge["comid"], gauge_root, merit_mapping)
        assert "00000001" in root
        assert "00000002" in root
        assert root["00000001"].attrs["gage_comid"] == 71028858
        assert root["00000002"].attrs["gage_comid"] == 71029768
