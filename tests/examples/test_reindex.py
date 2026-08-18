"""Tests for the Juniata bundle re-indexing — pure functions, no data stores."""

import numpy as np

from examples.juniata.extract_bundle import reindex_subgraph


def _toy_conus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 6-reach CONUS: order maps position -> COMID
    conus_order = np.array([900, 901, 902, 903, 904, 905], dtype=np.int32)
    length = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    slope = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
    return conus_order, length, slope


class TestReindexSubgraph:
    def test_remaps_edges_and_gage_idx(self) -> None:
        conus_order, length, slope = _toy_conus()
        # Subgraph = positions {1, 3, 4} (COMIDs 901, 903, 904); edges 1->3, 3->4
        out = reindex_subgraph(
            gage_indices_0=np.array([3, 4], dtype=np.int32),  # downstream (row)
            gage_indices_1=np.array([1, 3], dtype=np.int32),  # upstream (col)
            gage_order=np.array([901, 903, 904], dtype=np.int32),
            gage_idx=4,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        # New space sorted by CONUS position: 901->0, 903->1, 904->2
        np.testing.assert_array_equal(out["order"], [901, 903, 904])
        np.testing.assert_array_equal(out["indices_0"], [1, 2])
        np.testing.assert_array_equal(out["indices_1"], [0, 1])
        assert out["gage_idx"] == 2
        assert out["shape"] == [3, 3]

    def test_physical_arrays_align_to_new_order(self) -> None:
        conus_order, length, slope = _toy_conus()
        out = reindex_subgraph(
            gage_indices_0=np.array([3], dtype=np.int32),
            gage_indices_1=np.array([1], dtype=np.int32),
            gage_order=np.array([901, 903], dtype=np.int32),
            gage_idx=3,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        np.testing.assert_allclose(out["length_m"], [11.0, 13.0])
        np.testing.assert_allclose(out["slope"], [2e-3, 4e-3])

    def test_lower_triangular_preserved(self) -> None:
        # Downstream position must stay > upstream position in the new space —
        # the CSR solve is forward substitution and requires it.
        conus_order, length, slope = _toy_conus()
        out = reindex_subgraph(
            gage_indices_0=np.array([3, 4, 4], dtype=np.int32),
            gage_indices_1=np.array([1, 3, 1], dtype=np.int32),
            gage_order=np.array([901, 903, 904], dtype=np.int32),
            gage_idx=4,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        assert (out["indices_0"] > out["indices_1"]).all()
