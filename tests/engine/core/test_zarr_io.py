"""Tests for ddr_engine.core.zarr_io — COO zarr read/write."""

from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

pytest.importorskip("ddr_engine")

import zarr
from ddr_engine.core.converters import MeritOrderConverter
from ddr_engine.core.zarr_io import (
    coo_from_zarr,
    coo_from_zarr_generic,
    coo_to_zarr,
    coo_to_zarr_generic,
    coo_to_zarr_group,
    coo_to_zarr_group_generic,
)


@pytest.fixture
def sample_coo() -> sparse.coo_matrix:
    """5x5 COO from sandbox network: 10→30, 20→30, 30→50, 40→50."""
    row = np.array([2, 2, 4, 4], dtype=np.int32)
    col = np.array([0, 1, 2, 3], dtype=np.int32)
    data = np.array([1, 1, 1, 1], dtype=np.uint8)
    return sparse.coo_matrix((data, (row, col)), shape=(5, 5))


@pytest.fixture
def merit_order() -> list[int]:
    return [10, 20, 30, 40, 50]


@pytest.fixture
def lynker_order() -> list[str]:
    return ["wb-10", "wb-20", "wb-30", "wb-40", "wb-50"]


class TestCooToZarr:
    """Tests for coo_to_zarr (high-level, geodataset name)."""

    def test_creates_store(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        assert out.exists()

    def test_has_geodataset_attr(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        root = zarr.open_group(store=out, mode="r")
        assert root.attrs["geodataset"] == "merit"

    def test_has_format_attr(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        root = zarr.open_group(store=out, mode="r")
        assert root.attrs["format"] == "COO"

    def test_has_shape_attr(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        root = zarr.open_group(store=out, mode="r")
        assert root.attrs["shape"] == [5, 5]

    def test_arrays_present(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        root = zarr.open_group(store=out, mode="r")
        for name in ["indices_0", "indices_1", "values", "order"]:
            assert name in root, f"Missing array: {name}"

    def test_merit_order_raw_ints(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        root = zarr.open_group(store=out, mode="r")
        np.testing.assert_array_equal(root["order"][:], np.array([10, 20, 30, 40, 50], dtype=np.int32))


class TestCooFromZarr:
    """Tests for coo_from_zarr (auto-detect geodataset)."""

    def test_roundtrip_merit(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, merit_order, out, "merit")
        loaded_coo, loaded_order = coo_from_zarr(out)
        np.testing.assert_array_equal(loaded_coo.toarray(), sample_coo.toarray())
        assert loaded_order == merit_order

    def test_roundtrip_lynker(self, tmp_path: Path, sample_coo, lynker_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr(sample_coo, lynker_order, out, "lynker")
        loaded_coo, loaded_order = coo_from_zarr(out)
        np.testing.assert_array_equal(loaded_coo.toarray(), sample_coo.toarray())
        assert loaded_order == ["wb-10", "wb-20", "wb-30", "wb-40", "wb-50"]

    def test_no_geodataset_raises(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr_generic(sample_coo, merit_order, out, MeritOrderConverter(), geodataset=None)
        with pytest.raises(ValueError, match="geodataset"):
            coo_from_zarr(out)

    def test_generic_explicit_converter(self, tmp_path: Path, sample_coo, merit_order) -> None:
        out = tmp_path / "test.zarr"
        coo_to_zarr_generic(sample_coo, merit_order, out, MeritOrderConverter(), geodataset=None)
        loaded_coo, loaded_order = coo_from_zarr_generic(out, MeritOrderConverter())
        np.testing.assert_array_equal(loaded_coo.toarray(), sample_coo.toarray())
        assert loaded_order == merit_order


class TestCooToZarrGroup:
    """Tests for coo_to_zarr_group (gauge subgroups)."""

    def test_creates_subgroup(self, tmp_path: Path, sample_coo, merit_order) -> None:
        store = zarr.storage.LocalStore(root=tmp_path / "gages.zarr")
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000050")
        mapping = {comid: i for i, comid in enumerate(merit_order)}
        coo_to_zarr_group(sample_coo, merit_order, 50, gauge_root, mapping, "merit")
        for name in ["indices_0", "indices_1", "values", "order"]:
            assert name in gauge_root, f"Missing array: {name}"

    def test_has_gage_attrs(self, tmp_path: Path, sample_coo, merit_order) -> None:
        store = zarr.storage.LocalStore(root=tmp_path / "gages.zarr")
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000050")
        mapping = {comid: i for i, comid in enumerate(merit_order)}
        coo_to_zarr_group(sample_coo, merit_order, 50, gauge_root, mapping, "merit")
        assert gauge_root.attrs["gage_catchment"] == 50
        assert gauge_root.attrs["gage_idx"] == 4

    def test_generic_no_geodataset(self, tmp_path: Path, sample_coo, merit_order) -> None:
        store = zarr.storage.LocalStore(root=tmp_path / "gages.zarr")
        root = zarr.create_group(store=store)
        gauge_root = root.create_group("00000050")
        mapping = {comid: i for i, comid in enumerate(merit_order)}
        coo_to_zarr_group_generic(
            sample_coo,
            merit_order,
            50,
            gauge_root,
            mapping,
            MeritOrderConverter(),
            geodataset=None,
        )
        assert "geodataset" not in gauge_root.attrs
