"""Tests for flowpath attribute extraction into zarr stores (Lynker Hydrofabric)."""

import sqlite3
from pathlib import Path

import numpy as np
import pytest
import zarr

pytest.importorskip("ddr_engine")

from ddr_engine.lynker_hydrofabric import write_flowpath_attributes


@pytest.fixture()
def mock_gpkg(tmp_path: Path) -> Path:
    """Create a minimal GeoPackage (sqlite) with flowpath-attributes-ml and flowpaths layers."""
    gpkg_path = tmp_path / "mock_hydrofabric.gpkg"
    conn = sqlite3.connect(gpkg_path)

    # Create flowpath-attributes-ml table
    conn.execute(
        """
        CREATE TABLE "flowpath-attributes-ml" (
            id TEXT PRIMARY KEY,
            Length_m REAL,
            So REAL,
            TopWdth REAL,
            ChSlp REAL,
            MusX REAL
        )
        """
    )
    conn.executemany(
        'INSERT INTO "flowpath-attributes-ml" (id, Length_m, So, TopWdth, ChSlp, MusX) VALUES (?, ?, ?, ?, ?, ?)',
        [
            ("wb-10", 1000.0, 0.001, 15.0, 2.0, 0.2),
            ("wb-20", 2000.0, 0.002, 20.0, 2.5, 0.25),
            ("wb-30", 3000.0, 0.003, 25.0, 3.0, 0.3),
            ("wb-40", 4000.0, 0.004, 30.0, 3.5, 0.35),
            # wb-50 intentionally missing to test NaN handling
        ],
    )

    # Create flowpaths table
    conn.execute(
        """
        CREATE TABLE flowpaths (
            id TEXT PRIMARY KEY,
            toid TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO flowpaths (id, toid) VALUES (?, ?)",
        [
            ("wb-10", "nex-10"),
            ("wb-20", "nex-20"),
            ("wb-30", "nex-30"),
            ("wb-40", "nex-40"),
            ("wb-50", None),
        ],
    )
    conn.commit()
    conn.close()
    return gpkg_path


@pytest.fixture()
def zarr_with_attrs(tmp_path: Path, mock_gpkg: Path) -> Path:
    """Build a minimal zarr store with order array, then write flowpath attributes."""
    zarr_path = tmp_path / "test_adjacency.zarr"
    store = zarr.storage.LocalStore(root=zarr_path)
    root = zarr.create_group(store=store)

    # order corresponds to numeric portion of wb-* IDs
    order_data = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    order_arr = root.create_array(name="order", shape=order_data.shape, dtype=order_data.dtype)
    order_arr[:] = order_data

    write_flowpath_attributes(mock_gpkg, zarr_path)
    return zarr_path


class TestWriteFlowpathAttributes:
    """Tests for write_flowpath_attributes."""

    def test_arrays_exist(self, zarr_with_attrs: Path) -> None:
        """All expected arrays should exist in the zarr store."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        expected_arrays = ["length_m", "slope", "top_width", "side_slope", "muskingum_x", "toid"]
        for name in expected_arrays:
            assert name in root, f"Array '{name}' not found in zarr store"

    def test_arrays_same_length_as_order(self, zarr_with_attrs: Path) -> None:
        """All new arrays should have the same length as order."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        order_len = len(root["order"][:])
        for name in ["length_m", "slope", "top_width", "side_slope", "muskingum_x", "toid"]:
            assert len(root[name][:]) == order_len, f"Array '{name}' length mismatch with order"

    def test_float32_dtypes(self, zarr_with_attrs: Path) -> None:
        """Numeric attribute arrays should be float32."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        for name in ["length_m", "slope", "top_width", "side_slope", "muskingum_x"]:
            arr = root[name][:]
            assert arr.dtype == np.float32, f"Array '{name}' is {arr.dtype}, expected float32"

    def test_toid_is_string(self, zarr_with_attrs: Path) -> None:
        """The toid array should contain string values."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        toid = root["toid"][:]
        # zarr stores strings as object or str dtype
        assert isinstance(toid[0], str), f"toid values should be strings, got {type(toid[0])}"

    def test_values_correct(self, zarr_with_attrs: Path) -> None:
        """Verify that attribute values match the mock GeoPackage data."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        length_m = root["length_m"][:]
        slope = root["slope"][:]
        top_width = root["top_width"][:]

        # order is [10, 20, 30, 40, 50]
        np.testing.assert_almost_equal(length_m[0], 1000.0, decimal=1)
        np.testing.assert_almost_equal(length_m[1], 2000.0, decimal=1)
        np.testing.assert_almost_equal(slope[2], 0.003, decimal=5)
        np.testing.assert_almost_equal(top_width[3], 30.0, decimal=1)

    def test_toid_values(self, zarr_with_attrs: Path) -> None:
        """Verify that toid values match the mock flowpaths data."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        toid = root["toid"][:]
        # order is [10, 20, 30, 40, 50]
        assert toid[0] == "nex-10"
        assert toid[1] == "nex-20"
        assert toid[4] == ""  # wb-50 has toid=None -> ""

    def test_nan_for_missing_segments(self, zarr_with_attrs: Path) -> None:
        """Segments not in the GeoPackage should have NaN values."""
        root = zarr.open_group(store=zarr_with_attrs, mode="r")
        # wb-50 (index 4) was not in flowpath-attributes-ml
        for name in ["length_m", "slope", "top_width", "side_slope", "muskingum_x"]:
            arr = root[name][:]
            assert np.isnan(arr[4]), f"Array '{name}' at index 4 should be NaN for missing wb-50"
