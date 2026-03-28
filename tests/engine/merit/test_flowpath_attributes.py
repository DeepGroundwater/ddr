"""Tests for flowpath attribute extraction into zarr stores (MERIT)."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import zarr
from shapely.geometry import LineString

pytest.importorskip("ddr_engine")

from ddr_engine.merit import build_merit_adjacency, write_merit_flowpath_attributes


@pytest.fixture()
def mock_merit_fp_with_attrs() -> gpd.GeoDataFrame:
    """Create a MERIT-compatible GeoDataFrame with lengthkm and slope columns."""
    records = [
        {
            "COMID": 10,
            "NextDownID": 30,
            "up1": 0,
            "up2": 0,
            "up3": 0,
            "up4": 0,
            "lengthkm": 1.0,
            "slope": 0.001,
        },
        {
            "COMID": 20,
            "NextDownID": 30,
            "up1": 0,
            "up2": 0,
            "up3": 0,
            "up4": 0,
            "lengthkm": 2.0,
            "slope": 0.002,
        },
        {
            "COMID": 30,
            "NextDownID": 50,
            "up1": 10,
            "up2": 20,
            "up3": 0,
            "up4": 0,
            "lengthkm": 3.0,
            "slope": 0.003,
        },
        {
            "COMID": 40,
            "NextDownID": 50,
            "up1": 0,
            "up2": 0,
            "up3": 0,
            "up4": 0,
            "lengthkm": 4.0,
            "slope": 0.004,
        },
        {
            "COMID": 50,
            "NextDownID": 0,
            "up1": 30,
            "up2": 40,
            "up3": 0,
            "up4": 0,
            "lengthkm": 5.0,
            "slope": 0.005,
        },
    ]
    df = pd.DataFrame(records)
    df["geometry"] = [LineString([(0, i), (1, i)]) for i in range(len(df))]
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


@pytest.fixture()
def merit_zarr_with_attrs(tmp_path: Path, mock_merit_fp_with_attrs: gpd.GeoDataFrame) -> Path:
    """Build a MERIT zarr store (with adjacency + flowpath attributes)."""
    out_path = tmp_path / "merit_test_adjacency.zarr"
    build_merit_adjacency(mock_merit_fp_with_attrs, out_path)
    return out_path


class TestWriteMeritFlowpathAttributes:
    """Tests for write_merit_flowpath_attributes."""

    def test_arrays_exist(self, merit_zarr_with_attrs: Path) -> None:
        """length_m and slope arrays should exist in the zarr store."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        assert "length_m" in root, "Array 'length_m' not found in zarr store"
        assert "slope" in root, "Array 'slope' not found in zarr store"

    def test_arrays_same_length_as_order(self, merit_zarr_with_attrs: Path) -> None:
        """length_m and slope should have the same length as order."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        order_len = len(root["order"][:])
        assert len(root["length_m"][:]) == order_len
        assert len(root["slope"][:]) == order_len

    def test_float32_dtypes(self, merit_zarr_with_attrs: Path) -> None:
        """Attribute arrays should be float32."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        assert root["length_m"][:].dtype == np.float32
        assert root["slope"][:].dtype == np.float32

    def test_length_converted_to_meters(self, merit_zarr_with_attrs: Path) -> None:
        """lengthkm should be converted to metres (* 1000)."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        order = root["order"][:]
        length_m = root["length_m"][:]

        # Build COMID -> index mapping from the order array
        comid_to_idx = {int(c): i for i, c in enumerate(order)}

        # COMID 10 had lengthkm=1.0, so length_m should be 1000.0
        np.testing.assert_almost_equal(length_m[comid_to_idx[10]], 1000.0, decimal=1)
        # COMID 50 had lengthkm=5.0, so length_m should be 5000.0
        np.testing.assert_almost_equal(length_m[comid_to_idx[50]], 5000.0, decimal=1)

    def test_slope_values(self, merit_zarr_with_attrs: Path) -> None:
        """Slope values should match the input data."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        order = root["order"][:]
        slope = root["slope"][:]

        comid_to_idx = {int(c): i for i, c in enumerate(order)}
        np.testing.assert_almost_equal(slope[comid_to_idx[30]], 0.003, decimal=5)

    def test_nan_for_missing_comids(self, tmp_path: Path) -> None:
        """COMIDs in order but not in the GeoDataFrame should get NaN."""
        # Build a zarr with an extra COMID in order that is not in the GeoDataFrame
        zarr_path = tmp_path / "merit_missing_test.zarr"
        store = zarr.storage.LocalStore(root=zarr_path)
        root = zarr.create_group(store=store)

        order_data = np.array([10, 20, 999], dtype=np.int32)
        order_arr = root.create_array(name="order", shape=order_data.shape, dtype=order_data.dtype)
        order_arr[:] = order_data

        records = [
            {"COMID": 10, "lengthkm": 1.0, "slope": 0.001},
            {"COMID": 20, "lengthkm": 2.0, "slope": 0.002},
        ]
        df = pd.DataFrame(records)
        df["geometry"] = [LineString([(0, i), (1, i)]) for i in range(len(df))]
        fp = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

        write_merit_flowpath_attributes(fp, zarr_path)

        root = zarr.open_group(store=zarr_path, mode="r")
        length_m = root["length_m"][:]
        slope = root["slope"][:]

        # COMID 999 is at index 2 and should be NaN
        assert np.isnan(length_m[2]), "Missing COMID should have NaN length_m"
        assert np.isnan(slope[2]), "Missing COMID should have NaN slope"

    def test_no_extra_lynker_arrays(self, merit_zarr_with_attrs: Path) -> None:
        """MERIT zarr should NOT have top_width, side_slope, or muskingum_x."""
        root = zarr.open_group(store=merit_zarr_with_attrs, mode="r")
        assert "top_width" not in root
        assert "side_slope" not in root
        assert "muskingum_x" not in root
        assert "toid" not in root
