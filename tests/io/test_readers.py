"""Tests for ddr.io.readers — convert_ft3_s_to_m3_s, read_gage_info, read_coo, read_zarr."""

import csv
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage

from ddr.io.readers import convert_ft3_s_to_m3_s, read_gage_info, read_zarr


class TestConvertFt3ToM3:
    """Tests for convert_ft3_s_to_m3_s()."""

    def test_convert_ft3_to_m3(self) -> None:
        result = convert_ft3_s_to_m3_s(np.array(1.0))
        assert np.isclose(result, 0.0283168)


class TestReadGageInfo:
    """Tests for read_gage_info()."""

    def test_read_gage_info_valid_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["STAID", "STANAME", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "STAID": "01563500",
                    "STANAME": "Test Station",
                    "DRAIN_SQKM": "100.0",
                    "LAT_GAGE": "40.0",
                    "LNG_GAGE": "-77.0",
                }
            )

        result = read_gage_info(csv_path)
        assert isinstance(result, dict)
        assert "STAID" in result
        assert len(result["STAID"]) == 1

    def test_read_gage_info_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_gage_info(Path("/nonexistent/path/gages.csv"))


class TestReadZarr:
    """Tests for read_zarr()."""

    def test_read_zarr_valid(self, tmp_path: Path) -> None:
        store_path = tmp_path / "test.zarr"
        store = zarr.storage.LocalStore(root=store_path)
        root = zarr.open_group(store, mode="w")
        arr = root.create_array("data", shape=(3,), dtype="i4")
        arr[:] = np.array([1, 2, 3])

        result = read_zarr(store_path)
        assert isinstance(result, zarr.Group)

    def test_read_zarr_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            read_zarr(Path("/nonexistent/path/store.zarr"))
