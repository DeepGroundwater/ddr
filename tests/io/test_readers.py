"""Tests for ddr.io.readers — convert_ft3_s_to_m3_s, read_gage_info, filter_gages_by_area_threshold, filter_gages_by_da_valid, read_coo, read_zarr."""

import csv
from pathlib import Path

import numpy as np
import pytest
import zarr
import zarr.storage

from ddr.io.readers import (
    convert_ft3_s_to_m3_s,
    filter_gages_by_area_threshold,
    filter_gages_by_da_valid,
    read_gage_info,
    read_zarr,
)


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


class TestReadGageInfoOptionalColumns:
    """Tests for optional column support in read_gage_info()."""

    def test_optional_columns_returned_when_present(self, tmp_path: Path) -> None:
        """CSV with COMID, ABS_DIFF etc → returned in dict."""
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "STAID",
                    "STANAME",
                    "DRAIN_SQKM",
                    "LAT_GAGE",
                    "LNG_GAGE",
                    "COMID",
                    "COMID_DRAIN_SQKM",
                    "ABS_DIFF",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "STAID": "01563500",
                    "STANAME": "Test",
                    "DRAIN_SQKM": "100.0",
                    "LAT_GAGE": "40.0",
                    "LNG_GAGE": "-77.0",
                    "COMID": "12345",
                    "COMID_DRAIN_SQKM": "105.0",
                    "ABS_DIFF": "5.0",
                }
            )
        result = read_gage_info(csv_path)
        assert "ABS_DIFF" in result
        assert result["ABS_DIFF"] == [5.0]
        assert "COMID" in result
        assert result["COMID"] == [12345]

    def test_optional_columns_absent_still_works(self, tmp_path: Path) -> None:
        """CSV without optional columns → dict has only required keys."""
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
                    "STANAME": "Test",
                    "DRAIN_SQKM": "100.0",
                    "LAT_GAGE": "40.0",
                    "LNG_GAGE": "-77.0",
                }
            )
        result = read_gage_info(csv_path)
        assert "ABS_DIFF" not in result
        assert "STAID" in result

    def test_da_valid_and_flow_scale_returned(self, tmp_path: Path) -> None:
        """CSV with DA_VALID and FLOW_SCALE → returned with correct types."""
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "STAID",
                    "STANAME",
                    "DRAIN_SQKM",
                    "LAT_GAGE",
                    "LNG_GAGE",
                    "DA_VALID",
                    "FLOW_SCALE",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "STAID": "01563500",
                    "STANAME": "Test",
                    "DRAIN_SQKM": "100.0",
                    "LAT_GAGE": "40.0",
                    "LNG_GAGE": "-77.0",
                    "DA_VALID": "True",
                    "FLOW_SCALE": "0.85",
                }
            )
            writer.writerow(
                {
                    "STAID": "01563501",
                    "STANAME": "Test2",
                    "DRAIN_SQKM": "200.0",
                    "LAT_GAGE": "41.0",
                    "LNG_GAGE": "-78.0",
                    "DA_VALID": "False",
                    "FLOW_SCALE": "1.0",
                }
            )
        result = read_gage_info(csv_path)
        assert "DA_VALID" in result
        assert "FLOW_SCALE" in result
        assert result["DA_VALID"] == [True, False]
        assert result["FLOW_SCALE"] == [0.85, 1.0]

    def test_staname_fallback_with_optional_columns(self, tmp_path: Path) -> None:
        """CSV missing STANAME but having optional columns → STANAME populated from STAID."""
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["STAID", "DRAIN_SQKM", "LAT_GAGE", "LNG_GAGE", "ABS_DIFF"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "STAID": "01563500",
                    "DRAIN_SQKM": "100.0",
                    "LAT_GAGE": "40.0",
                    "LNG_GAGE": "-77.0",
                    "ABS_DIFF": "5.0",
                }
            )
        result = read_gage_info(csv_path)
        assert "STANAME" in result
        assert len(result["STANAME"]) == 1
        assert "ABS_DIFF" in result


class TestFilterGagesByAreaThreshold:
    """Tests for filter_gages_by_area_threshold()."""

    def test_filters_gages_above_threshold(self) -> None:
        gage_ids = np.array(["00000001", "00000002", "00000003"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002", "00000003"],
            "ABS_DIFF": [10.0, 60.0, 5.0],
        }
        filtered, removed = filter_gages_by_area_threshold(gage_ids, gage_dict, 50.0)
        assert list(filtered) == ["00000001", "00000003"]
        assert removed == 1

    def test_no_filtering_below_threshold(self) -> None:
        gage_ids = np.array(["00000001", "00000002"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002"],
            "ABS_DIFF": [10.0, 20.0],
        }
        filtered, removed = filter_gages_by_area_threshold(gage_ids, gage_dict, 50.0)
        assert len(filtered) == 2
        assert removed == 0

    def test_all_filtered(self) -> None:
        """If all gages filtered, returns empty array."""
        gage_ids = np.array(["00000001"])
        gage_dict: dict[str, list] = {"STAID": ["00000001"], "ABS_DIFF": [100.0]}
        filtered, removed = filter_gages_by_area_threshold(gage_ids, gage_dict, 5.0)
        assert len(filtered) == 0
        assert removed == 1

    def test_missing_abs_diff_raises(self) -> None:
        gage_ids = np.array(["00000001"])
        gage_dict: dict[str, list] = {"STAID": ["00000001"]}
        with pytest.raises(KeyError):
            filter_gages_by_area_threshold(gage_ids, gage_dict, 50.0)

    def test_zero_threshold_exact_match_only(self) -> None:
        gage_ids = np.array(["00000001", "00000002"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002"],
            "ABS_DIFF": [0.0, 0.001],
        }
        filtered, removed = filter_gages_by_area_threshold(gage_ids, gage_dict, 0.0)
        assert list(filtered) == ["00000001"]

    def test_gage_ids_subset_of_dict(self) -> None:
        """gage_ids may be a subset of gage_dict STAIDs."""
        gage_ids = np.array(["00000002"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002", "00000003"],
            "ABS_DIFF": [10.0, 60.0, 5.0],
        }
        filtered, removed = filter_gages_by_area_threshold(gage_ids, gage_dict, 50.0)
        assert len(filtered) == 0
        assert removed == 1


class TestFilterGagesByDaValid:
    """Tests for filter_gages_by_da_valid()."""

    def test_filters_invalid_gages(self) -> None:
        gage_ids = np.array(["00000001", "00000002", "00000003"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002", "00000003"],
            "DA_VALID": [True, False, True],
        }
        filtered, removed = filter_gages_by_da_valid(gage_ids, gage_dict)
        assert list(filtered) == ["00000001", "00000003"]
        assert removed == 1

    def test_all_valid_no_removal(self) -> None:
        gage_ids = np.array(["00000001", "00000002"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002"],
            "DA_VALID": [True, True],
        }
        filtered, removed = filter_gages_by_da_valid(gage_ids, gage_dict)
        assert len(filtered) == 2
        assert removed == 0

    def test_all_invalid(self) -> None:
        gage_ids = np.array(["00000001", "00000002"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001", "00000002"],
            "DA_VALID": [False, False],
        }
        filtered, removed = filter_gages_by_da_valid(gage_ids, gage_dict)
        assert len(filtered) == 0
        assert removed == 2

    def test_missing_da_valid_raises(self) -> None:
        gage_ids = np.array(["00000001"])
        gage_dict: dict[str, list] = {"STAID": ["00000001"]}
        with pytest.raises(KeyError):
            filter_gages_by_da_valid(gage_ids, gage_dict)

    def test_gage_not_in_dict_excluded(self) -> None:
        """A STAID in gage_ids but not in gage_dict defaults to False (excluded)."""
        gage_ids = np.array(["00000001", "99999999"])
        gage_dict: dict[str, list] = {
            "STAID": ["00000001"],
            "DA_VALID": [True],
        }
        filtered, removed = filter_gages_by_da_valid(gage_ids, gage_dict)
        assert list(filtered) == ["00000001"]
        assert removed == 1
