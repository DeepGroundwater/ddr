"""Tests for ddr.io.readers — convert_ft3_s_to_m3_s, read_gage_info, filter_gages_by_area_threshold, filter_gages_by_da_valid, read_coo, read_zarr, StreamflowReader."""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.storage

from ddr.io.readers import (
    StreamflowReader,
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


def _make_streamflow_ds(start_date: str, n_days: int = 400, n_divides: int = 3) -> xr.Dataset:
    """Build a minimal xr.Dataset that mimics an icechunk streamflow store."""
    time = pd.date_range(start_date, periods=n_days, freq="D")
    divide_ids = np.arange(1000, 1000 + n_divides)
    rng = np.random.default_rng(42)
    data = rng.uniform(0.01, 5.0, size=(n_divides, n_days)).astype(np.float32)
    return xr.Dataset(
        {"Qr": (["divide_id", "time"], data, {"units": "m^3/s"})},
        coords={"divide_id": divide_ids, "time": time},
    )


@dataclass
class _FakeDates:
    """Minimal stand-in for geodatazoo.dataclasses.Dates used by StreamflowReader."""

    numerical_time_range: np.ndarray = field(default_factory=lambda: np.empty(0))
    batch_daily_time_range: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    batch_hourly_time_range: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))


@dataclass
class _FakeRoutingDC:
    """Minimal stand-in for RoutingDataclass used by StreamflowReader.forward()."""

    divide_ids: np.ndarray = field(default_factory=lambda: np.empty(0))
    dates: _FakeDates = field(default_factory=_FakeDates)


def _make_hourly_streamflow_ds(start_date: str, n_hours: int = 9600, n_divides: int = 3) -> xr.Dataset:
    """Build a minimal xr.Dataset that mimics an hourly icechunk streamflow store."""
    time = pd.date_range(start_date, periods=n_hours, freq="h")
    divide_ids = np.arange(1000, 1000 + n_divides)
    rng = np.random.default_rng(42)
    data = rng.uniform(0.01, 5.0, size=(n_divides, n_hours)).astype(np.float32)
    return xr.Dataset(
        {"Qr": (["divide_id", "time"], data, {"units": "m^3/s"})},
        coords={"divide_id": divide_ids, "time": time},
    )


def _build_reader(ds: xr.Dataset, is_hourly: bool = False) -> StreamflowReader:
    """Construct a StreamflowReader without hitting icechunk by patching read_ic."""
    with patch("ddr.io.readers.read_ic", return_value=ds):
        # Config is only used for read_ic args; the mock ignores it
        reader = StreamflowReader.__new__(StreamflowReader)
        reader.cfg = None  # type: ignore[assignment,unused-ignore]
        reader.ds = ds
        reader.is_hourly = is_hourly
        reader.divide_id_to_index = {did: i for i, did in enumerate(ds.divide_id.values)}
        reader._store_start = pd.Timestamp(ds.time.values[0])
        origin = pd.Timestamp("1980/01/01")
        reader._time_offset = (reader._store_start - origin).days
    return reader


class TestStreamflowReaderTimeOffset:
    """Tests for StreamflowReader._time_offset and adjusted isel indexing."""

    def test_offset_zero_for_origin_aligned_store(self) -> None:
        """A store starting 1980-01-01 should have offset 0."""
        ds = _make_streamflow_ds("1980-01-01")
        reader = _build_reader(ds)
        assert reader._time_offset == 0

    def test_offset_366_for_1981_store(self) -> None:
        """A store starting 1981-01-01 should have offset 366 (1980 is a leap year)."""
        ds = _make_streamflow_ds("1981-01-01")
        reader = _build_reader(ds)
        assert reader._time_offset == 366

    def test_forward_returns_valid_data_with_offset(self) -> None:
        """forward() with a 1981-start store should produce non-NaN output."""
        ds = _make_streamflow_ds("1981-01-01", n_days=400, n_divides=3)
        reader = _build_reader(ds)

        # Build dates that would land correctly: 1981-06-01 (day 517 from origin)
        batch_start = pd.Timestamp("1981-06-01")
        n_batch_days = 10
        batch_daily = pd.date_range(batch_start, periods=n_batch_days, freq="D")
        batch_hourly = pd.date_range(batch_start, periods=(n_batch_days - 1) * 24, freq="h")

        origin = pd.Timestamp("1980/01/01")
        day_offset_start = (batch_start - origin).days
        numerical = np.arange(day_offset_start, day_offset_start + n_batch_days)

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000, 1001, 1002]),
            dates=_FakeDates(
                numerical_time_range=numerical,
                batch_daily_time_range=batch_daily,
                batch_hourly_time_range=batch_hourly,
            ),
        )

        output = reader.forward(routing_dataclass=rc, device="cpu")
        assert output.shape == ((n_batch_days - 1) * 24, 3)
        assert not np.isnan(output.numpy()).any(), "Output contains NaN — time offset is wrong"

    def test_forward_asserts_on_dates_before_store(self) -> None:
        """Requesting dates before the store's start should raise AssertionError."""
        ds = _make_streamflow_ds("1982-01-01", n_days=100)
        reader = _build_reader(ds)

        # Request 1981-06-01 (day 517 from origin), but store starts at day 730
        batch_start = pd.Timestamp("1981-06-01")
        origin = pd.Timestamp("1980/01/01")
        day_offset = (batch_start - origin).days
        numerical = np.arange(day_offset, day_offset + 10)

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000]),
            dates=_FakeDates(
                numerical_time_range=numerical,
                batch_daily_time_range=pd.date_range(batch_start, periods=10, freq="D"),
                batch_hourly_time_range=pd.date_range(batch_start, periods=9 * 24, freq="h"),
            ),
        )

        with pytest.raises(AssertionError, match="negative"):
            reader.forward(routing_dataclass=rc, device="cpu")

    def test_forward_asserts_on_dates_beyond_store(self) -> None:
        """Requesting dates past the store's end should raise AssertionError."""
        ds = _make_streamflow_ds("1981-01-01", n_days=30)
        reader = _build_reader(ds)

        # Request 1981-06-01 — 151 days into a 30-day store
        batch_start = pd.Timestamp("1981-06-01")
        origin = pd.Timestamp("1980/01/01")
        day_offset = (batch_start - origin).days
        numerical = np.arange(day_offset, day_offset + 10)

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000]),
            dates=_FakeDates(
                numerical_time_range=numerical,
                batch_daily_time_range=pd.date_range(batch_start, periods=10, freq="D"),
                batch_hourly_time_range=pd.date_range(batch_start, periods=9 * 24, freq="h"),
            ),
        )

        with pytest.raises(AssertionError, match="exceeds store length"):
            reader.forward(routing_dataclass=rc, device="cpu")


class TestStreamflowReaderHourly:
    """Tests for StreamflowReader with is_hourly=True."""

    def test_hourly_forward_returns_correct_shape(self) -> None:
        """Hourly reader should return (num_hours, num_divides) without interpolation."""
        ds = _make_hourly_streamflow_ds("1981-01-13", n_hours=9600, n_divides=3)
        reader = _build_reader(ds, is_hourly=True)

        batch_start = pd.Timestamp("1981-02-01")
        n_batch_days = 10
        batch_daily = pd.date_range(batch_start, periods=n_batch_days, freq="D")
        batch_hourly = pd.date_range(batch_start, periods=(n_batch_days - 1) * 24, freq="h")

        origin = pd.Timestamp("1980/01/01")
        day_offset_start = (batch_start - origin).days
        numerical = np.arange(day_offset_start, day_offset_start + n_batch_days)

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000, 1001, 1002]),
            dates=_FakeDates(
                numerical_time_range=numerical,
                batch_daily_time_range=batch_daily,
                batch_hourly_time_range=batch_hourly,
            ),
        )

        output = reader.forward(routing_dataclass=rc, device="cpu")
        assert output.shape == ((n_batch_days - 1) * 24, 3)
        assert not np.isnan(output.numpy()).any()

    def test_hourly_forward_asserts_on_dates_before_store(self) -> None:
        """Requesting hourly dates before the store's start should raise AssertionError."""
        ds = _make_hourly_streamflow_ds("1982-01-01", n_hours=2400)
        reader = _build_reader(ds, is_hourly=True)

        batch_start = pd.Timestamp("1981-06-01")
        batch_hourly = pd.date_range(batch_start, periods=9 * 24, freq="h")

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000]),
            dates=_FakeDates(
                numerical_time_range=np.empty(0),
                batch_daily_time_range=pd.date_range(batch_start, periods=10, freq="D"),
                batch_hourly_time_range=batch_hourly,
            ),
        )

        with pytest.raises(AssertionError, match="negative"):
            reader.forward(routing_dataclass=rc, device="cpu")

    def test_hourly_forward_asserts_on_dates_beyond_store(self) -> None:
        """Requesting hourly dates past the store's end should raise AssertionError."""
        ds = _make_hourly_streamflow_ds("1981-01-13", n_hours=720)  # only 30 days
        reader = _build_reader(ds, is_hourly=True)

        batch_start = pd.Timestamp("1981-06-01")
        batch_hourly = pd.date_range(batch_start, periods=9 * 24, freq="h")

        rc = _FakeRoutingDC(
            divide_ids=np.array([1000]),
            dates=_FakeDates(
                numerical_time_range=np.empty(0),
                batch_daily_time_range=pd.date_range(batch_start, periods=10, freq="D"),
                batch_hourly_time_range=batch_hourly,
            ),
        )

        with pytest.raises(AssertionError, match="exceeds store length"):
            reader.forward(routing_dataclass=rc, device="cpu")
