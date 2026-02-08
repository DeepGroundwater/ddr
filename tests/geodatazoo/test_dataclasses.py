"""Tests for ddr.geodatazoo.dataclasses."""

import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from ddr.geodatazoo.dataclasses import (
    Dates,
    Gauge,
    GaugeSet,
    MERITGauge,
    RoutingDataclass,
    validate_gages,
)


class TestGauge:
    """Test Gauge pydantic model."""

    def test_gauge_staid_zero_padding(self) -> None:
        g = Gauge(STAID="123", DRAIN_SQKM=100.0)
        assert g.STAID == "00000123"

    def test_gauge_staid_already_padded(self) -> None:
        g = Gauge(STAID="01563500", DRAIN_SQKM=100.0)
        assert g.STAID == "01563500"

    def test_gauge_negative_drain_raises(self) -> None:
        with pytest.raises(ValidationError):
            Gauge(STAID="123", DRAIN_SQKM=-10.0)

    def test_gauge_zero_drain_raises(self) -> None:
        with pytest.raises(ValidationError):
            Gauge(STAID="123", DRAIN_SQKM=0.0)


class TestMERITGauge:
    """Test MERITGauge pydantic model."""

    def test_merit_gauge_requires_comid(self) -> None:
        with pytest.raises(ValidationError):
            MERITGauge(STAID="123", DRAIN_SQKM=100.0)

    def test_merit_gauge_valid(self) -> None:
        g = MERITGauge(STAID="123", DRAIN_SQKM=100.0, COMID=12345)
        assert g.COMID == 12345
        assert g.STAID == "00000123"


class TestValidateGages:
    """Test validate_gages CSV reading."""

    def test_validate_gages_from_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["STAID", "DRAIN_SQKM"])
            writer.writeheader()
            writer.writerow({"STAID": "123", "DRAIN_SQKM": "100.0"})
            writer.writerow({"STAID": "01563500", "DRAIN_SQKM": "200.5"})

        gs = validate_gages(csv_path)
        assert isinstance(gs, GaugeSet)
        assert len(gs.gauges) == 2
        assert gs.gauges[0].STAID == "00000123"
        assert gs.gauges[1].DRAIN_SQKM == 200.5

    def test_validate_gages_merit_type(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "gages.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["STAID", "DRAIN_SQKM", "COMID"])
            writer.writeheader()
            writer.writerow({"STAID": "456", "DRAIN_SQKM": "150.0", "COMID": "99999"})

        gs = validate_gages(csv_path, type=MERITGauge)
        assert isinstance(gs.gauges[0], MERITGauge)
        assert gs.gauges[0].COMID == 99999


class TestDates:
    """Test Dates pydantic model."""

    def test_dates_daily_range_count(self) -> None:
        d = Dates(start_time="2020/01/01", end_time="2020/01/10")
        # inclusive="both" → 10 days
        assert len(d.daily_time_range) == 10

    def test_dates_hourly_range_count(self) -> None:
        d = Dates(start_time="2020/01/01", end_time="2020/01/10")
        # inclusive="left" for hourly → 9 full days * 24
        assert len(d.hourly_time_range) == 9 * 24

    def test_dates_rho_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="Rho needs to be smaller"):
            Dates(start_time="2020/01/01", end_time="2020/01/10", rho=100)

    def test_dates_set_batch_time(self) -> None:
        d = Dates(start_time="2020/01/01", end_time="2020/01/10")
        sub_range = d.daily_time_range[:5]
        d.set_batch_time(sub_range)

        assert len(d.batch_hourly_time_range) == 4 * 24  # 4 full days from 5-element range
        assert len(d.numerical_time_range) == 5

    def test_dates_calculate_time_period(self) -> None:
        torch.manual_seed(42)
        d = Dates(start_time="2020/01/01", end_time="2020/12/31", rho=30)
        d.calculate_time_period()

        assert len(d.batch_daily_time_range) == 30

    def test_dates_create_time_windows(self) -> None:
        d = Dates(start_time="2020/01/01", end_time="2020/04/09", rho=10)
        windows = d.create_time_windows()

        assert windows.ndim == 2
        assert windows.shape[1] == 10

    def test_dates_create_time_windows_no_rho_raises(self) -> None:
        d = Dates(start_time="2020/01/01", end_time="2020/01/10")
        with pytest.raises(ValueError, match="rho must be set"):
            d.create_time_windows()

    def test_dates_numerical_time_range_offset(self) -> None:
        # Origin is 1980/01/01. 2020/01/01 is 14610 days later
        d = Dates(start_time="2020/01/01", end_time="2020/01/02")
        expected_start = (np.datetime64("2020-01-01") - np.datetime64("1980-01-01")) / np.timedelta64(1, "D")

        assert d.numerical_time_range[0] == int(expected_start)


class TestRoutingDataclass:
    """Test RoutingDataclass defaults."""

    def test_routing_dataclass_defaults_none(self) -> None:
        rd = RoutingDataclass()
        assert rd.adjacency_matrix is None
        assert rd.spatial_attributes is None
        assert rd.length is None
        assert rd.slope is None
        assert rd.dates is None
        assert rd.divide_ids is None
