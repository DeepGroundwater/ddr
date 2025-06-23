"""
@author Tadd Bindas

@date June 23, 2025

Tests for functionality of the subset adjacency module
"""

import sys
from pathlib import Path

import polars as pl
import pytest

from ddr import Gauge

sys.path.insert(0, str(Path(__file__).parents[1] / "engine"))

from build_gage_adjacency import (
    find_origin,
)


@pytest.mark.parametrize(
    "fp, network, gauge",
    [
        ("simple_flowpaths_pl", "simple_network_pl", "existing_gauge"),
        ("complex_flowpaths_pl", "complex_network_pl", "existing_gauge"),
    ],
)
def test_find_origin_success(fp, network, gauge, request):
    """Test successful origin finding."""
    fp: pl.LazyFrame = request.getfixturevalue(fp)
    network: pl.LazyFrame = request.getfixturevalue(network)
    gauge: Gauge = request.getfixturevalue(gauge)

    origin = find_origin(gauge, fp, network)
    assert origin is not None


@pytest.mark.parametrize(
    "fp, network, gauge",
    [
        ("simple_flowpaths_pl", "simple_network_pl", "non_existing_gage"),
        ("complex_flowpaths_pl", "complex_network_pl", "non_existing_gage"),
    ],
)
def test_find_origin_raises_value_error(fp, network, gauge, request):
    """Test that find_origin raises ValueError for non-existing gauges."""
    fp: pl.LazyFrame = request.getfixturevalue(fp)
    network: pl.LazyFrame = request.getfixturevalue(network)
    gauge: Gauge = request.getfixturevalue(gauge)

    with pytest.raises(ValueError):
        find_origin(gauge, fp, network)
