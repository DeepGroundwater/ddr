"""
@author Tadd Bindas

@date June 23, 2025

Tests for functionality of the subset adjacency module
"""

import polars as pl
import pytest
from ddr_engine.hydrofabric import find_origin, preprocess_river_network, subset

from ddr.geodatazoo.dataclasses import Gauge


def test_simple_subset(
    simple_flowpaths: pl.LazyFrame,
    simple_network: pl.LazyFrame,
    existing_gauge: Gauge,
    simple_river_network_dictionary: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    simple_network_extended = pl.concat(
        [
            simple_network.collect(),
            pl.DataFrame({"id": ["nex-1"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    origin = find_origin(existing_gauge, simple_flowpaths, simple_network_extended)
    assert origin == "wb-2", "Finding the incorrect flowpath for the gauge"
    connections = subset(origin, simple_river_network_dictionary)
    assert connections == [], "Found a headwater gauge connection. Connections are incorrect"


def test_complex_subset(
    complex_flowpaths: pl.LazyFrame,
    complex_network: pl.LazyFrame,
    existing_gauge: Gauge,
    complex_river_network_dictionary: dict[str, list[str]],
    complex_connections: list[str],
) -> None:
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    complex_network_extended = pl.concat(
        [
            complex_network.collect(),
            pl.DataFrame({"id": ["nex-12"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    origin = find_origin(existing_gauge, complex_flowpaths, complex_network_extended)
    assert origin == "wb-14", "Finding the incorrect flowpath for the gauge"
    connections = subset(origin, complex_river_network_dictionary)
    assert set(complex_connections) == set(connections), (
        f"Connections for the subsets are not correct. Expected: {complex_connections}, Got: {connections}"
    )


def test_simple_preprocess_river_networks(
    simple_network: pl.LazyFrame,
    simple_river_network_dictionary: dict[str, list[str]],
    request: pytest.FixtureRequest,
) -> None:
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    simple_network_extended = pl.concat(
        [
            simple_network.collect(),
            pl.DataFrame({"id": ["nex-1"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    wb_river_dictionary = preprocess_river_network(simple_network_extended)

    # NOTE: ordering of the values inside of the dict[str, list[str]] does not matter
    assert wb_river_dictionary.keys() == simple_river_network_dictionary.keys()
    for key in wb_river_dictionary.keys():
        assert set(wb_river_dictionary[key]) == set(simple_river_network_dictionary[key]), (
            f"Mismatch for key {key}: {wb_river_dictionary[key]} != {simple_river_network_dictionary[key]}"
        )


def test_complex_preprocess_river_networks(
    complex_network: pl.LazyFrame,
    complex_river_network_dictionary: dict[str, list[str]],
) -> None:
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    complex_network_extended = pl.concat(
        [
            complex_network.collect(),
            pl.DataFrame({"id": ["nex-12"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    wb_river_dictionary = preprocess_river_network(complex_network_extended)

    # NOTE: ordering of the values inside of the dict[str, list[str]] does not matter
    assert wb_river_dictionary.keys() == complex_river_network_dictionary.keys()
    for key in wb_river_dictionary.keys():
        assert set(wb_river_dictionary[key]) == set(complex_river_network_dictionary[key]), (
            f"Mismatch for key {key}: {wb_river_dictionary[key]} != {complex_river_network_dictionary[key]}"
        )


@pytest.mark.parametrize(
    "fp, network, gauge",
    [
        ("simple_flowpaths", "simple_network", "existing_gauge"),
        ("complex_flowpaths", "complex_network", "existing_gauge"),
    ],
)
def test_find_origin_success(fp: str, network: str, gauge: str, request: pytest.FixtureRequest) -> None:
    """Test successful origin finding."""
    fp_fixture: pl.LazyFrame = request.getfixturevalue(fp)
    network_fixture: pl.LazyFrame = request.getfixturevalue(network)
    gauge_fixture: Gauge = request.getfixturevalue(gauge)

    origin = find_origin(gauge_fixture, fp_fixture, network_fixture)
    assert origin is not None


@pytest.mark.parametrize(
    "fp, network, gauge",
    [
        ("simple_flowpaths", "simple_network", "non_existing_gage"),
        ("complex_flowpaths", "complex_network", "non_existing_gage"),
    ],
)
def test_find_origin_raises_value_error(
    fp: str, network: str, gauge: str, request: pytest.FixtureRequest
) -> None:
    """Test that find_origin raises ValueError for non-existing gauges."""
    fp_fixture: pl.LazyFrame = request.getfixturevalue(fp)
    network_fixture: pl.LazyFrame = request.getfixturevalue(network)
    gauge_fixture: Gauge = request.getfixturevalue(gauge)

    with pytest.raises(ValueError):
        find_origin(gauge_fixture, fp_fixture, network_fixture)
