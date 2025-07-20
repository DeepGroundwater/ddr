"""
@author Tadd Bindas

@date June 23, 2025

Tests for functionality of the subset adjacency module
"""

import polars as pl
import pytest

from ddr.dataset import Gauge
from engine.gages_adjacency import find_origin, preprocess_river_network, subset


def test_simple_subset(
    simple_flowpaths, simple_network, existing_gauge, simple_river_network_dictionary, request
):
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    simple_network = pl.concat(
        [
            simple_network.collect(),
            pl.DataFrame({"id": ["nex-1"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    origin = find_origin(existing_gauge, simple_flowpaths, simple_network)
    assert origin == "wb-2", "Finding the incorrect flowpath for the gauge"
    connections = subset(origin, simple_river_network_dictionary)
    assert connections == [], "Found a headwater gauge connection. Connections are incorrect"


def test_complex_subset(
    complex_flowpaths,
    complex_network,
    existing_gauge,
    complex_river_network_dictionary,
    complex_connections,
):
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    complex_network = pl.concat(
        [
            complex_network.collect(),
            pl.DataFrame({"id": ["nex-12"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    origin = find_origin(existing_gauge, complex_flowpaths, complex_network)
    assert origin == "wb-14", "Finding the incorrect flowpath for the gauge"
    connections = subset(origin, complex_river_network_dictionary)
    assert set(complex_connections) == set(connections), (
        f"Connections for the subsets are not correct. Expected: {complex_connections}, Got: {connections}"
    )


def test_simple_preprocess_river_networks(simple_network, simple_river_network_dictionary, request):
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    simple_network = pl.concat(
        [
            simple_network.collect(),
            pl.DataFrame({"id": ["nex-1"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    wb_river_dictionary = preprocess_river_network(simple_network)

    # NOTE: ordering of the values inside of the dict[str, list[str]] does not matter
    assert wb_river_dictionary.keys() == simple_river_network_dictionary.keys()
    for key in wb_river_dictionary.keys():
        assert set(wb_river_dictionary[key]) == set(simple_river_network_dictionary[key]), (
            f"Mismatch for key {key}: {wb_river_dictionary[key]} != {simple_river_network_dictionary[key]}"
        )


def test_complex_preprocess_river_networks(complex_network, complex_river_network_dictionary):
    """Tests the creation of a one -> many [toid, [id]] dictionary"""
    complex_network = pl.concat(
        [
            complex_network.collect(),
            pl.DataFrame({"id": ["nex-12"], "toid": ["wb-0"], "hl_uri": [None]}),
        ]
    ).lazy()
    wb_river_dictionary = preprocess_river_network(complex_network)

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
def test_find_origin_success(fp, network, gauge, request):
    """Test successful origin finding."""
    fp: pl.DataFrame = request.getfixturevalue(fp)
    network: pl.LazyFrame = request.getfixturevalue(network)
    gauge: Gauge = request.getfixturevalue(gauge)

    origin = find_origin(gauge, fp, network)
    assert origin is not None


@pytest.mark.parametrize(
    "fp, network, gauge",
    [
        ("simple_flowpaths", "simple_network", "non_existing_gage"),
        ("complex_flowpaths", "complex_network", "non_existing_gage"),
    ],
)
def test_find_origin_raises_value_error(fp, network, gauge, request):
    """Test that find_origin raises ValueError for non-existing gauges."""
    fp: pl.DataFrame = request.getfixturevalue(fp)
    network: pl.LazyFrame = request.getfixturevalue(network)
    gauge: Gauge = request.getfixturevalue(gauge)

    with pytest.raises(ValueError):
        find_origin(gauge, fp, network)
