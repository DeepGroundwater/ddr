"""
@author Nels Frazier
@author Tadd Bindas

@date June 23, 2025

Pytest fixtures to test adjacency matrix creation
"""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest

from ddr import Gauge


@pytest.fixture
def simple_flowpaths() -> gpd.GeoDataFrame:
    """Create a simple flowpaths GeoDataFrame for testing."""
    data = {
        "toid": ["nex-1", "nex-1"],  # Terminal flowpath has NaN
        "geometry": [None],  # Simplified for testing
    }
    index = ["wb-1", "wb-2"]
    fp = gpd.GeoDataFrame(data, index=index)
    fp.index.name = "id"
    return fp


@pytest.fixture
def simple_network() -> gpd.GeoDataFrame:
    """Create a simple network GeoDataFrame for testing."""
    data = {
        "toid": [np.nan, "nex-1", "nex-1"],  # Terminal nexus has NaN
        "geometry": [None],  # Simplified for testing
    }
    index = ["nex-1", "wb-1", "wb-2"]
    network = gpd.GeoDataFrame(data, index=index)
    network.index.name = "id"
    return network


@pytest.fixture
def complex_flowpaths() -> gpd.GeoDataFrame:
    """Create a more complex flowpaths GeoDataFrame for testing."""
    data = {"toid": ["nex-10", "nex-10", "nex-10", "nex-11", "nex-12", "nex-12"], "geometry": [None]}
    index = ["wb-10", "wb-11", "wb-12", "wb-13", "wb-14", "wb-15"]
    fp = gpd.GeoDataFrame(data, index=index)
    fp.index.name = "id"
    return fp


@pytest.fixture
def complex_network(complex_flowpaths) -> gpd.GeoDataFrame:
    """Create a more complex network GeoDataFrame for testing."""
    data = {"toid": ["wb-13", "wb-14", np.nan] + complex_flowpaths["toid"].tolist(), "geometry": [None]}
    index = ["nex-10", "nex-11", "nex-12"] + complex_flowpaths.index.tolist()
    network = gpd.GeoDataFrame(data, index=index)
    network.index.name = "id"

    return network


@pytest.fixture
def simple_flowpaths_pl() -> pl.LazyFrame:
    """Create a simple flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-1", "wb-2"],
        "toid": ["nex-1", "nex-1"],
        "tot_drainage_areasqkm": [60, 120],
    }
    fp = pl.LazyFrame(
        data,
        schema={
            "id": pl.String,
            "toid": pl.String,
            "tot_drainage_areasqkm": pl.Float64,
        },
    )
    return fp


@pytest.fixture
def simple_network_pl() -> pl.LazyFrame:
    """Create a simple network LazyFrame for testing."""
    data = {
        "id": ["nex-1", "wb-1", "wb-2"],
        "toid": [None, "nex-1", "nex-1"],  # Use None for null values
        "hl_uri": [None, "gages-01234567", "gages-01234567"],
    }
    network = pl.LazyFrame(
        data,
        schema={
            "id": pl.String,
            "toid": pl.String,
            "hl_uri": pl.String,
        },
    )
    return network


@pytest.fixture
def complex_flowpaths_pl() -> pl.LazyFrame:
    """Create a more complex flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-10", "wb-11", "wb-12", "wb-13", "wb-14", "wb-15"],
        "toid": ["nex-10", "nex-10", "nex-10", "nex-11", "nex-12", "nex-12"],
        "tot_drainage_areasqkm": [10, 20, 30, 60, 120, 20],
    }
    fp = pl.LazyFrame(
        data,
        schema={
            "id": pl.String,
            "toid": pl.String,
            "tot_drainage_areasqkm": pl.Float64,
        },
    )
    return fp


@pytest.fixture
def complex_network_pl(complex_flowpaths_pl: pl.LazyFrame) -> pl.LazyFrame:
    """Create a more complex network LazyFrame for testing."""
    flowpath_ids = complex_flowpaths_pl.select(pl.col("id")).collect().to_series().to_list()
    flowpath_toids = complex_flowpaths_pl.select(pl.col("toid")).collect().to_series().to_list()

    data = {
        "id": ["nex-10", "nex-11", "nex-12"] + flowpath_ids,
        "toid": ["wb-13", "wb-14", None] + flowpath_toids,
        "hl_uri": [None, None, None, None, None, None, None, "gages-01234567", "gages-01234567"],
    }
    network = pl.LazyFrame(
        data,
        schema={
            "id": pl.String,
            "toid": pl.String,
            "hl_uri": pl.String,
        },
    )
    return network


@pytest.fixture
def existing_gauge():
    """Creates a gauge within the testing fixtures"""
    return Gauge(STAID="01234567", DRAIN_SQKM=123.4)


@pytest.fixture
def non_existing_gage():
    """Creates a gauge not in the testing fixtures"""
    return Gauge(STAID="0000", DRAIN_SQKM=123.4)
