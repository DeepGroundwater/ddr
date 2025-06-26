"""
@author Nels Frazier
@author Tadd Bindas

@date June 25, 2025

Pytest fixtures to test adjacency matrix creation
"""

import polars as pl
import pytest

from ddr import Gauge

_flowpath_table_schema = {
    "id": pl.String,
    "toid": pl.String,
    "tot_drainage_areasqkm": pl.Float64,
}
_network_table_schema = {
    "id": pl.String,
    "toid": pl.String,
    "hl_uri": pl.String,
}


@pytest.fixture
def simple_flowpaths() -> pl.LazyFrame:
    """Create a simple flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-1", "wb-2"],
        "toid": ["nex-1", "nex-1"],
        "tot_drainage_areasqkm": [60, 120],
    }
    fp = pl.LazyFrame(data, schema=_flowpath_table_schema)
    return fp


@pytest.fixture
def simple_network() -> pl.LazyFrame:
    """Create a simple network LazyFrame for testing."""
    data = {
        "id": ["nex-1", "wb-1", "wb-2"],
        "toid": [None, "nex-1", "nex-1"],  # Use None for null values
        "hl_uri": [None, "gages-01234567", "gages-01234567"],
    }
    network = pl.LazyFrame(data, schema=_network_table_schema)
    return network


@pytest.fixture
def complex_flowpaths() -> pl.LazyFrame:
    """Create a more complex flowpaths LazyFrame for testing."""
    data = {
        "id": ["wb-10", "wb-11", "wb-12", "wb-13", "wb-14", "wb-15"],
        "toid": ["nex-10", "nex-10", "nex-10", "nex-11", "nex-12", "nex-12"],
        "tot_drainage_areasqkm": [10, 20, 30, 60, 120, 20],
    }
    fp = pl.LazyFrame(data, schema=_flowpath_table_schema)
    return fp


@pytest.fixture
def complex_network(complex_flowpaths: pl.LazyFrame) -> pl.LazyFrame:
    """Create a more complex network LazyFrame for testing."""
    flowpath_ids = complex_flowpaths.select(pl.col("id")).collect().to_series().to_list()
    flowpath_toids = complex_flowpaths.select(pl.col("toid")).collect().to_series().to_list()

    data = {
        "id": ["nex-10", "nex-11", "nex-12"] + flowpath_ids,
        "toid": ["wb-13", "wb-14", None] + flowpath_toids,
        "hl_uri": [None, None, None, None, None, None, None, "gages-01234567", "gages-01234567"],
    }
    network = pl.LazyFrame(data, schema=_network_table_schema)
    return network


@pytest.fixture
def existing_gauge():
    """Creates a gauge within the testing fixtures"""
    return Gauge(STAID="01234567", DRAIN_SQKM=123.4)


@pytest.fixture
def non_existing_gage():
    """Creates a gauge not in the testing fixtures"""
    return Gauge(STAID="0000", DRAIN_SQKM=123.4)


@pytest.fixture
def simple_river_network_dictionary() -> dict[str, list[str]]:
    """Creates a gauge dictionary based on the simple river network"""
    return {"wb-0": ["wb-1", "wb-2"]}


@pytest.fixture
def complex_river_network_dictionary() -> dict[str, list[str]]:
    """Creates a gauge dictionary based on the complex river network"""
    return {"wb-13": ["wb-10", "wb-11", "wb-12"], "wb-14": ["wb-13"], "wb-0": ["wb-14", "wb-15"]}


@pytest.fixture
def complex_connections() -> list[tuple[str]]:
    """Creates a list of list of strings (Or tuples) where"""
    return [("wb-14", "wb-13"), ("wb-13", "wb-10"), ("wb-13", "wb-11"), ("wb-13", "wb-12")]
