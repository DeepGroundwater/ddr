"""
@author Tadd Bindas
@author Nels Frazier

Pytest fixtures to test engine package functionality
"""

import geopandas as gpd
import numpy as np
import polars as pl
import pytest
from ddr_engine.merit import _build_rustworkx_object, _build_upstream_dict_from_merit, create_matrix
from shapely.geometry import LineString

from ddr.dataset import Gauge

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
def simple_merit_flowpaths() -> gpd.GeoDataFrame:
    data = {
        "COMID": [
            71028858,
            71029036,
            71029190,
            71029284,
            71029426,
            71029768,
            71029904,
            71030162,
            71030203,
            71030224,
            71030297,
            71030355,
            71032437,
            71032605,
            71032623,
            71032681,
            71032740,
            71032794,
            71032941,
            71032955,
            71033062,
            71033111,
            71033228,
            71033238,
            71033328,
        ],
        "lengthkm": [
            4.766063,
            15.292738,
            11.098549,
            4.872664,
            30.568782,
            11.018444,
            17.074646,
            7.704272,
            0.974492,
            10.020941,
            9.278696,
            1.936047,
            12.675343,
            2.701338,
            0.441787,
            6.533671,
            2.578889,
            8.791780,
            5.077842,
            2.844401,
            4.700433,
            0.585435,
            0.590725,
            7.019752,
            6.102915,
        ],
        "lengthdir": [
            2.442213,
            9.363669,
            6.631747,
            3.982376,
            13.583714,
            6.987515,
            12.976398,
            6.165008,
            0.864490,
            5.981756,
            5.622161,
            1.430969,
            9.633355,
            1.840115,
            0.301573,
            4.167637,
            2.054736,
            5.702285,
            3.868863,
            1.841729,
            2.028674,
            0.544467,
            0.504944,
            4.344070,
            4.671418,
        ],
        "sinuosity": [
            1.951535,
            1.633199,
            1.673548,
            1.223557,
            2.250399,
            1.576876,
            1.315823,
            1.249678,
            1.127245,
            1.675251,
            1.650379,
            1.352962,
            1.315777,
            1.468027,
            1.464941,
            1.567716,
            1.255095,
            1.541800,
            1.312490,
            1.544419,
            2.316997,
            1.075243,
            1.169882,
            1.615939,
            1.306437,
        ],
        "slope": [
            0.000105,
            0.000157,
            0.000414,
            0.000656,
            0.000872,
            0.000833,
            0.000316,
            0.001154,
            0.000000,
            0.001185,
            0.002012,
            0.001289,
            0.003890,
            0.001699,
            0.005649,
            0.001940,
            0.002749,
            0.002248,
            0.002143,
            0.003369,
            0.003187,
            0.001876,
            0.001352,
            0.002887,
            0.004025,
        ],
        "uparea": [
            1098.895469,
            631.444952,
            516.209408,
            428.939563,
            437.215553,
            255.165205,
            266.976384,
            149.335824,
            85.293630,
            173.168512,
            128.938256,
            76.675030,
            53.102681,
            34.473798,
            41.235038,
            45.392689,
            44.297805,
            56.497637,
            51.054904,
            27.734232,
            34.515449,
            25.185231,
            35.414099,
            43.095592,
            41.421901,
        ],
        "order": [3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "strmDrop_t": [
            0.5,
            2.4,
            4.6,
            3.2,
            26.7,
            9.2,
            5.4,
            8.9,
            0.0,
            11.9,
            18.7,
            2.5,
            49.4,
            4.6,
            2.5,
            12.7,
            7.1,
            19.8,
            10.9,
            9.6,
            15.0,
            1.1,
            0.8,
            20.3,
            24.6,
        ],
        "slope_taud": [
            0.000105,
            0.000157,
            0.000414,
            0.000656,
            0.000872,
            0.000833,
            0.000316,
            0.001154,
            0.000000,
            0.001185,
            0.002012,
            0.001289,
            0.003890,
            0.001699,
            0.005649,
            0.001940,
            0.002749,
            0.002248,
            0.002143,
            0.003369,
            0.003187,
            0.001876,
            0.001352,
            0.002887,
            0.004025,
        ],
        "NextDownID": [
            0,
            71028858,
            71029036,
            71029190,
            71028858,
            71029426,
            71029284,
            71029284,
            71030162,
            71029768,
            71029904,
            71030224,
            71029426,
            71030355,
            71030355,
            71030224,
            71029768,
            71029036,
            71030297,
            71029190,
            71030297,
            71029904,
            71030162,
            71030203,
            71030203,
        ],
        "maxup": [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "up1": [
            71029036,
            71029190,
            71029284,
            71029904,
            71029768,
            71030224,
            71030297,
            71030203,
            71033238,
            71030355,
            71032941,
            71032605,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "up2": [
            71029426,
            71032794,
            71032955,
            71030162,
            71032437,
            71032740,
            71033111,
            71033228,
            71033328,
            71032681,
            71033062,
            71032623,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "up3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "up4": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    n = len(data["COMID"])
    geometries = [LineString([(i, 0), (i, 1)]) for i in range(n)]

    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
    return gdf


@pytest.fixture
def merit_flowpaths_with_cycles() -> gpd.GeoDataFrame:
    data = {
        "COMID": [
            78025040,
            78025154,
            78025845,
            78025880,
            78025914,
            74030207,
            76004317,
            77013933,
            77027716,
            77036548,
            77051408,
        ],
        "lengthkm": [
            9.462773,
            4.617604,
            6.323838,
            3.704298,
            10.140333,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "lengthdir": [
            5.170342,
            3.381714,
            4.233337,
            3.044976,
            6.106415,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "sinuosity": [
            1.830203,
            1.365463,
            1.493819,
            1.216528,
            1.660604,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "slope": [
            0.000190,
            0.005847,
            0.001043,
            0.000566,
            0.006311,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "uparea": [
            239.659283,
            104.240429,
            47.237891,
            30.764579,
            76.656997,
            29.655183,
            31.268485,
            41.694581,
            27.178321,
            26.995637,
            26.085935,
        ],
        "order": [
            2,
            2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        "strmDrop_t": [
            1.8,
            0.0,
            6.6,
            2.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "slope_taud": [
            0.000190,
            0.000000,
            0.001043,
            0.000566,
            0.000000,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "NextDownID": [
            0,
            78025040,
            78025154,
            78025154,
            78025040,
            74030207,
            76004317,
            77013933,
            77027716,
            77036548,
            77051408,
        ],
        "maxup": [
            2,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "up1": [
            78025154,
            78025845,
            0,
            0,
            0,
            74030207,
            76004317,
            77013933,
            77027716,
            77036548,
            77051408,
        ],
        "up2": [
            78025914,
            78025880,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        "up3": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "up4": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    n = len(data["COMID"])
    geometries = [LineString([(i, 0), (i, 1)]) for i in range(n)]

    gdf = gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")
    return gdf


@pytest.fixture
def graph_and_indices(simple_merit_flowpaths):
    """Build graph and node indices from simple_merit_flowpaths."""
    upstream_dict = _build_upstream_dict_from_merit(simple_merit_flowpaths)
    graph, node_indices = _build_rustworkx_object(upstream_dict)
    return graph, node_indices


@pytest.fixture
def graph_and_indices_with_cycle(merit_flowpaths_with_cycles):
    """Build graph and node indices from merit_flowpaths_with_cycles."""
    upstream_dict = _build_upstream_dict_from_merit(merit_flowpaths_with_cycles)
    graph, node_indices = _build_rustworkx_object(upstream_dict)
    return graph, node_indices


@pytest.fixture
def merit_mapping(simple_merit_flowpaths):
    """Create MERIT mapping using actual topological order from create_matrix."""
    _, ts_order = create_matrix(simple_merit_flowpaths)
    return {comid: idx for idx, comid in enumerate(ts_order)}


@pytest.fixture
def merit_mapping_cycles(merit_flowpaths_with_cycles):
    """Create MERIT mapping for network with cycles (after cycle removal)."""
    _, ts_order = create_matrix(merit_flowpaths_with_cycles)
    return {comid: idx for idx, comid in enumerate(ts_order)}


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
