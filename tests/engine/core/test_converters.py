"""Tests for ddr_engine.core.converters — order converter registry."""

import numpy as np
import pytest

pytest.importorskip("ddr_engine")

from ddr_engine.core.converters import (
    _GEODATASET_REGISTRY,
    LynkerOrderConverter,
    MeritOrderConverter,
    get_converter,
    list_geodatasets,
    register_converter,
)


class TestMeritOrderConverter:
    """Tests for MeritOrderConverter."""

    def test_to_zarr_identity(self) -> None:
        c = MeritOrderConverter()
        result = c.to_zarr([10, 20, 30])
        np.testing.assert_array_equal(result, np.array([10, 20, 30], dtype=np.int32))

    def test_to_zarr_dtype(self) -> None:
        c = MeritOrderConverter()
        result = c.to_zarr([10, 20])
        assert result.dtype == np.int32

    def test_from_zarr_returns_list(self) -> None:
        c = MeritOrderConverter()
        result = c.from_zarr(np.array([10, 20], dtype=np.int32))
        assert result == [10, 20]

    def test_roundtrip(self) -> None:
        c = MeritOrderConverter()
        ids = [12345, 67890, 11111]
        assert c.from_zarr(c.to_zarr(ids)) == ids


class TestLynkerOrderConverter:
    """Tests for LynkerOrderConverter."""

    def test_to_zarr_extracts_numeric(self) -> None:
        c = LynkerOrderConverter()
        result = c.to_zarr(["wb-123", "wb-456"])
        np.testing.assert_array_equal(result, np.array([123, 456], dtype=np.int32))

    def test_to_zarr_dtype(self) -> None:
        c = LynkerOrderConverter()
        result = c.to_zarr(["wb-1"])
        assert result.dtype == np.int32

    def test_to_zarr_ghost_nodes(self) -> None:
        c = LynkerOrderConverter()
        result = c.to_zarr(["ghost-0", "wb-123"])
        np.testing.assert_array_equal(result, np.array([0, 123], dtype=np.int32))

    def test_from_zarr_reconstructs_prefix(self) -> None:
        c = LynkerOrderConverter()
        result = c.from_zarr(np.array([123, 456], dtype=np.int32))
        assert result == ["wb-123", "wb-456"]

    def test_ghost_node_lossy(self) -> None:
        """Ghost nodes round-trip as 'wb-0' (documented lossy behavior)."""
        c = LynkerOrderConverter()
        arr = c.to_zarr(["ghost-0", "wb-123"])
        result = c.from_zarr(arr)
        assert result == ["wb-0", "wb-123"]


class TestGetConverter:
    """Tests for get_converter() registry lookup."""

    def test_get_converter_merit(self) -> None:
        c = get_converter("merit")
        assert isinstance(c, MeritOrderConverter)

    def test_get_converter_lynker(self) -> None:
        c = get_converter("lynker")
        assert isinstance(c, LynkerOrderConverter)

    def test_get_converter_hydrofabric_alias(self) -> None:
        c = get_converter("hydrofabric_v2.2")
        assert isinstance(c, LynkerOrderConverter)

    def test_get_converter_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown geodataset"):
            get_converter("nonexistent")

    def test_get_converter_error_lists_available(self) -> None:
        with pytest.raises(ValueError, match="merit") as exc_info:
            get_converter("nonexistent")
        assert "lynker" in str(exc_info.value)


class TestRegisterConverter:
    """Tests for register_converter()."""

    def test_register_converter_adds_to_registry(self) -> None:
        class CustomConverter:
            def to_zarr(self, ids):
                return np.array(ids, dtype=np.int32)

            def from_zarr(self, order):
                return order.tolist()

        try:
            register_converter("_test_custom", CustomConverter())
            c = get_converter("_test_custom")
            assert isinstance(c, CustomConverter)
        finally:
            _GEODATASET_REGISTRY.pop("_test_custom", None)


class TestListGeodatasets:
    """Tests for list_geodatasets()."""

    def test_list_geodatasets_sorted(self) -> None:
        result = list_geodatasets()
        assert result == sorted(result)
        assert "merit" in result
        assert "lynker" in result
        assert "hydrofabric_v2.2" in result
