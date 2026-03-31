"""Tests for ddr.geometry.adapters — attribute conversion."""

import numpy as np
import pytest
import xarray as xr

from ddr.geometry.adapters import (
    HYDROATLAS_TO_MERIT,
    MERIT_ATTRIBUTE_NAMES,
    adapt_attributes,
    detect_source,
)


def _make_merit_ds(n_reaches: int = 10) -> xr.Dataset:
    """Create a synthetic MERIT-format attribute dataset."""
    rng = np.random.default_rng(42)
    return xr.Dataset({name: ("reach", rng.uniform(0.1, 100, n_reaches)) for name in MERIT_ATTRIBUTE_NAMES})


def _make_hydroatlas_ds(n_reaches: int = 10) -> xr.Dataset:
    """Create a synthetic HydroATLAS-format attribute dataset."""
    rng = np.random.default_rng(42)
    return xr.Dataset({name: ("reach", rng.uniform(0.1, 100, n_reaches)) for name in HYDROATLAS_TO_MERIT})


class TestDetectSource:
    def test_detects_merit(self):
        ds = _make_merit_ds()
        assert detect_source(ds) == "merit"

    def test_detects_hydroatlas(self):
        ds = _make_hydroatlas_ds()
        assert detect_source(ds) == "hydroatlas"

    def test_returns_none_for_unknown(self):
        ds = xr.Dataset({"unknown_var": ("x", [1, 2, 3])})
        assert detect_source(ds) is None

    def test_merit_takes_precedence_over_extra_vars(self):
        ds = _make_merit_ds()
        ds["extra_var"] = ("reach", np.zeros(10))
        assert detect_source(ds) == "merit"


class TestAdaptAttributes:
    def test_merit_passthrough(self):
        ds = _make_merit_ds()
        result = adapt_attributes(ds, source="merit")
        assert list(result.data_vars) == list(MERIT_ATTRIBUTE_NAMES)
        np.testing.assert_array_equal(result["aridity"].values, ds["aridity"].values)

    def test_merit_auto_detect(self):
        ds = _make_merit_ds()
        result = adapt_attributes(ds, source="auto")
        assert list(result.data_vars) == list(MERIT_ATTRIBUTE_NAMES)

    def test_hydroatlas_conversion(self):
        ds = _make_hydroatlas_ds()
        result = adapt_attributes(ds, source="hydroatlas")
        assert list(result.data_vars) == list(MERIT_ATTRIBUTE_NAMES)

    def test_hydroatlas_auto_detect(self):
        ds = _make_hydroatlas_ds()
        result = adapt_attributes(ds, source="auto")
        assert list(result.data_vars) == list(MERIT_ATTRIBUTE_NAMES)

    def test_hydroatlas_log_transform_uparea(self):
        """upa_sk_smx should be log10-transformed to log10_uparea."""
        ds = _make_hydroatlas_ds()
        ds["upa_sk_smx"] = ("reach", np.array([100.0, 1000.0, 10000.0] + [50.0] * 7))
        result = adapt_attributes(ds, source="hydroatlas")
        expected = np.log10(np.array([100.0, 1000.0, 10000.0]))
        np.testing.assert_allclose(result["log10_uparea"].values[:3], expected, rtol=1e-5)

    def test_preserves_coordinates(self):
        ds = _make_hydroatlas_ds()
        ds = ds.assign_coords(reach_id=("reach", np.arange(10)))
        result = adapt_attributes(ds, source="hydroatlas")
        assert "reach_id" in result.coords

    def test_missing_merit_attribute_raises(self):
        ds = _make_merit_ds()
        ds = ds.drop_vars("aridity")
        with pytest.raises(ValueError, match="Missing MERIT attributes"):
            adapt_attributes(ds, source="merit")

    def test_missing_hydroatlas_attribute_raises(self):
        ds = _make_hydroatlas_ds()
        ds = ds.drop_vars("cly_pc_sav")
        with pytest.raises(ValueError, match="Missing hydroatlas attributes"):
            adapt_attributes(ds, source="hydroatlas")

    def test_unknown_source_raises(self):
        ds = _make_merit_ds()
        with pytest.raises(ValueError, match="Unknown attribute source"):
            adapt_attributes(ds, source="camels")

    def test_auto_detect_failure_raises(self):
        ds = xr.Dataset({"foo": ("x", [1, 2, 3])})
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            adapt_attributes(ds, source="auto")
