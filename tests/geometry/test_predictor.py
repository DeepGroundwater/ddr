"""Tests for ddr.geometry.predictor — GeometryPredictor class."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from ddr.geometry.adapters import MERIT_ATTRIBUTE_NAMES
from ddr.geometry.predictor import GeometryPredictor
from ddr.nn import kan


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path) -> Path:
    """Create a minimal KAN checkpoint for testing."""
    nn_model = kan(
        input_var_names=list(MERIT_ATTRIBUTE_NAMES),
        learnable_parameters=["n", "q_spatial"],
        hidden_size=11,
        num_hidden_layers=1,
        grid=3,
        k=3,
        seed=42,
        device="cpu",
    )
    ckpt_path = tmp_path / "test_checkpoint.pt"
    torch.save({"model_state_dict": nn_model.state_dict(), "epoch": 0, "mini_batch": 0}, ckpt_path)
    return ckpt_path


@pytest.fixture
def synthetic_stats(tmp_path: Path) -> Path:
    """Create a minimal attribute statistics JSON for testing."""
    stats = {}
    rng = np.random.default_rng(42)
    for attr in MERIT_ATTRIBUTE_NAMES:
        mean_val = rng.uniform(1, 50)
        std_val = rng.uniform(0.5, 10)
        stats[attr] = {
            "min": float(mean_val - 3 * std_val),
            "max": float(mean_val + 3 * std_val),
            "mean": float(mean_val),
            "std": float(std_val),
            "p10": float(mean_val - 1.28 * std_val),
            "p90": float(mean_val + 1.28 * std_val),
        }
    stats_path = tmp_path / "merit_attribute_statistics_test.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    return stats_path


@pytest.fixture
def predictor(synthetic_checkpoint: Path, synthetic_stats: Path) -> GeometryPredictor:
    """Build a GeometryPredictor from synthetic artifacts."""
    nn_model = kan(
        input_var_names=list(MERIT_ATTRIBUTE_NAMES),
        learnable_parameters=["n", "q_spatial"],
        hidden_size=11,
        num_hidden_layers=1,
        grid=3,
        k=3,
        seed=42,
        device="cpu",
    )
    state = torch.load(synthetic_checkpoint, map_location="cpu")
    nn_model.load_state_dict(state["model_state_dict"])

    with open(synthetic_stats) as f:
        stats = json.load(f)

    means = torch.tensor([stats[a]["mean"] for a in MERIT_ATTRIBUTE_NAMES], dtype=torch.float32)
    stds = torch.tensor([stats[a]["std"] for a in MERIT_ATTRIBUTE_NAMES], dtype=torch.float32)
    ranges = {a: {"p10": stats[a]["p10"], "p90": stats[a]["p90"]} for a in MERIT_ATTRIBUTE_NAMES}

    return GeometryPredictor(
        nn_model=nn_model,
        attribute_names=list(MERIT_ATTRIBUTE_NAMES),
        means=means,
        stds=stds,
        parameter_ranges={"n": [0.015, 0.25], "q_spatial": [0.0, 1.0]},
        log_space_parameters=[],
        defaults={"p_spatial": 21.0},
        attribute_minimums={"discharge": 0.0001, "slope": 0.0001, "depth": 0.01, "bottom_width": 0.01},
        stats_ranges=ranges,
        device="cpu",
    )


def _make_inputs(n_reaches: int = 50) -> tuple[xr.Dataset, xr.DataArray, xr.DataArray]:
    """Create synthetic MERIT-format inputs for predict()."""
    rng = np.random.default_rng(42)
    attrs = xr.Dataset({name: ("reach", rng.uniform(0.1, 100, n_reaches)) for name in MERIT_ATTRIBUTE_NAMES})
    discharge = xr.DataArray(rng.uniform(1.0, 500.0, n_reaches), dims="reach")
    slope = xr.DataArray(rng.uniform(0.0001, 0.01, n_reaches), dims="reach")
    return attrs, discharge, slope


class TestGeometryPredictor:
    def test_predict_returns_xr_dataset(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        assert isinstance(result, xr.Dataset)

    def test_predict_has_all_geometry_vars(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        expected_vars = {
            "depth",
            "top_width",
            "bottom_width",
            "side_slope",
            "cross_sectional_area",
            "wetted_perimeter",
            "hydraulic_radius",
            "velocity",
            "n",
            "p_spatial",
            "q_spatial",
        }
        assert set(result.data_vars) == expected_vars

    def test_predict_output_shape(self, predictor: GeometryPredictor):
        n_reaches = 50
        attrs, discharge, slope = _make_inputs(n_reaches)
        result = predictor.predict(attrs, discharge, slope)
        for var in result.data_vars:
            assert result[var].shape == (n_reaches,), f"{var} has wrong shape"

    def test_predict_all_values_positive(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        for var in result.data_vars:
            assert (result[var].values > 0).all(), f"{var} has non-positive values"

    def test_p_spatial_uses_default_when_not_learned(self, predictor: GeometryPredictor):
        """KAN only learns n and q_spatial; p_spatial should be the default (21.0)."""
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        np.testing.assert_allclose(result["p_spatial"].values, 21.0)

    def test_n_within_bounds(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        assert (result["n"].values >= 0.015).all()
        assert (result["n"].values <= 0.25).all()

    def test_q_spatial_within_bounds(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result = predictor.predict(attrs, discharge, slope)
        assert (result["q_spatial"].values >= 0.0).all()
        assert (result["q_spatial"].values <= 1.0).all()

    def test_predict_deterministic(self, predictor: GeometryPredictor):
        attrs, discharge, slope = _make_inputs()
        result1 = predictor.predict(attrs, discharge, slope)
        result2 = predictor.predict(attrs, discharge, slope)
        for var in result1.data_vars:
            np.testing.assert_array_equal(result1[var].values, result2[var].values)

    def test_predict_with_hydroatlas_names(self, predictor: GeometryPredictor):
        """Should auto-adapt HydroATLAS attribute names."""
        from ddr.geometry.adapters import HYDROATLAS_TO_MERIT

        rng = np.random.default_rng(42)
        n_reaches = 10
        attrs = xr.Dataset(
            {name: ("reach", rng.uniform(0.1, 100, n_reaches)) for name in HYDROATLAS_TO_MERIT}
        )
        discharge = xr.DataArray(rng.uniform(1.0, 500.0, n_reaches), dims="reach")
        slope = xr.DataArray(rng.uniform(0.0001, 0.01, n_reaches), dims="reach")
        result = predictor.predict(attrs, discharge, slope, source="hydroatlas")
        assert isinstance(result, xr.Dataset)
        assert "depth" in result.data_vars
