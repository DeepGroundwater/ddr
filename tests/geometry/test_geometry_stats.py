"""Tests for ddr.geometry.statistics — temporal geometry statistics."""

import numpy as np
import torch

from ddr.geometry.statistics import compute_geometry_statistics


def _make_params(n_reaches: int = 20) -> dict[str, torch.Tensor]:
    """Create uniform KAN parameters for a small network."""
    return {
        "n": torch.full((n_reaches,), 0.035),
        "p_spatial": torch.full((n_reaches,), 21.0),
        "q_spatial": torch.full((n_reaches,), 0.5),
        "slope": torch.full((n_reaches,), 0.001),
    }


class TestComputeGeometryStatistics:
    def test_min_le_median_le_max(self):
        params = _make_params(10)
        q = np.random.default_rng(42).uniform(1.0, 500.0, (30, 10)).astype(np.float32)
        result = compute_geometry_statistics(**params, daily_accumulated_discharge=q)
        for var in ("depth", "top_width", "bottom_width", "side_slope", "hydraulic_radius", "discharge"):
            assert (result[f"{var}_min"] <= result[f"{var}_median"] + 1e-6).all()
            assert (result[f"{var}_median"] <= result[f"{var}_max"] + 1e-6).all()
            assert (result[f"{var}_min"] <= result[f"{var}_mean"] + 1e-6).all()
            assert (result[f"{var}_mean"] <= result[f"{var}_max"] + 1e-6).all()

    def test_constant_discharge_gives_equal_stats(self):
        """When Q is the same every day, min == max == median == mean."""
        params = _make_params(5)
        q = np.full((10, 5), 25.0, dtype=np.float32)
        result = compute_geometry_statistics(**params, daily_accumulated_discharge=q)
        for var in ("depth", "top_width", "bottom_width", "side_slope", "hydraulic_radius", "discharge"):
            np.testing.assert_allclose(result[f"{var}_min"], result[f"{var}_max"], rtol=1e-5)
            np.testing.assert_allclose(result[f"{var}_min"], result[f"{var}_median"], rtol=1e-5)
            np.testing.assert_allclose(result[f"{var}_min"], result[f"{var}_mean"], rtol=1e-5)

    def test_higher_discharge_gives_greater_depth(self):
        """Reach with higher Q across all days should have greater depth stats."""
        n = 2
        params = _make_params(n)
        q = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]], dtype=np.float32)
        result = compute_geometry_statistics(**params, daily_accumulated_discharge=q)
        assert result["depth_mean"][1] > result["depth_mean"][0]
        assert result["top_width_mean"][1] > result["top_width_mean"][0]

    def test_attribute_minimums_forwarded(self):
        """Custom depth lower bound should be respected."""
        params = _make_params(3)
        q = np.full((2, 3), 1e-8, dtype=np.float32)
        result = compute_geometry_statistics(
            **params,
            daily_accumulated_discharge=q,
            attribute_minimums={"depth": 0.5, "bottom_width": 0.01},
        )
        assert (result["depth_min"] >= 0.5).all()
