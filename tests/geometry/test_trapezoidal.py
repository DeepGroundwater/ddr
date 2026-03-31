"""Tests for ddr.geometry.trapezoidal — pure geometry computation."""

import pytest
import torch

from ddr.geometry.trapezoidal import compute_trapezoidal_geometry


class TestComputeTrapezoidalGeometry:
    """Tests for the trapezoidal cross-section geometry function."""

    def test_returns_all_expected_keys(self):
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.5]),
            discharge=torch.tensor([10.0]),
            slope=torch.tensor([0.001]),
        )
        expected_keys = {
            "depth",
            "top_width",
            "bottom_width",
            "side_slope",
            "cross_sectional_area",
            "wetted_perimeter",
            "hydraulic_radius",
            "velocity",
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes_match_input(self):
        n_reaches = 100
        result = compute_trapezoidal_geometry(
            n=torch.full((n_reaches,), 0.035),
            p_spatial=torch.full((n_reaches,), 21.0),
            q_spatial=torch.full((n_reaches,), 0.5),
            discharge=torch.full((n_reaches,), 10.0),
            slope=torch.full((n_reaches,), 0.001),
        )
        for key, tensor in result.items():
            assert tensor.shape == (n_reaches,), f"{key} has wrong shape"

    def test_all_values_positive(self):
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035, 0.05, 0.1]),
            p_spatial=torch.tensor([21.0, 50.0, 10.0]),
            q_spatial=torch.tensor([0.5, 0.3, 0.8]),
            discharge=torch.tensor([10.0, 100.0, 1.0]),
            slope=torch.tensor([0.001, 0.01, 0.0001]),
        )
        for key, tensor in result.items():
            assert (tensor > 0).all(), f"{key} has non-positive values"

    def test_higher_discharge_gives_greater_depth(self):
        """More flow should mean deeper water."""
        base = {
            "n": torch.tensor([0.035, 0.035]),
            "p_spatial": torch.tensor([21.0, 21.0]),
            "q_spatial": torch.tensor([0.5, 0.5]),
            "slope": torch.tensor([0.001, 0.001]),
        }
        result = compute_trapezoidal_geometry(
            discharge=torch.tensor([10.0, 100.0]),
            **base,
        )
        assert result["depth"][1] > result["depth"][0]
        assert result["top_width"][1] > result["top_width"][0]

    def test_higher_roughness_gives_greater_depth(self):
        """Higher Manning's n should produce deeper flow for same Q."""
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.02, 0.15]),
            p_spatial=torch.tensor([21.0, 21.0]),
            q_spatial=torch.tensor([0.5, 0.5]),
            discharge=torch.tensor([50.0, 50.0]),
            slope=torch.tensor([0.001, 0.001]),
        )
        assert result["depth"][1] > result["depth"][0]

    def test_steeper_slope_gives_lower_depth(self):
        """Steeper channels convey the same Q at shallower depth."""
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035, 0.035]),
            p_spatial=torch.tensor([21.0, 21.0]),
            q_spatial=torch.tensor([0.5, 0.5]),
            discharge=torch.tensor([50.0, 50.0]),
            slope=torch.tensor([0.0001, 0.01]),
        )
        assert result["depth"][0] > result["depth"][1]

    def test_depth_lower_bound_applied(self):
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.5]),
            discharge=torch.tensor([1e-8]),
            slope=torch.tensor([0.001]),
            depth_lb=0.05,
        )
        assert result["depth"].item() >= 0.05

    def test_bottom_width_lower_bound_applied(self):
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.99]),
            discharge=torch.tensor([0.01]),
            slope=torch.tensor([0.001]),
            bottom_width_lb=0.1,
        )
        assert result["bottom_width"].item() >= 0.1

    def test_rectangular_channel_when_q_zero(self):
        """When q_spatial ≈ 0, channel is nearly rectangular (side_slope ≈ 0.5 clamped)."""
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.0]),
            discharge=torch.tensor([10.0]),
            slope=torch.tensor([0.001]),
        )
        # Side slope should be at the lower clamp (0.5)
        assert result["side_slope"].item() == pytest.approx(0.5, abs=0.01)

    def test_area_consistent_with_trapezoid_formula(self):
        """Cross-sectional area should equal (top_width + bottom_width) * depth / 2."""
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.5]),
            discharge=torch.tensor([50.0]),
            slope=torch.tensor([0.001]),
        )
        expected_area = (result["top_width"] + result["bottom_width"]) * result["depth"] / 2
        assert result["cross_sectional_area"].item() == pytest.approx(expected_area.item(), rel=1e-5)

    def test_hydraulic_radius_consistent(self):
        """R = A / P should hold."""
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035]),
            p_spatial=torch.tensor([21.0]),
            q_spatial=torch.tensor([0.5]),
            discharge=torch.tensor([50.0]),
            slope=torch.tensor([0.001]),
        )
        expected_r = result["cross_sectional_area"] / result["wetted_perimeter"]
        assert result["hydraulic_radius"].item() == pytest.approx(expected_r.item(), rel=1e-5)

    def test_gpu_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        result = compute_trapezoidal_geometry(
            n=torch.tensor([0.035], device="cuda"),
            p_spatial=torch.tensor([21.0], device="cuda"),
            q_spatial=torch.tensor([0.5], device="cuda"),
            discharge=torch.tensor([10.0], device="cuda"),
            slope=torch.tensor([0.001], device="cuda"),
        )
        assert result["depth"].device.type == "cuda"
