"""Tests for ddr.io.functions — downsample()."""

import torch

from ddr.io.functions import downsample


class TestDownsample:
    """Tests for downsample()."""

    def test_downsample_shape(self) -> None:
        # 5 reaches, 48 hours → 2 days
        data = torch.rand(5, 48)
        result = downsample(data, rho=2)
        assert result.shape == (5, 2)

    def test_downsample_preserves_mean(self) -> None:
        # Constant data → mean should be preserved
        data = torch.full((3, 24), 7.0)
        result = downsample(data, rho=1)
        assert torch.isclose(result.mean(), torch.tensor(7.0), atol=1e-4)

    def test_downsample_matches_manual_mean(self) -> None:
        # 1 reach, 24 hours of flat ones → 1 day should average to 1.0
        data = torch.ones(1, 24)
        result = downsample(data, rho=1)
        assert torch.isclose(result[0, 0], torch.tensor(1.0), atol=1e-4)
