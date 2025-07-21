"""A file to test utility functions for the dataset module"""

import numpy as np
import pytest
import torch

from ddr.dataset import utils


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 3.0),
        (np.array([1.0, np.nan, 3.0]), 2.0),
        (np.array([1.0, np.inf, 3.0]), 2.0),
        (np.array([np.nan, np.inf, -np.inf]), np.nan),
        (np.array([]), np.nan),
    ],
)
def test_naninfmean(test_input, expected):
    result = utils.naninfmean(test_input)
    if np.isnan(expected):
        assert np.isnan(result)
    else:
        assert np.isclose(result, expected)


@pytest.mark.parametrize(
    "test_input,row_means,expected",
    [
        (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), None, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        (torch.tensor([1.0, float("nan"), 3.0]), None, torch.tensor([1.0, 2.0, 3.0])),
        (torch.tensor([1.0, float("nan"), 3.0]), torch.tensor(10.0), torch.tensor([1.0, 10.0, 3.0])),
        (
            torch.tensor([float("nan"), float("nan"), float("nan")]),
            torch.tensor(7.5),
            torch.tensor([7.5, 7.5, 7.5]),
        ),
        (torch.tensor([float("nan"), float("nan"), float("nan")]), None, "all_nan"),
        (torch.tensor([1.0, float("nan"), 3.0]), torch.tensor([10.0]), torch.tensor([1.0, 10.0, 3.0])),
        (torch.tensor([[1.0, float("nan")], [3.0, 4.0]]), None, torch.tensor([[1.0, 2.67], [3.0, 4.0]])),
        (
            torch.tensor([[1.0, float("nan")], [float("nan"), 4.0]]),
            torch.tensor([5.0, 6.0]),
            torch.tensor([[1.0, 5.0], [6.0, 4.0]]),
        ),
        (torch.tensor([[[1.0, float("nan")]]]), torch.tensor(99.0), torch.tensor([[[1.0, 99.0]]])),
        (torch.tensor([[[[1.0]]]], dtype=torch.float32), None, torch.tensor([[[[1.0]]]])),
    ],
)
def test_fill_nans(test_input, row_means, expected):
    result = utils.fill_nans(test_input, row_means=row_means)

    # Test shape preservation
    assert result.shape == test_input.shape, (
        f"Shape mismatch: input {test_input.shape} vs output {result.shape}"
    )

    if isinstance(expected, str) and expected == "all_nan":
        assert torch.isnan(result).all()
    else:
        assert torch.allclose(result, expected, equal_nan=False, rtol=1e-2)


def test_fill_nans_device_consistency():
    """Test that fill_nans handles device mismatches correctly"""
    if torch.cuda.is_available():
        attr_cpu = torch.tensor([1.0, float("nan"), 3.0])
        row_means_gpu = torch.tensor([10.0]).cuda()

        result = utils.fill_nans(attr_cpu, row_means=row_means_gpu)

        assert result.device == attr_cpu.device
        assert result.shape == attr_cpu.shape


def test_fill_nans_edge_case_shapes():
    """Test specific edge cases that could cause view() to fail"""

    single = torch.tensor([float("nan")])
    result = utils.fill_nans(single, torch.tensor([42.0]))
    assert result.shape == single.shape
    assert result.item() == 42.0

    high_dim = torch.tensor([1.0, float("nan")]).view(1, 1, 1, 2)
    result = utils.fill_nans(high_dim, torch.tensor([99.0]))
    assert result.shape == high_dim.shape
    assert result.flatten()[1].item() == 99.0
