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
        # Auto mean calculation (row_means=None)
        (torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), None, torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])),
        (torch.tensor([1.0, float("nan"), 3.0]), None, torch.tensor([1.0, 2.0, 3.0])),
        # Provided row_means
        (torch.tensor([1.0, float("nan"), 3.0]), torch.tensor(10.0), torch.tensor([1.0, 10.0, 3.0])),
        (
            torch.tensor([float("nan"), float("nan"), float("nan")]),
            torch.tensor(7.5),
            torch.tensor([7.5, 7.5, 7.5]),
        ),
        # Edge case: all NaN with auto mean
        (torch.tensor([float("nan"), float("nan"), float("nan")]), None, "all_nan"),
    ],
)
def test_fill_nans(test_input, row_means, expected):
    result = utils.fill_nans(test_input, row_means=row_means)

    if isinstance(expected, str) and expected == "all_nan":
        assert torch.isnan(result).all()
    else:
        assert torch.allclose(result, expected, equal_nan=False)
