"""A class to expose internal ddr functions to outside scripts"""

import numpy as np
import torch
import torch.nn.functional as F


def downsample(data: torch.Tensor, rho: int) -> torch.Tensor:
    """Downsamples data from hourly to daily resolution.

    Parameters
    ----------
    data : torch.Tensor
        The data to downsample.
    rho : int
        The number of days to downsample to.

    Returns
    -------
    torch.Tensor
        The downsampled daily data.
    """
    downsampled_data = F.interpolate(data.unsqueeze(1), size=(rho,), mode="area").squeeze(1)
    return downsampled_data


def mass_conservative_rescale(
    hourly_data: np.ndarray,
    daily_data: np.ndarray,
) -> np.ndarray:
    """Rescale linearly-interpolated hourly data to preserve daily mass balance.

    After linear interpolation from daily to hourly, the daily mean of each
    day's 24 hourly values may not equal the original daily value. This function
    rescales each day's 24 hours so their mean matches the original exactly.

    Parameters
    ----------
    hourly_data : np.ndarray
        Linearly interpolated hourly values, shape (num_hours, num_divides).
    daily_data : np.ndarray
        Original daily values, shape (num_days, num_divides).

    Returns
    -------
    np.ndarray
        Mass-corrected hourly values with same shape as hourly_data.
    """
    result = hourly_data.copy()
    num_hours = result.shape[0]
    num_complete_days = num_hours // 24

    if num_complete_days > 0:
        hourly_block = result[: num_complete_days * 24].reshape(num_complete_days, 24, -1)
        hourly_means = hourly_block.mean(axis=1, keepdims=True)
        daily_block = daily_data[:num_complete_days, np.newaxis, :]
        safe_means = np.where(hourly_means > 1e-8, hourly_means, 1.0)
        correction = np.where(hourly_means > 1e-8, daily_block / safe_means, 1.0)
        result[: num_complete_days * 24] = (hourly_block * correction).reshape(num_complete_days * 24, -1)

    return result
