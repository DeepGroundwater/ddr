"""Test utilities for routing tests."""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig


def create_mock_config() -> DictConfig:
    """Create a mock configuration for testing routing models."""
    cfg = {
        "params": {
            "parameter_ranges": {"range": {"n": [0.01, 0.1], "q_spatial": [0.1, 0.9]}},
            "defaults": {"p": 1.0},
            "attribute_minimums": {
                "velocity": 0.1,
                "depth": 0.01,
                "discharge": 0.001,
                "bottom_width": 0.1,
                "slope": 0.0001,
            },
            "tau": 24,
        },
        "device": "cpu",
    }
    return DictConfig(cfg)


def create_mock_hydrofabric(num_reaches: int = 10, device: str = "cpu") -> Any:
    """Create a mock hydrofabric object for testing.

    Parameters
    ----------
    num_reaches : int, optional
        Number of reaches in the network, by default 10
    device : str, optional
        Device for tensors, by default 'cpu'

    Returns
    -------
    Any
        Mock hydrofabric object
    """

    class MockObservations:
        def __init__(self):
            self.gage_id = torch.tensor([1, 2], device=device)

    class MockHydrofabric:
        def __init__(self):
            self.observations = MockObservations()
            # Create a simple network with sequential connections
            self.adjacency_matrix = torch.zeros(num_reaches, num_reaches, device=device)
            for i in range(num_reaches - 1):
                self.adjacency_matrix[i + 1, i] = 1.0  # i flows to i+1

            # Channel properties
            self.length = torch.ones(num_reaches, device=device) * 1000.0
            self.slope = torch.ones(num_reaches, device=device) * 0.001
            self.top_width = torch.ones(num_reaches, device=device) * 10.0
            self.side_slope = torch.ones(num_reaches, device=device) * 2.0
            self.x = torch.ones(num_reaches, device=device) * 0.2

            # Add some variability
            self.length += torch.randn(num_reaches, device=device) * 100
            self.slope += torch.randn(num_reaches, device=device) * 0.0001
            self.top_width += torch.randn(num_reaches, device=device) * 2

            # Ensure positive values
            self.length = torch.clamp(self.length, min=100.0)
            self.slope = torch.clamp(self.slope, min=0.0001)
            self.top_width = torch.clamp(self.top_width, min=1.0)
            self.side_slope = torch.clamp(self.side_slope, min=0.5)
            self.x = torch.clamp(self.x, min=0.1, max=0.4)

    return MockHydrofabric()


def create_mock_streamflow(num_timesteps: int, num_reaches: int, device: str = "cpu") -> torch.Tensor:
    """Create mock streamflow data for testing.

    Parameters
    ----------
    num_timesteps : int
        Number of time steps
    num_reaches : int
        Number of reaches
    device : str, optional
        Device for tensors, by default 'cpu'

    Returns
    -------
    torch.Tensor
        Mock streamflow data with shape (num_timesteps, num_reaches)
    """
    # Create realistic streamflow patterns
    base_flow = torch.ones(num_timesteps, num_reaches, device=device) * 5.0

    # Add some temporal variation (sine wave)
    time_variation = torch.sin(torch.linspace(0, 4 * np.pi, num_timesteps, device=device))
    base_flow += time_variation.unsqueeze(1) * 2.0

    # Add some spatial variation
    spatial_variation = torch.randn(num_reaches, device=device) * 0.5
    base_flow += spatial_variation.unsqueeze(0)

    # Ensure positive values
    return torch.clamp(base_flow, min=0.1)


def create_mock_spatial_parameters(num_reaches: int, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create mock spatial parameters for testing.

    Parameters
    ----------
    num_reaches : int
        Number of reaches
    device : str, optional
        Device for tensors, by default 'cpu'

    Returns
    -------
    Dict[str, torch.Tensor]
        Mock spatial parameters (normalized values between 0 and 1)
    """
    return {
        "n": torch.rand(num_reaches, device=device),  # Normalized Manning's n
        "q_spatial": torch.rand(num_reaches, device=device),  # Normalized q_spatial
    }


def assert_tensor_properties(
    tensor: torch.Tensor,
    expected_shape: tuple,
    expected_dtype: torch.dtype = torch.float32,
    min_val: float = None,
    max_val: float = None,
) -> None:
    """Assert tensor has expected properties.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    expected_shape : tuple
        Expected tensor shape
    expected_dtype : torch.dtype, optional
        Expected tensor dtype, by default torch.float32
    min_val : float, optional
        Minimum expected value, by default None
    max_val : float, optional
        Maximum expected value, by default None
    """
    assert isinstance(tensor, torch.Tensor), f"Expected torch.Tensor, got {type(tensor)}"
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"

    if min_val is not None:
        assert tensor.min().item() >= min_val, f"Minimum value {tensor.min().item()} < {min_val}"

    if max_val is not None:
        assert tensor.max().item() <= max_val, f"Maximum value {tensor.max().item()} > {max_val}"


def assert_no_nan_or_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert tensor contains no NaN or Inf values.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    name : str, optional
        Name of tensor for error messages, by default "tensor"
    """
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"


def create_test_scenarios() -> list:
    """Create various test scenarios for comprehensive testing.

    Returns
    -------
    list
        List of test scenario dictionaries
    """
    scenarios = [
        {
            "name": "small_network",
            "num_reaches": 5,
            "num_timesteps": 24,
            "description": "Small network with 5 reaches, 24 hours",
        },
        {
            "name": "medium_network",
            "num_reaches": 50,
            "num_timesteps": 48,
            "description": "Medium network with 50 reaches, 48 hours",
        },
        {
            "name": "large_network",
            "num_reaches": 100,
            "num_timesteps": 72,
            "description": "Large network with 100 reaches, 72 hours",
        },
        {
            "name": "single_reach",
            "num_reaches": 1,
            "num_timesteps": 12,
            "description": "Single reach, 12 hours",
        },
    ]
    return scenarios
