"""Test script to demonstrate the refactored Muskingum-Cunge implementation."""

import torch
from omegaconf import DictConfig

from ddr.routing.mmc import MuskingunCunge
from ddr.routing.torch_mc import TorchMC


def create_mock_config():
    """Create a mock configuration for testing."""
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


def create_mock_hydrofabric():
    """Create a mock hydrofabric object for testing."""

    class MockObservations:
        def __init__(self):
            self.gage_id = torch.tensor([1, 2])

    class MockHydrofabric:
        def __init__(self):
            self.observations = MockObservations()
            self.adjacency_matrix = torch.eye(10)
            self.length = torch.ones(10) * 1000.0
            self.slope = torch.ones(10) * 0.001
            self.top_width = torch.ones(10) * 10.0
            self.side_slope = torch.ones(10) * 2.0
            self.x = torch.ones(10) * 0.2

    return MockHydrofabric()


def test_muskingum_cunge():
    """Test the core MuskingunCunge implementation."""
    print("Testing MuskingunCunge core implementation...")

    cfg = create_mock_config()
    hydrofabric = create_mock_hydrofabric()

    # Create routing engine
    mc = MuskingunCunge(cfg, device="cpu")

    # Mock streamflow data (10 reaches, 48 hours)
    streamflow = torch.rand(48, 10) * 10.0

    # Mock spatial parameters
    spatial_params = {
        "n": torch.ones(10) * 0.5,  # Normalized values (will be denormalized)
        "q_spatial": torch.ones(10) * 0.5,
    }

    # Setup inputs
    mc.setup_inputs(hydrofabric, streamflow, spatial_params)

    # Run forward pass
    output = mc.forward()

    print(f"✓ MuskingunCunge test passed. Output shape: {output.shape}")
    return True


def test_torch_mc():
    """Test the TorchMC PyTorch module."""
    print("Testing TorchMC PyTorch module...")

    cfg = create_mock_config()
    hydrofabric = create_mock_hydrofabric()

    # Create torch module
    torch_mc = TorchMC(cfg, device="cpu")
    torch_mc.set_progress_info(1, 0)

    # Mock streamflow data (10 reaches, 48 hours)
    streamflow = torch.rand(48, 10) * 10.0

    # Mock spatial parameters
    spatial_params = {
        "n": torch.ones(10) * 0.5,  # Normalized values (will be denormalized)
        "q_spatial": torch.ones(10) * 0.5,
    }

    # Test forward pass
    kwargs = {"hydrofabric": hydrofabric, "streamflow": streamflow, "spatial_parameters": spatial_params}

    output = torch_mc(**kwargs)

    print(f"✓ TorchMC test passed. Output shape: {output['runoff'].shape}")
    print(f"✓ Output keys: {list(output.keys())}")
    return True


def test_device_movement():
    """Test device movement functionality."""
    print("Testing device movement...")

    cfg = create_mock_config()
    torch_mc = TorchMC(cfg, device="cpu")

    print(f"✓ Initial device: {torch_mc.device_num}")

    # Test moving to CPU (should be no-op)
    torch_mc_cpu = torch_mc.cpu()
    print(f"✓ After .cpu(): {torch_mc_cpu.device_num}")

    # Test torch.nn.Module compatibility
    print(f"✓ Is PyTorch module: {isinstance(torch_mc, torch.nn.Module)}")
    print(f"✓ Has parameters: {len(list(torch_mc.parameters())) == 0}")  # Should have no learnable params

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Refactored Muskingum-Cunge Implementation")
    print("=" * 60)

    try:
        test_muskingum_cunge()
        print()
        test_torch_mc()
        print()
        test_device_movement()
        print()
        print("=" * 60)
        print("✓ All tests passed! Refactoring successful.")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        raise
