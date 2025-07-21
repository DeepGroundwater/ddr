# """Tests for StreamflowReader class."""

# import pytest
# import torch
# import numpy as np
# import xarray as xr
# from unittest.mock import patch, MagicMock
# from omegaconf import DictConfig

# from ddr.dataset import StreamflowReader
# from tests.routing.test_utils import (
#     create_mock_config,
#     create_mock_hydrofabric,
# )

# @pytest.fixture
# def mock_hydrofabric():
#     return create_mock_hydrofabric()

# class TestStreamflowReader:
#     """Test suite for StreamflowReader class."""

#     @pytest.fixture
#     def mock_streamflow_config(self):
#         """Create a mock config for StreamflowReader testing."""
#         return create_mock_config()

#     @pytest.fixture
#     def mock_xarray_streamflow_dataset(self):
#         """Create a mock xarray dataset for streamflow data."""
#         def _create_dataset(divide_ids=None, num_timesteps=100):
#             if divide_ids is None:
#                 divide_ids = ["cat-1", "cat-2", "cat-3", "cat-4", "cat-5"]

#             # Create time dimension
#             time_range = np.arange(num_timesteps)

#             # Create realistic streamflow data (positive values, varying by location and time)
#             np.random.seed(42)  # For reproducible tests
#             streamflow_data = np.random.uniform(0.1, 50.0, (num_timesteps, len(divide_ids)))

#             # Add some temporal patterns (seasonal variation)
#             for i in range(len(divide_ids)):
#                 seasonal = 10 * np.sin(2 * np.pi * time_range / 24) + 15  # Daily cycle
#                 streamflow_data[:, i] += seasonal

#             # Ensure all values are positive
#             streamflow_data = np.maximum(streamflow_data, 0.001)

#             # Create dataset
#             ds = xr.Dataset(
#                 data_vars={
#                     "Qr": (["time", "divide_id"], streamflow_data)
#                 },
#                 coords={
#                     "time": time_range,
#                     "divide_id": divide_ids
#                 }
#             )
#             return ds

#         return _create_dataset

#     @pytest.fixture
#     def mock_icechunk_components(self):
#         """Create mock icechunk components for testing."""
#         from unittest.mock import MagicMock
#         import xarray as xr

#         # Create mock dataset with proper structure
#         mock_dataset = MagicMock(spec=xr.Dataset)
#         mock_dataset.dims = {"divide_id": 5, "time": 100}

#         # Create mock session and repo
#         mock_session = MagicMock()
#         mock_session.store = "mock_streamflow_store"
#         mock_repo = MagicMock()
#         mock_repo.readonly_session.return_value = mock_session

#         # Create mock storage objects
#         mock_local_storage = MagicMock()
#         mock_s3_storage = MagicMock()

#         return {
#             "mock_dataset": mock_dataset,
#             "mock_repo": mock_repo,
#             "mock_session": mock_session,
#             "mock_local_storage": mock_local_storage,
#             "mock_s3_storage": mock_s3_storage,
#         }

#     @pytest.fixture
#     def mock_read_ic(self, mock_xarray_streamflow_dataset, mock_icechunk_components):
#         """Mock the read_ic function using the icechunk pattern."""
#         with (
#             patch('ddr.dataset.streamflow.read_ic') as mock_read_ic,
#             patch('icechunk.local_filesystem_storage') as mock_local_fs,
#             patch('icechunk.s3_storage') as mock_s3_storage,
#             patch('icechunk.Repository.open') as mock_repo_open,
#             patch('xarray.open_zarr') as mock_open_zarr,
#         ):
#             # Setup the full icechunk mock chain
#             dataset = mock_xarray_streamflow_dataset()
#             mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
#             mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
#             mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
#             mock_open_zarr.return_value = dataset

#             # Make read_ic return the dataset directly
#             mock_read_ic.return_value = dataset
#             yield mock_read_ic

#     @pytest.fixture
#     def streamflow_reader(self, mock_streamflow_config, mock_read_ic):
#         """Create a StreamflowReader instance with mocked dependencies."""
#         return StreamflowReader(mock_streamflow_config)

#     def test_initialization(self, mock_streamflow_config, mock_read_ic):
#         """Test that StreamflowReader initializes correctly."""
#         reader = StreamflowReader(mock_streamflow_config)

#         # Check that config is stored
#         assert reader.cfg == mock_streamflow_config

#         # Check that read_ic was called with correct parameters
#         mock_read_ic.assert_called_once_with(
#             mock_streamflow_config.data_sources.streamflow,
#             region=mock_streamflow_config.s3_region
#         )

#         # Check that divide_id_to_index dictionary is created
#         assert isinstance(reader.divide_id_to_index, dict)
#         assert len(reader.divide_id_to_index) > 0

#         # Check that it's a proper PyTorch module
#         assert isinstance(reader, torch.nn.Module)

#     def test_forward_all_valid_divide_ids(self, mock_hydrofabric, streamflow_reader):
#         """Test forward method with all valid divide IDs."""
#         result = streamflow_reader(
#             hydrofabric=mock_hydrofabric,
#             device="cpu",
#             dtype=torch.float32,
#             use_hourly=False
#         )

#         # Check output properties
#         expected_shape = (48, len(mock_hydrofabric.divide_ids))  # 48 timesteps, 3 divides
#         assert result.shape == expected_shape
#         assert result.dtype == torch.float32
#         assert result.device.type == "cpu"

#         # Check that all values are positive (should be filled with real data)
#         assert (result > 0).all()

#         # Check that values are reasonable streamflow values
#         assert result.min() > 0.0
#         assert result.max() < 1000.0  # Reasonable upper bound

#     def test_forward_mixed_valid_invalid_divide_ids(self, mock_hydrofabric, streamflow_reader, caplog):
#         """Test forward method with mix of valid and invalid divide IDs."""
#         mock_hydrofabric[0] = "cat-010"
#         result = streamflow_reader(
#             hydrofabric=mock_hydrofabric,
#             device="cpu",
#             dtype=torch.float32,
#             use_hourly=False
#         )

#         # Check output properties
#         expected_shape = (48, 4)  # 48 timesteps, 4 divides (including invalid ones)
#         assert result.shape == expected_shape

#         # Check that invalid divide IDs are filled with default value (0.001)
#         # Columns 1 and 3 should have the fill value
#         assert torch.allclose(result[:, 1], torch.full((48,), 0.001))
#         assert torch.allclose(result[:, 3], torch.full((48,), 0.001))

#         # Check that valid divide IDs have real data (not the fill value)
#         assert not torch.allclose(result[:, 0], torch.full((48,), 0.001))
#         assert not torch.allclose(result[:, 2], torch.full((48,), 0.001))

#         # Check that warning messages were logged
#         assert "invalid-001 missing from the loaded attributes" in caplog.text
#         assert "invalid-002 missing from the loaded attributes" in caplog.text

#     def test_forward_no_valid_divide_ids(self, mock_hydrofabric, streamflow_reader):
#         """Test forward method with no valid divide IDs raises assertion error."""
#         mock_hydrofabric.divide_ids = np.array([f"wb-{i}"] for i in range(len(mock_hydrofabric.divide_ids)))
#         with pytest.raises(AssertionError, match="No valid divide IDs found"):
#             streamflow_reader(
#                 hydrofabric=mock_hydrofabric,
#                 device="cpu",
#                 dtype=torch.float32,
#                 use_hourly=False
#             )

#     def test_forward_use_hourly_false(self, mock_hydrofabric, streamflow_reader):
#         """Test forward method with use_hourly=False (with interpolation)."""
#         hydrofabric = mock_hydrofabric(
#             divide_ids=["wb-001", "wb-002"]
#         )

#         # This should work with the default mock setup
#         result = streamflow_reader(
#             hydrofabric=hydrofabric,
#             device="cpu",
#             dtype=torch.float32,
#             use_hourly=False
#         )

#         assert result.shape == (48, 2)
#         assert (result > 0).all()

#     def test_forward_single_divide_id(self, mock_hydrofabric, streamflow_reader):
#         """Test forward method with single divide ID."""
#         hydrofabric = mock_hydrofabric(divide_ids=["wb-001"])

#         result = streamflow_reader(
#             hydrofabric=hydrofabric,
#             device="cpu",
#             dtype=torch.float32,
#             use_hourly=False
#         )

#         expected_shape = (48, 1)
#         assert result.shape == expected_shape
#         assert (result > 0).all()


#     def test_empty_divide_ids_list(self, mock_hydrofabric, streamflow_reader):
#         """Test forward method with empty divide_ids list."""
#         hydrofabric = mock_hydrofabric(divide_ids=[])

#         with pytest.raises(AssertionError, match="No valid divide IDs found"):
#             streamflow_reader(
#                 hydrofabric=hydrofabric,
#                 device="cpu",
#                 dtype=torch.float32,
#                 use_hourly=False
#             )
