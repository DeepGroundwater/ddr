"""
Shared pytest fixtures for read_ic function tests.

This module contains all fixtures used across multiple test files.
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def mock_xarray_dataset():
    """Create a mock xarray Dataset that mimics the dhbv2.0 runoff structure."""
    # Create a realistic mock dataset structure
    mock_ds = MagicMock(spec=xr.Dataset)

    # Set realistic dimensions matching your example
    mock_ds.dims = {"divide_id": 822373, "time": 14610}

    # Mock time coordinates
    mock_time_coord = pd.date_range("1980-01-01", "2019-12-31", freq="D")[:14610]
    mock_divide_ids = [
        f"cat-{i}" for i in range(1068193, 1068193 + 822373)
    ]  # sequential catchment IDs based on an S3 snapshot

    mock_ds.coords = {"time": mock_time_coord, "divide_id": mock_divide_ids}

    # Mock data variables
    mock_ds.data_vars = ["Qr"]

    # Mock attributes
    mock_ds.attrs = {"description": "Runoff outputs from dhbv2.0 at the HFv2.2 catchment scale"}

    mock_qr = MagicMock()
    mock_qr.dims = ("divide_id", "time")
    mock_qr.shape = (822373, 14610)
    mock_qr.dtype = np.float32
    mock_ds.Qr = mock_qr

    # Mock common dataset methods
    mock_ds.sel.return_value = mock_ds  # For selecting subsets
    mock_ds.isel.return_value = mock_ds  # For integer-based selection
    mock_ds.compute.return_value = mock_ds  # For dask computation

    # Mock size properties
    mock_ds.nbytes = 48 * 1024**3  # 48GB in bytes

    return mock_ds


@pytest.fixture
def mock_icechunk_session():
    """Create a mock icechunk session."""
    mock_session = MagicMock()
    mock_session.store = "mock_store_object"
    return mock_session


@pytest.fixture
def mock_icechunk_repository(mock_icechunk_session):
    """Create a mock icechunk repository."""
    mock_repo = MagicMock()
    mock_repo.readonly_session.return_value = mock_icechunk_session
    return mock_repo


@pytest.fixture
def mock_icechunk_storage_configs():
    """Create mock storage configuration objects."""
    return {
        "s3_storage": MagicMock(),
        "local_storage": MagicMock(),
    }


@pytest.fixture
def mock_icechunk_components(
    mock_xarray_dataset, mock_icechunk_session, mock_icechunk_repository, mock_icechunk_storage_configs
):
    """Mock icechunk components (Repository, session, etc.)."""
    return {
        "mock_session": mock_icechunk_session,
        "mock_repo": mock_icechunk_repository,
        "mock_s3_storage": mock_icechunk_storage_configs["s3_storage"],
        "mock_local_storage": mock_icechunk_storage_configs["local_storage"],
        "mock_dataset": mock_xarray_dataset,
    }


@pytest.fixture
def sample_s3_paths():
    """Provide sample S3 paths for testing."""
    return {
        "simple": "s3://test-bucket/test-prefix",
        "complex": "s3://test-bucket/deep/nested/prefix/path",
        "with_dashes": "s3://bucket-with-dashes/prefix-with-dashes",
        "empty_prefix": "s3://bucket/",
        "dots": "s3://test.bucket/test.prefix",
        "real_example": "s3://mhpi-spatial/streamflow/ic-repo",
    }


@pytest.fixture
def sample_local_paths():
    """Provide sample local paths for testing."""
    return {
        "tmp": "/tmp/test_icechunk_store",
        "relative": "./test_store",
        "absolute": "/var/data/icechunk_store",
        "nonexistent": "/tmp/nonexistent_store",
    }


@pytest.fixture
def aws_regions():
    """Provide sample AWS regions for testing."""
    return [
        "us-east-1",
        "us-east-2",
        "us-west-1",
        "us-west-2",
        "eu-west-1",
        "ap-southeast-1",
    ]


@pytest.fixture
def s3_path_test_cases():
    """Provide S3 path parsing test cases."""
    return [
        ("s3://bucket-name/prefix", "bucket-name", "prefix"),
        ("s3://bucket/", "bucket", ""),
        ("s3://bucket-with-dashes/prefix-with-dashes", "bucket-with-dashes", "prefix-with-dashes"),
        ("s3://test.bucket/test.prefix", "test.bucket", "test.prefix"),
    ]


@pytest.fixture
def parametrized_s3_test_cases():
    """Provide parametrized S3 test cases with expected results."""
    return [
        ("s3://bucket1/prefix1", "bucket1", "prefix1", "us-east-2"),
        ("s3://bucket-2/prefix-2", "bucket-2", "prefix-2", "us-east-2"),
        ("s3://test.bucket/test.prefix", "test.bucket", "test.prefix", "us-east-2"),
        ("s3://bucket/", "bucket", "", "us-east-2"),
    ]


@pytest.fixture
def mock_boto3_s3_client():
    """Create a mock boto3 S3 client for integration tests."""
    mock_client = MagicMock()
    mock_client.create_bucket.return_value = {"Location": "/test-bucket"}
    mock_client.list_objects_v2.return_value = {"Contents": []}
    return mock_client


@pytest.fixture(scope="session")
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "bucket_name": "test-icechunk-bucket",
        "prefix": "test-prefix",
        "region": "us-east-1",
        "store_name": "test_icechunk_store",
    }
