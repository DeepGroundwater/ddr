"""
Tests for the read_ic function using pytest and moto for AWS S3 mocking.

This module tests both local filesystem and S3-based icechunk store reading.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from moto import mock_aws

from ddr import read_ic


class TestReadIc:
    """Test cases for the read_ic function."""

    @pytest.fixture
    def mock_xarray_dataset(self):
        """Create a mock xarray Dataset that mimics the dhbv2.0 runoff structure."""

        # Create a realistic mock dataset structure
        mock_ds = MagicMock(spec=xr.Dataset)
        # Set realistic dimensions matching your example
        mock_ds.dims = {"divide_id": 822373, "time": 14610}
        # Mock time coordinates
        mock_time_coord = pd.date_range("1980-01-01", "2019-12-31", freq="D")[:14610]
        mock_divide_ids = [
            f"cat-{i}" for i in range(1068193, 1068193 + 822373)
        ]  # making sequental catchment IDs based on an S3 snapshot

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
    def mock_icechunk_components(self, mock_xarray_dataset):
        """Mock icechunk components (Repository, session, etc.)."""
        # Mock session
        mock_session = MagicMock()
        mock_session.store = "mock_store_object"

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        # Mock storage configs
        mock_s3_storage = MagicMock()
        mock_local_storage = MagicMock()

        return {
            "mock_session": mock_session,
            "mock_repo": mock_repo,
            "mock_s3_storage": mock_s3_storage,
            "mock_local_storage": mock_local_storage,
            "mock_dataset": mock_xarray_dataset,
        }

    def test_read_ic_local_store(self, mock_icechunk_components):
        """Test reading from a local icechunk store."""
        local_store_path = "/tmp/test_icechunk_store"

        with (
            patch("icechunk.local_filesystem_storage") as mock_local_fs,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function
            result = read_ic(local_store_path)

            # Verify calls
            mock_local_fs.assert_called_once_with(local_store_path)
            mock_repo_open.assert_called_once_with(mock_icechunk_components["mock_local_storage"])
            mock_icechunk_components["mock_repo"].readonly_session.assert_called_once_with("main")
            mock_open_zarr.assert_called_once_with("mock_store_object", consolidated=False)

            # Verify result structure matches expected dhbv2.0 dataset
            assert result == mock_icechunk_components["mock_dataset"]
            assert hasattr(result, "Qr")  # Should have Qr data variable

    def test_read_ic_s3_store_default_region(self, mock_icechunk_components):
        """Test reading from an S3 icechunk store with default region."""
        s3_store_path = "s3://test-bucket/test-prefix"

        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function
            result = read_ic(s3_store_path)

            # Verify calls
            mock_s3_storage.assert_called_once_with(
                bucket="test-bucket", prefix="test-prefix", region="us-east-2", anonymous=True
            )
            mock_repo_open.assert_called_once_with(mock_icechunk_components["mock_s3_storage"])
            mock_icechunk_components["mock_repo"].readonly_session.assert_called_once_with("main")
            mock_open_zarr.assert_called_once_with("mock_store_object", consolidated=False)

            # Verify result structure matches expected dhbv2.0 dataset
            assert result == mock_icechunk_components["mock_dataset"]
            assert hasattr(result, "Qr")  # Should have Qr data variable

    def test_read_ic_s3_store_custom_region(self, mock_icechunk_components):
        """Test reading from an S3 icechunk store with custom region."""
        s3_store_path = "s3://test-bucket/test-prefix"
        custom_region = "us-west-1"

        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function with custom region
            result = read_ic(s3_store_path, region=custom_region)

            # Verify calls
            mock_s3_storage.assert_called_once_with(
                bucket="test-bucket", prefix="test-prefix", region=custom_region, anonymous=True
            )
            mock_repo_open.assert_called_once_with(mock_icechunk_components["mock_s3_storage"])
            mock_icechunk_components["mock_repo"].readonly_session.assert_called_once_with("main")
            mock_open_zarr.assert_called_once_with("mock_store_object", consolidated=False)

            # Verify result
            assert result == mock_icechunk_components["mock_dataset"]

    def test_read_ic_s3_path_parsing_edge_cases(self, mock_icechunk_components):
        """Test edge cases in S3 path parsing."""
        test_cases = [
            ("s3://bucket-name/prefix", "bucket-name", "prefix"),
            ("s3://bucket/", "bucket", ""),  # Empty prefix
            ("s3://bucket-with-dashes/prefix-with-dashes", "bucket-with-dashes", "prefix-with-dashes"),
        ]

        for s3_path, expected_bucket, expected_prefix in test_cases:
            with (
                patch("icechunk.s3_storage") as mock_s3_storage,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                # Setup mocks
                mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
                mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
                mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

                # Call function
                read_ic(s3_path)

                # Verify parsing
                mock_s3_storage.assert_called_once_with(
                    bucket=expected_bucket, prefix=expected_prefix, region="us-east-2", anonymous=True
                )

    def test_read_ic_local_store_with_pathlib(self, mock_icechunk_components):
        """Test reading from a local store using pathlib.Path."""
        local_store_path = Path("/tmp/test_icechunk_store")

        with (
            patch("icechunk.local_filesystem_storage") as mock_local_fs,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function with Path object
            result = read_ic(str(local_store_path))

            # Verify calls
            mock_local_fs.assert_called_once_with(str(local_store_path))
            assert result == mock_icechunk_components["mock_dataset"]

    def test_read_ic_repository_open_failure(self, mock_icechunk_components):
        """Test handling of repository open failure."""
        local_store_path = "/tmp/nonexistent_store"

        with (
            patch("icechunk.local_filesystem_storage") as mock_local_fs,
            patch("icechunk.Repository.open") as mock_repo_open,
        ):
            # Setup mocks
            mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
            mock_repo_open.side_effect = Exception("Repository not found")

            # Call function and expect exception
            with pytest.raises(Exception, match="Repository not found"):
                read_ic(local_store_path)

    def test_read_ic_xarray_open_failure(self, mock_icechunk_components):
        """Test handling of xarray open failure."""
        local_store_path = "/tmp/test_store"

        with (
            patch("icechunk.local_filesystem_storage") as mock_local_fs,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.side_effect = Exception("Failed to open zarr store")

            # Call function and expect exception
            with pytest.raises(Exception, match="Failed to open zarr store"):
                read_ic(local_store_path)


class TestReadIcIntegration:
    """Integration tests using actual mocked S3 infrastructure."""

    @mock_aws
    def test_read_ic_with_moto_s3_mock(self):
        """Integration test using moto to mock S3 infrastructure."""
        # Create mock S3 infrastructure
        bucket_name = "test-icechunk-bucket"
        prefix = "test-prefix"
        region = "us-east-1"

        # Setup moto S3 mock
        s3_client = boto3.client("s3", region_name=region)
        s3_client.create_bucket(Bucket=bucket_name)

        # Mock the icechunk components since we can't create real icechunk data easily
        mock_dataset = MagicMock(spec=xr.Dataset)
        mock_session = MagicMock()
        mock_session.store = "mock_store_object"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        s3_store_path = f"s3://{bucket_name}/{prefix}"

        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_s3_storage.return_value = "mock_s3_storage_config"
            mock_repo_open.return_value = mock_repo
            mock_open_zarr.return_value = mock_dataset

            # Call function
            result = read_ic(s3_store_path, region=region)

            # Verify the S3 storage configuration was called with correct parameters
            mock_s3_storage.assert_called_once_with(
                bucket=bucket_name, prefix=prefix, region=region, anonymous=True
            )

            # Verify the result
            assert result == mock_dataset

    def test_read_ic_with_tempfile(self):
        """Integration test using a temporary directory for local store."""
        mock_dataset = MagicMock(spec=xr.Dataset)
        mock_session = MagicMock()
        mock_session.store = "mock_store_object"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = str(Path(temp_dir) / "test_icechunk_store")

            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                # Setup mocks
                mock_local_fs.return_value = "mock_local_storage_config"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                # Call function
                result = read_ic(store_path)

                # Verify calls
                mock_local_fs.assert_called_once_with(store_path)
                assert result == mock_dataset


class TestReadIcParametrized:
    """Parametrized tests for different input scenarios."""

    @pytest.mark.parametrize(
        "store_path,expected_bucket,expected_prefix,expected_region",
        [
            ("s3://bucket1/prefix1", "bucket1", "prefix1", "us-east-2"),
            ("s3://bucket-2/prefix-2", "bucket-2", "prefix-2", "us-east-2"),
            ("s3://test.bucket/test.prefix", "test.bucket", "test.prefix", "us-east-2"),
            ("s3://bucket/", "bucket", "", "us-east-2"),
        ],
    )
    def test_s3_path_parsing_parametrized(
        self, store_path, expected_bucket, expected_prefix, expected_region, mock_icechunk_components
    ):
        """Parametrized test for S3 path parsing."""
        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function
            read_ic(store_path)

            # Verify parsing
            mock_s3_storage.assert_called_once_with(
                bucket=expected_bucket, prefix=expected_prefix, region=expected_region, anonymous=True
            )

    @pytest.mark.parametrize(
        "region",
        ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
    )
    def test_different_regions(self, region, mock_icechunk_components):
        """Test function with different AWS regions."""
        s3_store_path = "s3://test-bucket/test-prefix"

        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_s3_storage.return_value = mock_icechunk_components["mock_s3_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function with specific region
            read_ic(s3_store_path, region=region)

            # Verify region is passed correctly
            mock_s3_storage.assert_called_once_with(
                bucket="test-bucket", prefix="test-prefix", region=region, anonymous=True
            )


# Fixture to make mock_icechunk_components available to parametrized tests
@pytest.fixture
def mock_icechunk_components():
    """Create mock icechunk components for parametrized tests."""
    import numpy as np
    import pandas as pd

    # Create realistic mock dataset
    mock_dataset = MagicMock(spec=xr.Dataset)
    mock_dataset.dims = {"divide_id": 822373, "time": 14610}

    # Mock coordinates
    mock_time_coord = pd.date_range("1980-01-01", "2019-12-31", freq="D")[:14610]
    mock_divide_ids = [f"cat-{i}" for i in range(1068193, 1068193 + 822373)]

    mock_dataset.coords = {"time": mock_time_coord, "divide_id": mock_divide_ids}

    # Mock data variables
    mock_dataset.data_vars = ["Qr"]

    # Mock attributes
    mock_dataset.attrs = {"description": "Runoff outputs from dhbv2.0 at the HFv2.2 catchment scale"}

    # Mock the Qr data variable
    mock_qr = MagicMock()
    mock_qr.dims = ("divide_id", "time")
    mock_qr.shape = (822373, 14610)
    mock_qr.dtype = np.float32
    mock_dataset.Qr = mock_qr

    # Mock common dataset methods
    mock_dataset.sel.return_value = mock_dataset
    mock_dataset.isel.return_value = mock_dataset
    mock_dataset.compute.return_value = mock_dataset
    mock_dataset.nbytes = 48 * 1024**3  # 48GB

    # Other icechunk components
    mock_session = MagicMock()
    mock_session.store = "mock_store_object"
    mock_repo = MagicMock()
    mock_repo.readonly_session.return_value = mock_session
    mock_s3_storage = MagicMock()
    mock_local_storage = MagicMock()

    return {
        "mock_session": mock_session,
        "mock_repo": mock_repo,
        "mock_s3_storage": mock_s3_storage,
        "mock_local_storage": mock_local_storage,
        "mock_dataset": mock_dataset,
    }
