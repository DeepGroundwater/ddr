"""
Tests for any icechunk functionality
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
import xarray as xr
from moto import mock_aws

from ddr import read_ic


class TestReadIcLocalStore:
    """Unit tests for local store functionality."""

    def test_read_ic_local_store_basic(self, mock_icechunk_components, sample_local_paths):
        """Test reading from a local icechunk store."""
        local_store_path = sample_local_paths["tmp"]

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

    def test_read_ic_local_store_with_pathlib(self, mock_icechunk_components, sample_local_paths):
        """Test reading from a local store using pathlib.Path."""
        local_store_path = Path(sample_local_paths["tmp"])

        with (
            patch("icechunk.local_filesystem_storage") as mock_local_fs,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Setup mocks
            mock_local_fs.return_value = mock_icechunk_components["mock_local_storage"]
            mock_repo_open.return_value = mock_icechunk_components["mock_repo"]
            mock_open_zarr.return_value = mock_icechunk_components["mock_dataset"]

            # Call function with Path object (convert to string)
            result = read_ic(str(local_store_path))

            # Verify calls
            mock_local_fs.assert_called_once_with(str(local_store_path))
            assert result == mock_icechunk_components["mock_dataset"]

    @pytest.mark.parametrize("path_key", ["tmp", "relative", "absolute"])
    def test_read_ic_local_store_different_paths(
        self, mock_icechunk_components, sample_local_paths, path_key
    ):
        """Test reading from different local store paths."""
        local_store_path = sample_local_paths[path_key]

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
            assert result == mock_icechunk_components["mock_dataset"]


class TestReadIcLocalIntegration:
    """Integration tests for local store using real temporary filesystem."""

    def test_read_ic_with_tempfile_basic(self, mock_icechunk_components, integration_test_config):
        """Integration test using a temporary directory for local store."""
        store_name = integration_test_config["store_name"]

        mock_dataset = mock_icechunk_components["mock_dataset"]
        mock_session = MagicMock()
        mock_session.store = "temp_store_object"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = str(Path(temp_dir) / store_name)

            # Verify temp directory exists
            assert Path(temp_dir).exists()
            assert Path(temp_dir).is_dir()

            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                # Setup mocks
                mock_local_fs.return_value = "temp_local_storage_config"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                # Call function
                result = read_ic(store_path)

                # Verify calls
                mock_local_fs.assert_called_once_with(store_path)
                assert result == mock_dataset

    def test_read_ic_with_multiple_temp_directories(self):
        """Integration test with multiple temporary directories."""

        mock_dataset = MagicMock(spec=xr.Dataset)
        mock_session = MagicMock()
        mock_session.store = "multi_temp_store"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        # Test multiple temporary directories
        temp_dirs = []
        store_paths = []

        try:
            # Create multiple temp directories
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f"test_icechunk_{i}_")
                temp_dirs.append(temp_dir)
                store_path = str(Path(temp_dir) / f"store_{i}")
                store_paths.append(store_path)

                # Verify directory exists
                assert Path(temp_dir).exists()

            # Test each store path
            for i, store_path in enumerate(store_paths):
                with (
                    patch("icechunk.local_filesystem_storage") as mock_local_fs,
                    patch("icechunk.Repository.open") as mock_repo_open,
                    patch("xarray.open_zarr") as mock_open_zarr,
                ):
                    # Setup mocks
                    mock_local_fs.return_value = f"temp_config_{i}"
                    mock_repo_open.return_value = mock_repo
                    mock_open_zarr.return_value = mock_dataset

                    # Call function
                    result = read_ic(store_path)

                    # Verify
                    mock_local_fs.assert_called_once_with(store_path)
                    assert result == mock_dataset

        finally:
            # Cleanup temp directories
            import shutil

            for temp_dir in temp_dirs:
                if Path(temp_dir).exists():
                    shutil.rmtree(temp_dir)

    def test_read_ic_with_nested_temp_structure(self):
        """Integration test with nested temporary directory structure."""

        mock_dataset = MagicMock(spec=xr.Dataset)
        mock_session = MagicMock()
        mock_session.store = "nested_temp_store"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create nested structure
            nested_path = Path(temp_dir) / "data" / "icechunk" / "stores" / "test_store"
            nested_path.parent.mkdir(parents=True, exist_ok=True)

            store_path = str(nested_path)

            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                # Setup mocks
                mock_local_fs.return_value = "nested_local_config"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                # Call function
                result = read_ic(store_path)

                # Verify
                mock_local_fs.assert_called_once_with(store_path)
                assert result == mock_dataset

                # Verify nested structure exists
                assert nested_path.parent.exists()


class TestReadIcLocalErrorHandling:
    """Error handling tests for local store operations."""

    def test_read_ic_repository_open_failure(self, mock_icechunk_components, sample_local_paths):
        """Test handling of repository open failure."""
        local_store_path = sample_local_paths["nonexistent"]

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

    def test_read_ic_xarray_open_failure(self, mock_icechunk_components, sample_local_paths):
        """Test handling of xarray open failure."""
        local_store_path = sample_local_paths["tmp"]

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

    def test_read_ic_error_handling_integration(self):
        """Integration test for error handling with real filesystem operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with a path that exists but will fail in icechunk operations
            store_path = str(Path(temp_dir) / "failing_store")

            # Test repository open failure
            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
            ):
                mock_local_fs.return_value = "failing_config"
                mock_repo_open.side_effect = Exception("Integration test: Repository open failed")

                with pytest.raises(Exception, match="Integration test: Repository open failed"):
                    read_ic(store_path)

            # Test xarray open failure
            from unittest.mock import MagicMock

            mock_session = MagicMock()
            mock_session.store = "failing_store_object"
            mock_repo = MagicMock()
            mock_repo.readonly_session.return_value = mock_session

            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                mock_local_fs.return_value = "failing_config"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.side_effect = Exception("Integration test: XArray open failed")

                with pytest.raises(Exception, match="Integration test: XArray open failed"):
                    read_ic(store_path)


class TestReadIcS3Store:
    """Unit tests for S3 store functionality."""

    def test_read_ic_s3_store_default_region(self, mock_icechunk_components, sample_s3_paths):
        """Test reading from an S3 icechunk store with default region."""
        s3_store_path = sample_s3_paths["simple"]

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

    def test_read_ic_s3_store_custom_region(self, mock_icechunk_components, sample_s3_paths):
        """Test reading from an S3 icechunk store with custom region."""
        s3_store_path = sample_s3_paths["simple"]
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

    def test_read_ic_s3_complex_prefix(self, mock_icechunk_components, sample_s3_paths):
        """Test S3 store with complex prefix path."""
        s3_store_path = sample_s3_paths["complex"]

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

            # Verify that the full path after bucket is used as prefix
            mock_s3_storage.assert_called_once_with(
                bucket="test-bucket",
                prefix="deep/nested/prefix/path",  # Now expects full path
                region="us-east-2",
                anonymous=True,
            )

            # Verify result
            assert result == mock_icechunk_components["mock_dataset"]

    @pytest.mark.parametrize("region", ["us-east-1", "us-west-1", "us-west-2", "eu-west-1", "ap-southeast-1"])
    def test_read_ic_s3_different_regions(self, mock_icechunk_components, sample_s3_paths, region):
        """Test function with different AWS regions."""
        s3_store_path = sample_s3_paths["simple"]

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


class TestReadIcS3Integration:
    """Integration tests for S3 store using moto for S3 mocking."""

    @mock_aws
    def test_read_ic_with_moto_s3_mock_basic(self, integration_test_config):
        """Integration test using moto to mock S3 infrastructure."""
        # Get config values
        bucket_name = integration_test_config["bucket_name"]
        prefix = integration_test_config["prefix"]
        region = integration_test_config["region"]

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

    @mock_aws
    def test_read_ic_with_moto_s3_multiple_buckets(self, aws_regions):
        """Integration test with multiple S3 buckets in different regions."""
        test_cases = [
            ("test-bucket-1", "prefix-1", aws_regions[0]),
            ("test-bucket-2", "prefix-2", aws_regions[1]),
            ("test-bucket-3", "prefix-3", aws_regions[2]),
        ]

        for bucket_name, prefix, region in test_cases:
            # Setup S3 client for this region
            s3_client = boto3.client("s3", region_name=region)

            # Create bucket (different logic for us-east-1)
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )

            # Mock components
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_session = MagicMock()
            mock_session.store = f"mock_store_{bucket_name}"
            mock_repo = MagicMock()
            mock_repo.readonly_session.return_value = mock_session

            s3_store_path = f"s3://{bucket_name}/{prefix}"

            with (
                patch("icechunk.s3_storage") as mock_s3_storage,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                # Setup mocks
                mock_s3_storage.return_value = f"mock_s3_storage_config_{bucket_name}"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                # Call function
                result = read_ic(s3_store_path, region=region)

                # Verify correct parameters were passed
                mock_s3_storage.assert_called_once_with(
                    bucket=bucket_name, prefix=prefix, region=region, anonymous=True
                )

                # Verify result
                assert result == mock_dataset

    @mock_aws
    def test_read_ic_with_moto_s3_bucket_operations(self):
        """Integration test that verifies S3 bucket operations work correctly."""
        bucket_name = "integration-test-bucket"
        prefix = "data/streamflow"
        region = "us-west-2"

        # Setup S3 infrastructure
        s3_client = boto3.client("s3", region_name=region)
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region})

        # Verify bucket exists
        response = s3_client.list_buckets()
        bucket_names = [bucket["Name"] for bucket in response["Buckets"]]
        assert bucket_name in bucket_names

        # Test with our function
        s3_store_path = f"s3://{bucket_name}/{prefix}"

        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            # Mock components
            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_session = MagicMock()
            mock_session.store = "integration_test_store"
            mock_repo = MagicMock()
            mock_repo.readonly_session.return_value = mock_session

            # Setup mocks
            mock_s3_storage.return_value = "integration_s3_config"
            mock_repo_open.return_value = mock_repo
            mock_open_zarr.return_value = mock_dataset

            # Call function
            result = read_ic(s3_store_path, region=region)

            # Verify integration worked
            assert result == mock_dataset
            # Expect the full prefix path after bucket name
            mock_s3_storage.assert_called_once_with(
                bucket=bucket_name, prefix=prefix, region=region, anonymous=True
            )

    @mock_aws
    def test_read_ic_stress_test_s3_operations(self, aws_regions):
        """Stress test with multiple S3 operations."""
        operations_count = 3

        for i in range(operations_count):
            bucket_name = f"stress-test-bucket-{i}"
            prefix = f"stress-prefix-{i}/data"
            region = aws_regions[i % len(aws_regions)]

            s3_client = boto3.client("s3", region_name=region)

            # Handle region-specific bucket creation
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )

            s3_store_path = f"s3://{bucket_name}/{prefix}"

            mock_dataset = MagicMock(spec=xr.Dataset)
            mock_session = MagicMock()
            mock_session.store = f"stress_s3_store_{i}"
            mock_repo = MagicMock()
            mock_repo.readonly_session.return_value = mock_session

            with (
                patch("icechunk.s3_storage") as mock_s3_storage,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                mock_s3_storage.return_value = f"stress_s3_config_{i}"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                result = read_ic(s3_store_path, region=region)
                assert result == mock_dataset


class TestReadIcS3PathParsing:
    """Unit tests for S3 path parsing logic."""

    def test_s3_path_parsing_edge_cases(self, mock_icechunk_components, s3_path_test_cases):
        """Test edge cases in S3 path parsing."""
        for s3_path, expected_bucket, expected_prefix in s3_path_test_cases:
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

    @pytest.mark.parametrize(
        "store_path,expected_bucket,expected_prefix,expected_region",
        [
            ("s3://bucket1/prefix1", "bucket1", "prefix1", "us-east-2"),
            ("s3://bucket-2/prefix-2", "bucket-2", "prefix-2", "us-east-2"),
            ("s3://test.bucket/test.prefix", "test.bucket", "test.prefix", "us-east-2"),
            ("s3://bucket/", "bucket", "", "us-east-2"),
            ("s3://mhpi-spatial/streamflow/ic", "mhpi-spatial", "streamflow/ic", "us-east-2"),  # Real example
        ],
    )
    def test_s3_path_parsing_parametrized(
        self, mock_icechunk_components, store_path, expected_bucket, expected_prefix, expected_region
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


class TestReadIcMixedIntegration:
    """Integration tests combining S3 and local functionality."""

    @mock_aws
    def test_read_ic_s3_and_local_comparison(self, integration_test_config):
        """Integration test comparing S3 and local store behavior."""
        # S3 setup
        bucket_name = integration_test_config["bucket_name"]
        prefix = integration_test_config["prefix"]
        region = integration_test_config["region"]

        s3_client = boto3.client("s3", region_name=region)
        s3_client.create_bucket(Bucket=bucket_name)

        s3_store_path = f"s3://{bucket_name}/{prefix}"

        # Mock dataset for consistency
        mock_dataset = MagicMock(spec=xr.Dataset)
        mock_dataset.dims = {"divide_id": 822373, "time": 14610}
        mock_session = MagicMock()
        mock_session.store = "comparison_store"
        mock_repo = MagicMock()
        mock_repo.readonly_session.return_value = mock_session

        # Test S3 path
        with (
            patch("icechunk.s3_storage") as mock_s3_storage,
            patch("icechunk.Repository.open") as mock_repo_open,
            patch("xarray.open_zarr") as mock_open_zarr,
        ):
            mock_s3_storage.return_value = "s3_comparison_config"
            mock_repo_open.return_value = mock_repo
            mock_open_zarr.return_value = mock_dataset

            s3_result = read_ic(s3_store_path, region=region)

            # Verify S3 call
            mock_s3_storage.assert_called_once_with(
                bucket=bucket_name, prefix=prefix, region=region, anonymous=True
            )

            # Verify S3 result
            assert s3_result == mock_dataset
            assert s3_result.dims["divide_id"] == 822373
            assert s3_result.dims["time"] == 14610
            mock_repo_open.assert_called_once_with("s3_comparison_config")
            mock_open_zarr.assert_called_once_with("comparison_store", consolidated=False)

        # Test local path with temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            local_store_path = str(Path(temp_dir) / "comparison_store")

            with (
                patch("icechunk.local_filesystem_storage") as mock_local_fs,
                patch("icechunk.Repository.open") as mock_repo_open,
                patch("xarray.open_zarr") as mock_open_zarr,
            ):
                mock_local_fs.return_value = "local_comparison_config"
                mock_repo_open.return_value = mock_repo
                mock_open_zarr.return_value = mock_dataset

                local_result = read_ic(local_store_path)

                # Verify local call
                mock_local_fs.assert_called_once_with(local_store_path)

        # Both should return same mock dataset
        assert s3_result == local_result == mock_dataset
