"""A script to download the dHBV HF v2.2 1981-2019 retrospective"""
import argparse
from pathlib import Path

import icechunk as ic
import xarray as xr
from icechunk.xarray import to_icechunk


def download(download_path: Path):
    """Download icechunk data to specified path"""
    bucket = "mhpi-spatial"
    prefix = "hydrofabric_v2.2_dhbv_retrospective"
    storage_config = ic.s3_storage(bucket=bucket, prefix=prefix, region="us-east-2", anonymous=True)
    repo = ic.Repository.open(storage_config)
    session = repo.writable_session("main")
    ds = xr.open_zarr(session.store, consolidated=False)

    local_storage_config = ic.local_filesystem_storage(str(download_path))
    local_repo = ic.Repository.create(local_storage_config)
    local_session = local_repo.writable_session("main")
    print("### Downloadeding icechunk data. NOTE: THIS CAN TAKE SOME TIME ###")
    to_icechunk(ds, local_session)
    first_snapshot = session.commit("add RASM data to store")
    print(f"### Downloaded locally: {first_snapshot} ###")


def main():
    """A downloading script for icechunk streamflow values"""
    parser = argparse.ArgumentParser(
        description="Download dHBV HF v2.2 1981-2019 retrospective data from MHPI-Spatial"
    )

    parser.add_argument(
        "download_path", type=Path, help="Path where the data should be downloaded (local directory path)"
    )

    args = parser.parse_args()

    if args.download_path.exists():
        msg = f"Error: Path {args.download_path} already exists. Use --overwrite to overwrite."
        print(msg)
        raise FileExistsError(msg)

    args.download_path.parent.mkdir(parents=True, exist_ok=True)

    download(args.download_path)


if __name__ == "__main__":
    main()
