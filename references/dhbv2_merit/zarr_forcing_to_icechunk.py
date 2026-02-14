"""Convert MERIT forcing zarr v2 groups to a single icechunk store.

Source: 67 zarr v2 subgroups with P, PET, Temp forcings per MERIT region.
Target: Single icechunk store with dimensions (divide_id, time), matching
the existing streamflow store pattern.

Usage:
uv run python references/dhbv2_merit/zarr_forcing_to_icechunk.py --source /projects/mhpi/yxs275/Data/zarr_merit_for_conus_1980-10-01-2010-09-30/forcing  --output /projects/mhpi/data/icechunk/merit_forcing_conus
"""

import argparse
import os
from pathlib import Path

import icechunk as ic
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage
from icechunk.xarray import to_icechunk

VARIABLES = ["P", "PET", "Temp"]
BASE_DATE = "1980-10-01"
DEFAULT_OUTPUT = "/projects/mhpi/data/icechunk/merit_forcing_conus"
# Match the existing streamflow store chunk sizes
CHUNKS = {"divide_id": 3080, "time": 468}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert MERIT forcing zarr v2 groups to icechunk.")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to directory containing zarr v2 subgroups (e.g., 71_0, 72_1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output icechunk store path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def build_time_coord(time_indices: np.ndarray) -> pd.DatetimeIndex:
    """Convert int64 day offsets from 1980-10-01 to a DatetimeIndex.

    Parameters
    ----------
    time_indices : np.ndarray
        Array of int64 values representing days since 1980-10-01.

    Returns
    -------
    pd.DatetimeIndex
        Datetime coordinate array.
    """
    base = pd.Timestamp(BASE_DATE)
    return pd.DatetimeIndex(base + pd.to_timedelta(time_indices, unit="D"))


def load_group(
    group_path: str,
    time_coord: pd.DatetimeIndex,
    time_indices: np.ndarray,
) -> xr.Dataset:
    """Load one zarr v2 subgroup into an xarray Dataset.

    Parameters
    ----------
    group_path : str
        Absolute path to the zarr v2 group directory.
    time_coord : pd.DatetimeIndex
        Pre-built time coordinate (shared across all groups).
    time_indices : np.ndarray
        Expected time index array for validation.

    Returns
    -------
    xr.Dataset
        Dataset with dims (divide_id, time), int64 COMIDs, float32 variables.

    Raises
    ------
    ValueError
        If the group's time array doesn't match the expected one.
    """
    store = zarr.storage.LocalStore(root=group_path, read_only=True)
    root = zarr.open_group(store, mode="r", zarr_format=2)
    available = {name for name, _ in root.arrays()}

    # Validate time consistency (some groups omit the time array)
    if "time" in available:
        group_time = root["time"][:]
        if not np.array_equal(group_time, time_indices):
            raise ValueError(
                f"Time mismatch in {group_path}: expected {len(time_indices)} steps, got {len(group_time)}"
            )

    comids = np.array(root["COMID"][:]).astype(np.int64)
    n_comids = len(comids)

    data_vars = {}
    for var in VARIABLES:
        if var in available:
            data_vars[var] = (["divide_id", "time"], root[var][:].astype(np.float32))
        else:
            print(f"  WARNING: {var} missing in {group_path}, filling with NaN")
            data_vars[var] = (
                ["divide_id", "time"],
                np.full((n_comids, len(time_indices)), np.nan, dtype=np.float32),
            )

    ds = xr.Dataset(
        data_vars,
        coords={"divide_id": comids, "time": time_coord},
    )
    return ds.chunk(CHUNKS)


def main() -> None:
    """Convert all zarr v2 forcing groups into a single icechunk store."""
    args = parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output path already exists: {args.output}. Remove it first to avoid silent overwrite."
        )

    # Enumerate groups — supports flat (source/71_0/) or nested (source/71/71_0/) layouts
    group_paths: list[str] = []
    for entry in sorted(os.listdir(args.source)):
        entry_path = os.path.join(args.source, entry)
        if not os.path.isdir(entry_path):
            continue
        # Check if this directory is itself a zarr group (has a COMID array)
        if os.path.isdir(os.path.join(entry_path, "COMID")):
            group_paths.append(entry_path)
        else:
            # Nested: look one level deeper for zarr subgroups
            for sub in sorted(os.listdir(entry_path)):
                sub_path = os.path.join(entry_path, sub)
                if os.path.isdir(sub_path) and os.path.isdir(os.path.join(sub_path, "COMID")):
                    group_paths.append(sub_path)
    print(f"Found {len(group_paths)} groups in {args.source}")

    # Build time coordinate from first group
    first_store = zarr.storage.LocalStore(root=group_paths[0], read_only=True)
    first_root = zarr.open_group(first_store, mode="r", zarr_format=2)
    time_indices = first_root["time"][:]
    time_coord = build_time_coord(time_indices)
    print(f"Time range: {time_coord[0]} to {time_coord[-1]} ({len(time_coord)} steps)")

    # Create icechunk store
    storage = ic.local_filesystem_storage(str(args.output))
    repo = ic.Repository.create(storage)

    # Encoding for float32 output
    encoding = {var: {"dtype": "float32"} for var in VARIABLES}

    # Write first group
    print(f"[1/{len(group_paths)}] Writing {group_paths[0]}...")
    ds = load_group(group_paths[0], time_coord, time_indices)
    session = repo.writable_session("main")
    to_icechunk(ds, session, encoding=encoding)
    snapshot = session.commit(f"add group {os.path.basename(group_paths[0])}")
    print(f"  {len(ds.divide_id)} COMIDs, commit: {snapshot}")

    # Append remaining groups
    total_comids = len(ds.divide_id)
    for i, group_path in enumerate(group_paths[1:], start=2):
        print(f"[{i}/{len(group_paths)}] Appending {group_path}...")
        ds = load_group(group_path, time_coord, time_indices)
        session = repo.writable_session("main")
        to_icechunk(ds, session, append_dim="divide_id")
        snapshot = session.commit(f"add group {os.path.basename(group_path)}")
        n = len(ds.divide_id)
        total_comids += n
        print(f"  {n} COMIDs (total: {total_comids}), commit: {snapshot}")

    # Verify
    print("\nVerifying output...")
    repo = ic.Repository.open(storage)
    session = repo.readonly_session("main")
    out = xr.open_zarr(session.store, consolidated=False)
    print(f"  divide_id: {len(out.divide_id)} (expected {total_comids})")
    print(f"  time: {len(out.time)}")
    print(f"  variables: {list(out.data_vars)}")
    print(f"  dtypes: { {v: str(out[v].dtype) for v in out.data_vars} }")
    assert len(out.divide_id) == total_comids, (
        f"COMID count mismatch: got {len(out.divide_id)}, expected {total_comids}"
    )
    assert set(out.data_vars) == set(VARIABLES), (
        f"Variable mismatch: {set(out.data_vars)} != {set(VARIABLES)}"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
