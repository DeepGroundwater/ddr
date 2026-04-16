"""Clean LSTM lateral inflow icechunk store for DDR routing.

Operations (applied in order):
  1. If units are mm/day, convert to m3/s using MERIT catchment area.
  2. Replace NaN with 1e-6.
  3. Clamp all values < 1e-6 to 1e-6.

The mm/day -> m3/s conversion uses:
    Q_m3s = Q_mm_day * catchsize_km2 / 86.4

Usage:
    # Dry run (report only)
    python references/geo_io/clean_lstm_lateral_inflows.py \
        --store /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic \
        --attributes data/merit_global_attributes_v2.nc \
        --dry-run

    # Apply cleaning
    python references/geo_io/clean_lstm_lateral_inflows.py \
        --store /mnt/ssd1/data/icechunk/daily_lstm_merit_unit_catchments.ic \
        --attributes data/merit_global_attributes_v2.nc
"""

import argparse
from pathlib import Path

import icechunk as ic
import numpy as np
import xarray as xr
import zarr

FLOOR = np.float32(1e-6)
CHUNK_SIZE = 10_000
MM_DAY_UNITS = {"mm/day", "mm/d", "mm day-1"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean LSTM lateral inflows: NaN -> 1e-6, clamp < 1e-6, optional mm/day -> m3/s."
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="Path to the icechunk store containing Qr.",
    )
    parser.add_argument(
        "--attributes",
        type=Path,
        default=None,
        help="Path to MERIT attributes .nc file (needed for mm/day -> m3/s conversion).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report statistics without modifying the store.",
    )
    return parser.parse_args()


def build_catchsize_lookup(attributes_path: Path, divide_ids: np.ndarray) -> np.ndarray:
    """Build a per-divide catchsize array aligned to the icechunk divide_id dimension.

    Parameters
    ----------
    attributes_path : Path
        Path to the MERIT attributes netCDF file containing ``catchsize`` and ``COMID``.
    divide_ids : np.ndarray
        Array of divide_id (COMID) values from the icechunk store.

    Returns
    -------
    np.ndarray
        catchsize in km2, shape ``(len(divide_ids),)``, aligned to divide_ids order.

    Raises
    ------
    ValueError
        If any divide_id is missing from the attributes file.
    """
    attrs_ds = xr.open_mfdataset(attributes_path)
    comid_to_idx = {int(c): i for i, c in enumerate(attrs_ds.COMID.values)}
    catchsize_all = attrs_ds["catchsize"].values

    catchsize = np.empty(len(divide_ids), dtype=np.float64)
    missing = []
    for i, did in enumerate(divide_ids):
        idx = comid_to_idx.get(int(did))
        if idx is None:
            missing.append(did)
            catchsize[i] = np.nan
        else:
            catchsize[i] = catchsize_all[idx]

    attrs_ds.close()

    if missing:
        raise ValueError(f"{len(missing)} divide_ids not found in attributes file. First 10: {missing[:10]}")

    return catchsize


def report_stats(data: np.ndarray, label: str) -> None:
    """Print summary statistics for a chunk of Qr data."""
    n_nan = int(np.isnan(data).sum())
    n_below = int((data < FLOOR).sum())
    n_neg = int((data < 0).sum())
    finite = data[np.isfinite(data)]
    print(
        f"  {label}: "
        f"NaN={n_nan}, <1e-6={n_below}, neg={n_neg}, "
        f"min={np.nanmin(data):.2e}, median={np.nanmedian(finite):.2e}, max={np.nanmax(data):.2e}"
    )


def clean_chunk(
    data: np.ndarray,
    needs_conversion: bool,
    catchsize: np.ndarray | None,
) -> np.ndarray:
    """Clean a chunk of Qr data in-place.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(n_divides, n_times)``, float32.
    needs_conversion : bool
        If True, convert mm/day -> m3/s using catchsize.
    catchsize : np.ndarray or None
        Shape ``(n_divides,)`` in km2. Required when ``needs_conversion`` is True.

    Returns
    -------
    np.ndarray
        Cleaned data (same array, modified in-place).
    """
    if needs_conversion:
        assert catchsize is not None
        # catchsize[:, None] broadcasts over time dimension
        # Q_m3s = Q_mm_day * area_km2 / 86.4
        data = data * (catchsize[:, None].astype(np.float32) / np.float32(86.4))

    np.nan_to_num(data, nan=FLOOR, copy=False)
    np.clip(data, a_min=FLOOR, a_max=None, out=data)
    return data


def main() -> None:
    """Clean an LSTM lateral inflow icechunk store."""
    args = parse_args()

    storage = ic.local_filesystem_storage(str(args.store))
    repo = ic.Repository.open(storage)

    # Use readonly session for dry-run, writable for actual cleaning
    if args.dry_run:
        session = repo.readonly_session("main")
    else:
        session = repo.writable_session("main")

    ds = xr.open_zarr(session.store, consolidated=False)
    divide_ids = ds.divide_id.values
    n_divides = len(divide_ids)
    n_times = len(ds.time)
    units = ds["Qr"].attrs.get("units", "unknown")

    print(f"Store: {args.store}")
    print(f"Shape: ({n_divides} divides, {n_times} timesteps)")
    print(f"Units: {units}")

    needs_conversion = units.lower().strip() in MM_DAY_UNITS
    if needs_conversion:
        print("Detected mm/day units -- will convert to m3/s")
        if args.attributes is None:
            raise ValueError("--attributes is required for mm/day -> m3/s conversion")
        catchsize = build_catchsize_lookup(args.attributes, divide_ids)
    else:
        if units not in {"m^3/s", "m3/s", "cms"}:
            print(f"WARNING: unrecognized units '{units}', proceeding without conversion")
        catchsize = None

    # Process in chunks along divide_id
    total_nan = 0
    total_below = 0

    zstore = zarr.open_group(session.store, mode="r" if args.dry_run else "r+")

    for start in range(0, n_divides, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_divides)
        chunk_label = f"divides [{start}:{end}]"

        # Read chunk
        data = np.array(zstore["Qr"][start:end, :], dtype=np.float32)

        total_nan += int(np.isnan(data).sum())
        total_below += int((data < FLOOR).sum())

        if args.dry_run:
            report_stats(data, chunk_label)
            continue

        chunk_catchsize = catchsize[start:end] if catchsize is not None else None
        data = clean_chunk(data, needs_conversion, chunk_catchsize)

        # Write back
        zstore["Qr"][start:end, :] = data

        print(f"  Cleaned {chunk_label}")

    print(f"\nTotal NaN found: {total_nan}")
    print(f"Total values < 1e-6: {total_below}")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return

    # Update units attr
    zstore["Qr"].attrs["units"] = "m^3/s"

    commit_msg = "clean_lstm_lateral_inflows: NaN->1e-6, clamp<1e-6"
    if needs_conversion:
        commit_msg += ", mm/day->m3/s"
    snapshot_id = session.commit(commit_msg)
    print(f"\nCommitted: {commit_msg}")
    print(f"Snapshot: {snapshot_id}")


if __name__ == "__main__":
    main()
