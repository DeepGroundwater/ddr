"""Extract the Juniata (01567000) bundle from the CONUS stores.

Maintainer-run, one time, on the machine that has /mnt/ssd1. Collaborators
receive the output committed under examples/juniata/data/.

Usage:
    uv run python examples/juniata/extract_bundle.py --out examples/juniata/data
"""

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import icechunk as ic
import numpy as np
import xarray as xr
import zarr

from ddr.io.readers import read_ic

CONUS_ADJ = Path("data/merit_conus_adjacency.zarr")
GAGES_ADJ = Path("data/merit_gages_conus_adjacency.zarr")
ATTRS_NC = Path("data/merit_global_attributes_v2.nc")
GAGES_CSV = Path("references/gage_info/gages_3000.csv")
QPRIME_IC = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"
OBS_IC = "/mnt/ssd1/data/icechunk/usgs_daily_observations"
# Both bundle stores MUST start 1980-01-01: the obs reader indexes time
# positionally from the 1980/01/01 Dates origin with no offset.
TIME_SLICE = slice("1980-01-01", "2010-12-31")

KAN_INPUT_VARS = [
    "SoilGrids1km_clay",
    "aridity",
    "meanelevation",
    "meanP",
    "NDVI",
    "meanslope",
    "log10_uparea",
    "SoilGrids1km_sand",
    "ETPOT_Hargr",
    "Porosity",
]


def reindex_subgraph(
    gage_indices_0: np.ndarray,
    gage_indices_1: np.ndarray,
    gage_order: np.ndarray,
    gage_idx: int,
    conus_order: np.ndarray,
    conus_length_m: np.ndarray,
    conus_slope: np.ndarray,
) -> dict[str, Any]:
    """Remap a CONUS-positional subgraph into a compact 0..N-1 space.

    The new positional order is the members sorted by their CONUS position,
    which preserves the topological (lower-triangular) ordering the forward-
    substitution solve requires.

    Parameters
    ----------
    gage_indices_0, gage_indices_1 : np.ndarray
        COO row (downstream) / col (upstream) as CONUS positional indices.
    gage_order : np.ndarray
        COMIDs of the subgraph members.
    gage_idx : int
        CONUS positional index of the gauge reach.
    conus_order : np.ndarray
        CONUS position -> COMID (full 346k array).
    conus_length_m, conus_slope : np.ndarray
        CONUS positional physical arrays.

    Returns
    -------
    dict[str, Any]
        Re-indexed arrays: indices_0, indices_1, order, length_m, slope,
        gage_idx, shape.
    """
    comid_to_conus_pos = {int(c): i for i, c in enumerate(conus_order)}
    member_conus_pos = np.array(sorted(comid_to_conus_pos[int(c)] for c in gage_order))
    old_to_new = {int(old): new for new, old in enumerate(member_conus_pos)}

    n = len(member_conus_pos)
    new_order = conus_order[member_conus_pos].astype(np.int32)
    return {
        "indices_0": np.array([old_to_new[int(i)] for i in gage_indices_0], dtype=np.int32),
        "indices_1": np.array([old_to_new[int(i)] for i in gage_indices_1], dtype=np.int32),
        "order": new_order,
        "length_m": conus_length_m[member_conus_pos],
        "slope": conus_slope[member_conus_pos],
        "gage_idx": old_to_new[int(gage_idx)],
        "shape": [n, n],
    }


def _write_local_ic(path: Path, ds: xr.Dataset, message: str) -> None:
    """Create a local icechunk store at path and commit ds into it."""
    storage = ic.local_filesystem_storage(str(path))
    repo = ic.Repository.create(storage)
    session = repo.writable_session("main")
    ds.to_zarr(session.store, consolidated=False, mode="w")
    session.commit(message)


def main() -> None:
    """Run the extraction."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("examples/juniata/data"))
    parser.add_argument("--gage", type=str, default="01567000")
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # --- 1. Read the subgraph and re-index it ---
    gages_root = zarr.open_group(str(GAGES_ADJ), mode="r")
    grp = gages_root[args.gage]
    attrs = dict(grp.attrs)
    conus = zarr.open_group(str(CONUS_ADJ), mode="r")
    sub = reindex_subgraph(
        gage_indices_0=grp["indices_0"][:],
        gage_indices_1=grp["indices_1"][:],
        gage_order=grp["order"][:],
        gage_idx=int(attrs["gage_idx"]),
        conus_order=conus["order"][:],
        conus_length_m=conus["length_m"][:],
        conus_slope=conus["slope"][:],
    )
    comids = sub["order"]

    # --- 2. juniata_conus_adjacency.zarr ---
    cz = zarr.open_group(str(out / "juniata_conus_adjacency.zarr"), mode="w")
    for key in ("indices_0", "indices_1", "order"):
        cz.create_array(key, data=sub[key] if key != "order" else sub["order"])
    cz.create_array("values", data=np.ones(len(sub["indices_0"]), dtype=np.uint8))
    cz.create_array("length_m", data=sub["length_m"])
    cz.create_array("slope", data=sub["slope"])

    # --- 3. juniata_gages_adjacency.zarr (single gage group, same schema) ---
    gz = zarr.open_group(str(out / "juniata_gages_adjacency.zarr"), mode="w")
    g = gz.create_group(args.gage)
    g.create_array("indices_0", data=sub["indices_0"])
    g.create_array("indices_1", data=sub["indices_1"])
    g.create_array("values", data=np.ones(len(sub["indices_0"]), dtype=np.uint8))
    g.create_array("order", data=sub["order"])
    g.attrs.update(
        {
            "format": "COO",
            "shape": sub["shape"],
            "geodataset": "merit",
            "gage_catchment": int(attrs["gage_catchment"]),
            "gage_idx": int(sub["gage_idx"]),
            "data_types": {"indices_0": "int32", "indices_1": "int32", "values": "uint8"},
        }
    )

    # --- 4. Q' icechunk store ---
    qds = read_ic(QPRIME_IC)
    q_sub = qds.sel(divide_id=comids, time=TIME_SLICE).compute()
    _write_local_ic(out / "juniata_qprime.ic", q_sub, f"Juniata Q' subset ({args.gage})")

    # --- 5. Obs icechunk store ---
    ods = read_ic(OBS_IC)
    o_sub = ods.sel(gage_id=[args.gage], time=TIME_SLICE).compute()
    _write_local_ic(out / "juniata_obs.ic", o_sub, f"USGS obs subset ({args.gage})")

    # --- 6. Attributes NetCDF ---
    ads = xr.open_mfdataset(str(ATTRS_NC))
    a_sub = ads[KAN_INPUT_VARS].sel(COMID=comids).compute()
    a_sub.to_netcdf(out / "juniata_attributes.nc")

    # --- 7. One-row gage CSV ---
    with open(GAGES_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(r for r in reader if r[0] == args.gage)
    with open(out / "juniata_gage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)

    # --- 8. Manifest ---
    size = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle: {out} ({size})")
    print(f"reaches: {len(comids)}, edges: {len(sub['indices_0'])}, gage_idx: {sub['gage_idx']}")
    print(f"q' time: {q_sub.time.values[0]} .. {q_sub.time.values[-1]}")


if __name__ == "__main__":
    main()
