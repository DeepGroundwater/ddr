"""Regrid the dHBV2 UH retrospective (per-MERIT-catchment Q') onto DDM30 cells.

Q' is a volumetric flux per unit catchment (m3/s), so this is mass-conserving
aggregation: each catchment's discharge is split across the cells it overlaps in
proportion to overlapping area, then summed per cell. Straddling catchments are
handled exactly, unlike midpoint assignment (which the benchmark used).

Writes an icechunk store with dim ``divide_id`` = flat DDM30 cell id and daily
``Qr`` in m3/s, i.e. the same contract ``StreamflowReader`` reads for MERIT.
The catchment->cell area weights are cached alongside as parquet.

Usage:
    uv run python scripts/build_gridded_qprime.py [--limit N] [--out PATH]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from ddr_engine.gridded.forcing import apply_weights, area_weights

from ddr.io.readers import read_ic

CATCHMENTS = Path("/mnt/ssd1/data/merit/cat_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp")
QR_STORE = "/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic"
ADJACENCY = Path("data/ddm30/ddm30_adjacency.zarr")
WEIGHTS_CACHE = Path("data/ddm30/catchment_cell_weights.parquet")
OUT = Path("/mnt/ssd1/data/icechunk/ddm30_conus_uh_retrospective_regridded.ic")
CHUNK_DAYS = 1826  # ~5 years per pass

log = logging.getLogger("build_gridded_qprime")


def build_weights(comids: np.ndarray, adjacency: Path, cache: Path, limit: int | None) -> pd.DataFrame:
    """Area weights from MERIT unit catchments to DDM30 cells (cached to parquet)."""
    if cache.exists() and limit is None:
        log.info("using cached weights %s", cache)
        return pd.read_parquet(cache)
    log.info("reading %d catchment polygons", len(comids))
    gdf = gpd.read_file(CATCHMENTS, columns=["COMID"])
    gdf = gdf[gdf["COMID"].isin(comids)]
    if limit:
        gdf = gdf.head(limit)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    g = zarr.open_group(adjacency, mode="r")
    order, lat, lon = g["order"][:], g["lat"][:], g["lon"][:]
    keep = (lon >= xmin - 0.5) & (lon <= xmax + 0.5) & (lat >= ymin - 0.5) & (lat <= ymax + 0.5)
    cells = order[keep]
    log.info("overlaying %d catchments with %d cells", len(gdf), len(cells))
    w = area_weights(gdf, cells)
    log.info("%d catchment-cell pairs; %d cells receive water", len(w), w["cell"].nunique())
    if limit is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        w.to_parquet(cache)
    return w


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjacency", type=Path, default=ADJACENCY)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=None, help="only N catchments (smoke test)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ds = read_ic(QR_STORE)
    comids = ds.divide_id.values
    weights = build_weights(comids, args.adjacency, WEIGHTS_CACHE, args.limit)
    cells = np.sort(weights["cell"].unique())

    n_time = ds.sizes["time"]
    out = np.zeros((n_time, len(cells)), dtype=np.float32)
    for start in range(0, n_time, CHUNK_DAYS):
        stop = min(start + CHUNK_DAYS, n_time)
        qr = ds["Qr"].isel(time=slice(start, stop)).transpose("time", "divide_id").values
        out[start:stop] = apply_weights(qr, comids, weights, cells).astype(np.float32)
        log.info("regridded days %d-%d", start, stop)

    # Write (divide_id, time), matching the source store's contract. Readers should still
    # transpose by name: a store written the other way round is silently readable, because
    # an out-of-range read returns fill values rather than raising, which yields an
    # all-NaN baseline instead of an error.
    result = xr.Dataset(
        {"Qr": (("divide_id", "time"), out.T)},
        coords={"divide_id": cells, "time": ds.time.values},
    )
    result["Qr"].attrs["units"] = "m^3/s"
    result.attrs["description"] = (
        "dHBV2 UH retrospective Q' area-weighted from MERIT unit catchments onto DDM30 cells"
    )

    import icechunk as ic

    repo = ic.Repository.open_or_create(ic.local_filesystem_storage(str(args.out)))
    session = repo.writable_session("main")
    result.to_zarr(session.store, mode="w", consolidated=False, zarr_format=3)
    session.commit("gridded DDM30 Q' from dHBV2 UH retrospective")
    log.info("wrote %s: %d cells x %d days", args.out, len(cells), n_time)


if __name__ == "__main__":
    main()
