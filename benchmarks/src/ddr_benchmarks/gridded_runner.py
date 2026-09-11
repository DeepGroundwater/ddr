"""CONUS-scale gridded (DDM30) routing benchmark runner.

Routes MERIT dHBV2 lateral inflows, aggregated to ISIMIP DDM30 0.5-degree
cells, through ddr's Muskingum-Cunge engine with fixed spatial parameters
(no KAN), and scores daily discharge against USGS observations at
drainage-area-matched large gauges. A summed-Q' (instantaneous
accumulation, no routing) baseline quantifies the value of routing.

This is a fixed-parameter benchmark: gridded KAN attributes do not exist
yet, so Manning's n and q_spatial are spatially uniform.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr

from .gridded import (
    assign_cells,
    cell_areas_km2,
    downstream_closure,
    snap_gauges,
    topo_accumulate,
)

NCOLS = 720


@dataclass
class GriddedPaths:
    """Data sources for the CONUS gridded benchmark."""

    ddm30_zarr: Path
    riv_shapefile: Path
    qr_icechunk: Path
    obs_icechunk: Path
    gages_csv: Path
    cache_dir: Path
    # Area-weighted Q' regridded onto cells (scripts/build_gridded_qprime.py). When set,
    # the benchmark uses it, matching what training reads. When None it falls back to
    # flowline-midpoint assignment, which misplaces the ~40% of catchments that straddle
    # a cell edge and is kept only for comparison.
    qprime_icechunk: Path | None = None


def _open_icechunk(path: Path) -> "xr.Dataset":  # type: ignore[name-defined]  # noqa: F821
    import icechunk
    import xarray as xr

    storage = icechunk.local_filesystem_storage(str(path))
    repo = icechunk.Repository.open(storage)
    return xr.open_zarr(repo.readonly_session("main").store, consolidated=False)


def comid_cell_map(paths: GriddedPaths) -> pd.DataFrame:
    """COMID -> DDM30 cell id via flowline midpoints (cached parquet)."""
    cache = paths.cache_dir / "comid_cell.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    from pyogrio import read_dataframe

    fp = read_dataframe(str(paths.riv_shapefile), columns=["COMID"])
    mid = fp.geometry.interpolate(0.5, normalized=True)
    df = pd.DataFrame(
        {
            "COMID": fp["COMID"].astype(np.int64),
            "cell": assign_cells(mid.y.values, mid.x.values),
        }
    )
    df.to_parquet(cache)
    return df


def _make_config(save_path: Path) -> "Config":  # type: ignore[name-defined]  # noqa: F821
    from ddr.validation.configs import Config

    return Config(
        name="gridded-conus-benchmark",
        mode="testing",
        geodataset="merit",
        device="cpu",
        data_sources={
            "geospatial_fabric_gpkg": "unused",
            "conus_adjacency": "unused",
        },
        params={
            "save_path": str(save_path),
            "parameter_ranges": {"n": [0.02, 0.2], "q_spatial": [0.0, 1.0]},
            "log_space_parameters": [],
            "defaults": {"p_spatial": 21},
        },
        kan={"input_var_names": [], "learnable_parameters": ["n", "q_spatial"]},
        experiment={"start_time": "1995/10/01", "end_time": "2000/09/30"},
    )


def run_conus_benchmark(
    paths: GriddedPaths,
    start: str = "1995-10-01",
    end: str = "1997-09-30",
    n_manning: float = 0.05,
    q_spatial: float = 0.4,
    min_da_km2: float = 5000.0,
    min_obs_coverage: float = 0.9,
) -> dict:
    """Run the CONUS gridded benchmark.

    Returns
    -------
    dict
        ``gauges``: per-gauge DataFrame (STAID, cell, da_ratio, nse_routed,
        nse_summed, kge_routed, kge_summed); ``neg_solve_rate``: fraction of
        negative triangular solves; ``n_cells``, ``runtime_s``.
    """
    root = zarr.open_group(store=paths.ddm30_zarr, mode="r")
    order = root["order"][:]
    rows_g, cols_g = root["indices_0"][:], root["indices_1"][:]
    pos = {int(cid): i for i, cid in enumerate(order)}
    dn = np.full(len(order), -1, dtype=np.int64)
    dn[cols_g] = rows_g

    # --- network: cells receiving Q' plus their downstream chains ---
    if paths.qprime_icechunk is not None:
        qp_ds = _open_icechunk(paths.qprime_icechunk).sel(time=slice(start, end))
        forced_pos = np.array(sorted({pos[int(c)] for c in qp_ds.divide_id.values if int(c) in pos}))
    else:
        comid_cell = comid_cell_map(paths)
        forced_pos = np.array(sorted({pos[c] for c in comid_cell.cell.unique() if c in pos}))
    node_pos = downstream_closure(forced_pos, dn)
    n = len(node_pos)
    pos_to_c = {int(p): i for i, p in enumerate(node_pos)}

    mask = np.isin(rows_g, node_pos) & np.isin(cols_g, node_pos)
    r_c = np.array([pos_to_c[p] for p in rows_g[mask]])
    c_c = np.array([pos_to_c[p] for p in cols_g[mask]])
    adjacency = torch.sparse_coo_tensor(
        np.stack([r_c, c_c]), torch.ones(len(r_c)), size=(n, n)
    ).to_sparse_csr()
    dn_c = np.full(n, -1, dtype=np.int64)
    dn_c[c_c] = r_c

    # --- gauges: DA-matched snap against accumulated cell areas ---
    upstream_area_all = topo_accumulate(cell_areas_km2(root["lat"][:]), dn)
    upstream_area = {int(order[p]): float(upstream_area_all[p]) for p in node_pos}

    gauges = pd.read_csv(paths.gages_csv, dtype={"STAID": str})
    gauges["STAID"] = gauges.STAID.str.zfill(8)
    gauges = gauges[gauges.DRAIN_SQKM > min_da_km2]
    gauges = snap_gauges(gauges, upstream_area)

    obs_ds = _open_icechunk(paths.obs_icechunk)
    obs_ids = set(obs_ds.gage_id.values.astype(str))
    gauges = gauges[gauges.STAID.isin(obs_ids)]
    coverage = (
        obs_ds["streamflow"]
        .sel(gage_id=gauges.STAID.tolist(), time=slice(start, end))
        .notnull()
        .mean("time")
        .values
    )
    gauges = gauges[coverage > min_obs_coverage]
    outflow_idx = [np.array([pos_to_c[pos[c]]]) for c in gauges.cell]

    # --- lateral inflow: daily Qr summed per cell, repeated to hourly ---
    if paths.qprime_icechunk is not None:
        qp_daily = _cell_qprime(qp_ds, node_pos, order, pos_to_c)
    else:
        qp_daily = _aggregate_qprime(paths, comid_cell, node_pos, order, pos_to_c, pos, start, end)

    return _route_and_score(
        paths,
        root,
        node_pos,
        adjacency,
        dn_c,
        qp_daily,
        gauges,
        outflow_idx,
        obs_ds,
        start,
        end,
        n_manning,
        q_spatial,
    )


def _route_and_score(  # type: ignore[no-untyped-def] # zarr/torch/xarray objects, lazily imported
    paths: GriddedPaths,
    root,
    node_pos,
    adjacency,
    dn_c,
    qp_daily,
    gauges,
    outflow_idx,
    obs_ds,
    start: str,
    end: str,
    n_manning: float,
    q_spatial: float,
) -> dict:
    """Route the prepared network and score it against observations."""
    import numpy as np
    import torch

    from ddr.geodatazoo.dataclasses import RoutingDataclass
    from ddr.routing.torch_mc import dmc
    from ddr_benchmarks.gridded import topo_accumulate

    n = len(node_pos)
    rd = RoutingDataclass(
        adjacency_matrix=adjacency,
        length=torch.tensor(root["length_m"][:][node_pos], dtype=torch.float32),
        slope=torch.tensor(root["slope"][:][node_pos], dtype=torch.float32),
        side_slope=torch.empty(0),
        top_width=torch.empty(0),
        divide_ids=np.arange(n),
        outflow_idx=outflow_idx,
        gage_catchment=gauges.STAID.tolist(),
        flow_scale=None,
    )
    n_norm = (n_manning - 0.02) / (0.2 - 0.02)
    spatial_parameters = {
        "n": torch.full((n,), n_norm),
        "q_spatial": torch.full((n,), q_spatial),
    }
    t0 = time.time()
    model = dmc(cfg=_make_config(paths.cache_dir), device="cpu")
    with torch.no_grad():
        out = model(
            routing_dataclass=rd,
            streamflow=torch.tensor(np.repeat(qp_daily, 24, axis=0)),
            spatial_parameters=spatial_parameters,
        )
    runtime_s = time.time() - t0
    engine = model.routing_engine
    neg_solve_rate = engine.neg_solve_count / max(engine.neg_solve_total, 1)
    routed_daily = out["runoff"].numpy().reshape(len(gauges), -1, 24).mean(axis=2)

    # --- baseline and evaluation ---
    summed_all = topo_accumulate(qp_daily, dn_c)
    summed_daily = np.stack([summed_all[:, idx[0]] for idx in outflow_idx])
    obs = obs_ds["streamflow"].sel(gage_id=gauges.STAID.tolist(), time=slice(start, end)).values

    from ddr.validation.metrics import Metrics

    warmup = 30
    da_scale = (1.0 / gauges.da_ratio.values)[:, None]
    m_routed = Metrics(pred=routed_daily[:, warmup:] * da_scale, target=obs[:, warmup:])
    m_summed = Metrics(pred=summed_daily[:, warmup:] * da_scale, target=obs[:, warmup:])

    result = gauges[["STAID", "DRAIN_SQKM", "cell", "da_ratio"]].reset_index(drop=True)
    result["nse_routed"], result["nse_summed"] = m_routed.nse, m_summed.nse
    result["kge_routed"], result["kge_summed"] = m_routed.kge, m_summed.kge
    return {
        "gauges": result,
        "neg_solve_rate": neg_solve_rate,
        "n_cells": n,
        "runtime_s": runtime_s,
    }


def qprime_matrix(ds: "xr.Dataset", divide_ids: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined] # noqa: F821
    """Return Q' as a dense ``(time, divide)`` array for ``divide_ids``, whatever the store's layout.

    Q' stores are written ``Qr(divide_id, time)`` by contract, but a transposed store is
    silently readable: zarr answers an out-of-range read with fill values instead of
    raising, so indexing axes positionally yields an all-NaN inflow and a zero baseline
    rather than an error. Select and transpose by dimension name, and refuse anything
    whose dimensions match neither layout.
    """
    import numpy as np

    qr = ds["Qr"]
    if set(qr.dims) != {"divide_id", "time"}:
        raise ValueError(f"Qr must have dimensions (divide_id, time) in either order; got {qr.dims}")
    sel = qr.sel(divide_id=np.asarray(divide_ids)).transpose("time", "divide_id")
    return np.nan_to_num(sel.values.astype(np.float32))


def _cell_qprime(
    ds: "xr.Dataset",  # type: ignore[name-defined] # noqa: F821
    node_pos: "np.ndarray",
    order: "np.ndarray",
    pos_to_c: dict[int, int],
) -> "np.ndarray":
    """Lateral inflow per node from an already-cell-indexed Q' store.

    The store carries one series per cell, so this is a lookup rather than an
    aggregation; cells absent from the store contribute nothing.
    """
    import numpy as np

    cells = np.asarray(ds.divide_id.values, dtype=np.int64)
    qr = qprime_matrix(ds, cells)  # (time, cell), layout-safe
    col_of_cell = {int(c): j for j, c in enumerate(cells)}
    qp = np.zeros((qr.shape[0], len(node_pos)), dtype=np.float32)
    for p_ in node_pos:
        j = col_of_cell.get(int(order[int(p_)]))
        if j is not None:
            qp[:, pos_to_c[int(p_)]] = qr[:, j]
    return qp


def _aggregate_qprime(
    paths: GriddedPaths,
    comid_cell: pd.DataFrame,
    node_pos: np.ndarray,
    order: np.ndarray,
    pos_to_c: dict[int, int],
    pos: dict[int, int],
    start: str,
    end: str,
) -> np.ndarray:
    ds = _open_icechunk(paths.qr_icechunk).sel(time=slice(start, end))
    cell_to_node = {int(order[p]): pos_to_c[int(p)] for p in node_pos}
    contributing = comid_cell[comid_cell.cell.isin(cell_to_node)]
    contributing = contributing[contributing.COMID.isin(pd.Index(ds.divide_id.values))]
    qr = qprime_matrix(ds, contributing.COMID.values)  # (time, comid), layout-safe
    qp = np.zeros((qr.shape[0], len(node_pos)), dtype=np.float32)
    np.add.at(qp.T, contributing.cell.map(cell_to_node).values, qr.T)
    return qp


def run_subreach_benchmark(
    paths: GriddedPaths,
    subreach_zarr: Path,
    qprime_icechunk: Path,
    start: str = "1995-10-01",
    end: str = "1996-09-30",
    n_manning: float = 0.05,
    q_spatial: float = 0.4,
    min_da_km2: float = 5000.0,
    min_obs_coverage: float = 0.9,
) -> dict:
    """Same benchmark on a sub-reach network built by scripts/build_subdivided_adjacency.py.

    Each cell's lateral inflow is split evenly across its sub-reaches (mass
    conserving), and a gauge reads the cell's most downstream sub-reach, which
    carries the whole cell's accumulated flow. Q' comes from the area-weighted
    regridded store rather than flowline-midpoint assignment.
    """
    import numpy as np
    import pandas as pd
    import torch
    import zarr

    from ddr_benchmarks.gridded import cell_areas_km2, downstream_closure, snap_gauges, topo_accumulate

    root = zarr.open_group(store=str(subreach_zarr), mode="r")
    node_ids, parent = root["order"][:], root["parent_cell"][:]
    rows_g, cols_g = root["indices_0"][:], root["indices_1"][:]
    dn = np.full(len(node_ids), -1, dtype=np.int64)
    dn[cols_g] = rows_g

    # node area = its share of the parent cell; accumulating gives upstream area per node
    _, inverse, counts = np.unique(parent, return_inverse=True, return_counts=True)
    k_per_node = counts[inverse]
    node_area = cell_areas_km2(root["lat"][:]) / k_per_node
    upstream_area_all = topo_accumulate(node_area, dn)

    qp_ds = _open_icechunk(qprime_icechunk)
    forced_cells = {int(c) for c in qp_ds.divide_id.values}
    forced_pos = np.array(sorted(i for i, p in enumerate(parent) if int(p) in forced_cells))
    node_pos = downstream_closure(forced_pos, dn)
    n = len(node_pos)
    pos_to_c = {int(p): i for i, p in enumerate(node_pos)}

    mask = np.isin(rows_g, node_pos) & np.isin(cols_g, node_pos)
    r_c = np.array([pos_to_c[p] for p in rows_g[mask]])
    c_c = np.array([pos_to_c[p] for p in cols_g[mask]])
    adjacency = torch.sparse_coo_tensor(
        np.stack([r_c, c_c]), torch.ones(len(r_c)), size=(n, n)
    ).to_sparse_csr()
    dn_c = np.full(n, -1, dtype=np.int64)
    dn_c[c_c] = r_c

    # a cell's outlet is its most downstream sub-reach: the one with the largest sub index
    sub_idx = node_ids % 1000
    outlet_of_cell: dict[int, int] = {}
    for p in node_pos:
        cell = int(parent[p])
        if cell not in outlet_of_cell or sub_idx[p] > sub_idx[outlet_of_cell[cell]]:
            outlet_of_cell[cell] = int(p)
    upstream_area = {c: float(upstream_area_all[p]) for c, p in outlet_of_cell.items()}

    gauges = pd.read_csv(paths.gages_csv, dtype={"STAID": str})
    gauges["STAID"] = gauges.STAID.str.zfill(8)
    gauges = gauges[gauges.DRAIN_SQKM > min_da_km2]
    gauges = snap_gauges(gauges, upstream_area)
    obs_ds = _open_icechunk(paths.obs_icechunk)
    obs_ids = set(obs_ds.gage_id.values.astype(str))
    gauges = gauges[gauges.STAID.isin(obs_ids)]
    coverage = (
        obs_ds["streamflow"]
        .sel(gage_id=gauges.STAID.tolist(), time=slice(start, end))
        .notnull()
        .mean("time")
        .values
    )
    gauges = gauges[coverage > min_obs_coverage]
    outflow_idx = [np.array([pos_to_c[outlet_of_cell[int(c)]]]) for c in gauges.cell]

    # lateral inflow: each cell's daily Q' split evenly across its sub-reaches
    qr = qp_ds["Qr"].sel(time=slice(start, end)).transpose("time", "divide_id")
    qr_cells = qr.divide_id.values.astype(np.int64)
    col_of_cell = {int(c): j for j, c in enumerate(qr_cells)}
    qp_daily = np.zeros((qr.shape[0], n), dtype=np.float32)
    qr_vals = np.nan_to_num(qr.values)
    for p in node_pos:
        j = col_of_cell.get(int(parent[p]))
        if j is not None:
            qp_daily[:, pos_to_c[int(p)]] = qr_vals[:, j] / k_per_node[p]

    return _route_and_score(
        paths,
        root,
        node_pos,
        adjacency,
        dn_c,
        qp_daily,
        gauges,
        outflow_idx,
        obs_ds,
        start,
        end,
        n_manning,
        q_spatial,
    )
