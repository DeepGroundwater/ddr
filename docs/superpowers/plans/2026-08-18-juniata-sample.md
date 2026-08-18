# Juniata Single-Catchment Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A self-contained `examples/juniata/` sample — data bundle, plain-Python train/test module, physics notebook — that runs unmodified DDR on the Juniata @ Newport (01567000) subgraph.

**Architecture:** A one-time extraction script re-indexes the 213-reach Juniata subgraph from CONUS positional space into a compact 0..212 space and writes bundle stores with the exact layouts the library readers expect (icechunk `Qr(divide_id,time)` and `streamflow(gage_id,time)`, binsparse COO zarrs, attributes NetCDF, one-row gage CSV). A no-Hydra module builds the same Pydantic `Config` the main scripts use and mirrors their train/test loops at batch_size=1. The notebook teaches the MC physics and calls the module.

**Tech Stack:** Python 3.12+, PyTorch, icechunk 2.0.3 (locked), zarr v3, xarray, nbformat (notebook generation only), `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-08-18-juniata-single-catchment-sample-design.md`

## Global Constraints

- Run everything with `uv` (`uv run pytest ...`, `uv run python -m ...`). Never bare `python`/`pip`.
- Line length 110; ruff + mypy strict; NumPy docstrings. Pre-commit runs on commit — if it modifies files, `git add` and commit again.
- No changes to `src/ddr/` — if a genuine library blocker appears, stop and surface it (spec: separately-reviewed change).
- Bundle committed to git only if `du -sh examples/juniata/data` ≤ 50 MB; otherwise release-tarball path (spec §1).
- Q' source: `/mnt/ssd1/data/icechunk/merit_dhbv2_UH_retrospective.ic`. Obs: `/mnt/ssd1/data/icechunk/usgs_daily_observations`. Both bundle stores must start at **1980-01-01** (the obs reader indexes time positionally from the 1980/01/01 origin with no offset; `readers.py:559`).
- Gauge: `01567000`, COMID 73005278, CONUS positional `gage_idx` 229340, 213-reach subgraph (verified 2026-08-18).
- Unit tests must pass without the bundle (skip with a clear message); bundle-dependent tests use `pytest.mark.skipif`.

## Verified store facts (do not re-derive)

- Per-gage group `data/merit_gages_conus_adjacency.zarr/01567000`: arrays `indices_0`, `indices_1`, `values` (212 edges, int32/uint8), `order` (213 COMIDs); attrs `{format: "COO", shape: [346321, 346321], geodataset: "merit", gage_catchment: 73005278, gage_idx: 229340, data_types: {...}}`. **Indices are CONUS-positional** (into `conus_adjacency["order"]`, length 346,321).
- `data/merit_conus_adjacency.zarr`: `order` (346321 COMIDs), `indices_0/indices_1/values` (338814 edges), `length_m`, `slope` (346321, positional).
- Obs store: variable `streamflow(gage_id, time)`, gage_id zero-padded strings, time 1980-01-01..2019-12-31 daily.
- Q' store: variable `Qr(divide_id, time)`, integer COMID divide_ids, time 1980-01-01..2020-12-31 daily, m³/s.
- `Merit.__init__` reads `conus_adjacency` positional arrays (`order`, `length_m`, `slope`); `_collate_gages`/`_build_routing_data_gages` use only per-gage COO indices + those positional arrays; statistics JSON is generated into `cfg.data_sources.statistics` on first run (`statistics.py:28-44`) — computed over the bundle's 213 catchments (basin-local z-scores; fine for a from-scratch sample, noted in README).
- `DataSources.geospatial_fabric_gpkg` has a default and is unused in training/gages modes — `make_config` does not set it.
- gage CSV header: `STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE,COMID,COMID_DRAIN_SQKM,COMID_UNITAREA_SQKM,ABS_DIFF,DA_VALID,FLOW_SCALE`.

## File structure

```
examples/juniata/
├── extract_bundle.py       # Task 1-2: maintainer-run extraction (re-index + write stores)
├── train_and_test.py       # Task 3: make_config / train / test / summed_qprime_baseline + CLI
├── make_notebook.py        # Task 4: nbformat builder (committed so the notebook is regenerable)
├── juniata_routing.ipynb   # Task 4: generated, outputs stripped
├── README.md               # Task 4
└── data/                   # Task 2: committed bundle (≤50 MB gate)
    ├── juniata_conus_adjacency.zarr/
    ├── juniata_gages_adjacency.zarr/
    ├── juniata_qprime.ic/
    ├── juniata_obs.ic/
    ├── juniata_attributes.nc
    ├── juniata_gage.csv
    └── statistics/         # generated on first run (gitignored)
tests/examples/
├── __init__.py
├── test_reindex.py         # Task 1: pure-function re-indexing tests (no data needed)
└── test_juniata_bundle.py  # Task 2-3: bundle contract + 1-window smoke (skipif no bundle)
```

---

### Task 1: Re-indexing logic + extraction script

The core correctness risk (spec Risk 1): remapping CONUS-positional indices to the compact 0..212 space consistently across the gage COO, the conus subset arrays, and `gage_idx`. Isolate it in a pure function with synthetic-array tests; the extraction script wraps it with store I/O.

**Files:**
- Create: `examples/juniata/extract_bundle.py`
- Create: `tests/examples/__init__.py` (empty)
- Test: `tests/examples/test_reindex.py`

**Interfaces:**
- Produces: `reindex_subgraph(gage_indices_0, gage_indices_1, gage_order, gage_idx, conus_order, conus_length_m, conus_slope) -> dict` with keys `indices_0`, `indices_1` (np.int32, remapped), `order` (np.int32 COMIDs, new positional order), `length_m`, `slope` (np.float subset, aligned to `order`), `gage_idx` (int, remapped), `shape` (list[int, int]). New positional space is **CONUS-positional sort order of the subgraph members** (preserves topological ordering of the lower-triangular system).
- Produces: CLI `uv run python examples/juniata/extract_bundle.py --out examples/juniata/data [--gage 01567000]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/examples/__init__.py` (empty) and `tests/examples/test_reindex.py`:

```python
"""Tests for the Juniata bundle re-indexing — pure functions, no data stores."""

import numpy as np

from examples.juniata.extract_bundle import reindex_subgraph


def _toy_conus() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 6-reach CONUS: order maps position -> COMID
    conus_order = np.array([900, 901, 902, 903, 904, 905], dtype=np.int32)
    length = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    slope = np.array([1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3])
    return conus_order, length, slope


class TestReindexSubgraph:
    def test_remaps_edges_and_gage_idx(self) -> None:
        conus_order, length, slope = _toy_conus()
        # Subgraph = positions {1, 3, 4} (COMIDs 901, 903, 904); edges 1->3, 3->4
        out = reindex_subgraph(
            gage_indices_0=np.array([3, 4], dtype=np.int32),  # downstream (row)
            gage_indices_1=np.array([1, 3], dtype=np.int32),  # upstream (col)
            gage_order=np.array([901, 903, 904], dtype=np.int32),
            gage_idx=4,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        # New space sorted by CONUS position: 901->0, 903->1, 904->2
        np.testing.assert_array_equal(out["order"], [901, 903, 904])
        np.testing.assert_array_equal(out["indices_0"], [1, 2])
        np.testing.assert_array_equal(out["indices_1"], [0, 1])
        assert out["gage_idx"] == 2
        assert out["shape"] == [3, 3]

    def test_physical_arrays_align_to_new_order(self) -> None:
        conus_order, length, slope = _toy_conus()
        out = reindex_subgraph(
            gage_indices_0=np.array([3], dtype=np.int32),
            gage_indices_1=np.array([1], dtype=np.int32),
            gage_order=np.array([901, 903], dtype=np.int32),
            gage_idx=3,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        np.testing.assert_allclose(out["length_m"], [11.0, 13.0])
        np.testing.assert_allclose(out["slope"], [2e-3, 4e-3])

    def test_lower_triangular_preserved(self) -> None:
        # Downstream position must stay > upstream position in the new space —
        # the CSR solve is forward substitution and requires it.
        conus_order, length, slope = _toy_conus()
        out = reindex_subgraph(
            gage_indices_0=np.array([3, 4, 4], dtype=np.int32),
            gage_indices_1=np.array([1, 3, 1], dtype=np.int32),
            gage_order=np.array([901, 903, 904], dtype=np.int32),
            gage_idx=4,
            conus_order=conus_order,
            conus_length_m=length,
            conus_slope=slope,
        )
        assert (out["indices_0"] > out["indices_1"]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/examples/test_reindex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.juniata'` — also create empty `examples/__init__.py` and `examples/juniata/__init__.py` if imports require them (check first: `ls examples/__init__.py`).

- [ ] **Step 3: Implement `extract_bundle.py`**

```python
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
    "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
    "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
]


def reindex_subgraph(
    gage_indices_0: np.ndarray,
    gage_indices_1: np.ndarray,
    gage_order: np.ndarray,
    gage_idx: int,
    conus_order: np.ndarray,
    conus_length_m: np.ndarray,
    conus_slope: np.ndarray,
) -> dict:
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
    dict
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
```

Note on `.sel(divide_id=comids)`: Q' `divide_id` is an integer COMID coordinate; label-selection with the COMID array both subsets and **orders** the store to match `order`, which is what `StreamflowReader.divide_id_to_index` needs. If any COMID is missing from the Q' store, `.sel` raises — that is the correct failure (surface it, don't impute).

Note on the zarr write API: `create_array(key, data=...)` calls above are schematic — the repo's canonical group-writing pattern is `engine/core/zarr_io.py` (it wrote these exact stores). Match its calls (`create_dataset`/`create_array` per the locked zarr version) so dtypes and layout come out identical; the Step-3 store-verification gate in Task 2 is the arbiter.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/examples/test_reindex.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/juniata/extract_bundle.py examples/__init__.py examples/juniata/__init__.py tests/examples/
git commit -m "feat(examples): Juniata bundle extraction with compact re-indexing"
```

---

### Task 2: Produce, verify, and commit the bundle

**Files:**
- Create: `examples/juniata/data/` (generated by Task 1's script)
- Test: `tests/examples/test_juniata_bundle.py`
- Modify: `.gitignore` (add `examples/juniata/data/statistics/`)

**Interfaces:**
- Consumes: `extract_bundle.py` CLI from Task 1.
- Produces: `BUNDLE = Path("examples/juniata/data")` layout exactly as in File structure; `needs_bundle` skip marker used by Task 3's tests.

- [ ] **Step 1: Write the bundle-contract test**

Create `tests/examples/test_juniata_bundle.py`:

```python
"""Bundle contract: the real library readers open the Juniata bundle."""

from pathlib import Path

import numpy as np
import pytest
import torch

BUNDLE = Path(__file__).parents[2] / "examples" / "juniata" / "data"
needs_bundle = pytest.mark.skipif(
    not (BUNDLE / "juniata_gage.csv").exists(),
    reason="Juniata bundle not present — run examples/juniata/extract_bundle.py",
)


@needs_bundle
class TestBundleContract:
    def test_merit_builds_routing_dataclass(self) -> None:
        from examples.juniata.train_and_test import make_config
        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn([g for g in dataset.gage_ids])
        n = batch.adjacency_matrix.shape[0]
        assert n == 213
        assert batch.length.shape == (n,)
        assert batch.slope.shape == (n,)
        assert batch.normalized_spatial_attributes.shape == (n, 10)
        assert torch.isfinite(batch.normalized_spatial_attributes).all()
        # Gauge prediction extracts the gauge reach itself (post-#192 semantics)
        assert len(batch.outflow_idx) == 1 and batch.outflow_idx[0].shape == (1,)

    def test_streamflow_reader_returns_hourly_tensor(self) -> None:
        from examples.juniata.train_and_test import make_config
        from ddr import streamflow
        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn([g for g in dataset.gage_ids])
        flow = streamflow(cfg)
        q_prime = flow(routing_dataclass=batch)
        assert q_prime.shape[1] == 213
        assert q_prime.shape[0] == len(batch.dates.batch_hourly_time_range)
        assert (q_prime >= 0).all() and torch.isfinite(q_prime).all()

    def test_observations_aligned(self) -> None:
        from examples.juniata.train_and_test import make_config
        from ddr.validation.enums import GeoDataset

        cfg = make_config(bundle_dir=BUNDLE)
        dataset = GeoDataset.get_dataset_class(cfg=cfg)
        batch = dataset.collate_fn([g for g in dataset.gage_ids])
        obs = batch.observations.streamflow.values
        assert obs.shape[0] == 1
        assert np.isfinite(obs).mean() > 0.9  # Juniata record is nearly complete
```

(This test also consumes Task 3's `make_config`; until Task 3 lands it fails on import — acceptable ordering, noted here so Task 2's run uses only Steps 2–4 checks and the full test passes at the end of Task 3.)

- [ ] **Step 2: Run the extraction**

Run: `uv run python examples/juniata/extract_bundle.py --out examples/juniata/data`
Expected output: `reaches: 213, edges: 212, gage_idx: <int in 0..212>`, q' time `1980-01-01 .. 2010-12-31`, bundle size printed.

- [ ] **Step 3: Verify readers open the stores directly**

Run:

```bash
uv run python - << 'EOF'
from ddr.io.readers import read_ic, read_zarr
from pathlib import Path
b = Path("examples/juniata/data")
q = read_ic(str(b / "juniata_qprime.ic")); assert dict(q.sizes)["divide_id"] == 213, q.sizes
o = read_ic(str(b / "juniata_obs.ic")); assert dict(o.sizes)["gage_id"] == 1, o.sizes
import pandas as pd
assert str(q.time.values[0])[:10] == "1980-01-01" and str(o.time.values[0])[:10] == "1980-01-01"
g = read_zarr(b / "juniata_gages_adjacency.zarr"); assert "01567000" in g
print("stores OK")
EOF
```

Expected: `stores OK`.

- [ ] **Step 4: Size gate and commit decision**

Run: `du -sh examples/juniata/data`
If ≤ 50 MB: add `examples/juniata/data/statistics/` to `.gitignore`, then `git add examples/juniata/data .gitignore tests/examples/test_juniata_bundle.py` and commit. If > 50 MB: do NOT add `data/`; instead `tar czf` the bundle, stop, and report to the user for the release-asset path (spec §1) before continuing.

- [ ] **Step 5: Commit**

```bash
git commit -m "data(examples): Juniata 01567000 bundle — 213-reach subgraph, 1980-2010"
```

---

### Task 3: Train/test module

**Files:**
- Create: `examples/juniata/train_and_test.py`
- Test: `tests/examples/test_juniata_bundle.py` (Task 2 file — now fully passing) plus the smoke test below appended to it.

**Interfaces:**
- Consumes: bundle layout from Task 2.
- Produces:
  - `make_config(bundle_dir: Path, device: str = "cpu", epochs: int = 30, rho: int = 90, train_period=("1981/10/01","1995/09/30"), test_period=("1995/10/01","2010/09/30")) -> Config`
  - `train(cfg: Config) -> Path` (last checkpoint path)
  - `test(cfg: Config, checkpoint: Path) -> xr.Dataset` (vars `predictions`, `observations`; attrs `nse`, `kge`, `rmse` medians)
  - `summed_qprime_baseline(cfg: Config) -> xr.Dataset` (var `predictions`, same day axis)
  - CLI `uv run python -m examples.juniata.train_and_test --bundle examples/juniata/data [--device cpu] [--epochs 30]`

- [ ] **Step 1: Write the module**

```python
"""Single-catchment DDR on the Juniata bundle — no Hydra, every knob visible.

train() / test() mirror scripts/train.py and scripts/test.py at batch_size=1.
The `q_prime = flow(...)` line is the seam for the future differentiable
runoff model: any module returning the same gradient-capable hourly
(num_timesteps, num_divides) m³/s tensor can replace the icechunk reader,
and gradients then flow end-to-end past the routing into runoff parameters.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ddr import dmc, kan, streamflow
from ddr.scripts_utils import resolve_learning_rate, tau_trim_and_downsample
from ddr.validation import Config, Metrics
from ddr.validation.enums import GeoDataset, Mode

log = logging.getLogger(__name__)


def make_config(
    bundle_dir: Path,
    device: str = "cpu",
    epochs: int = 30,
    rho: int = 90,
    train_period: tuple[str, str] = ("1981/10/01", "1995/09/30"),
    test_period: tuple[str, str] = ("1995/10/01", "2010/09/30"),
) -> Config:
    """Build the same Pydantic Config the main scripts use, in plain code.

    Single gauge -> one mini-batch (one random rho-day window) per epoch, so
    `epochs` is the optimizer-step count. 30 steps demonstrates learning; it
    is not a converged model (see README).
    """
    bundle_dir = Path(bundle_dir)
    return Config(
        name="juniata-sample",
        geodataset="merit",
        mode=Mode.TRAINING,
        device=device,
        seed=42,
        np_seed=42,
        data_sources={
            "attributes": str(bundle_dir / "juniata_attributes.nc"),
            "conus_adjacency": bundle_dir / "juniata_conus_adjacency.zarr",
            "gages_adjacency": str(bundle_dir / "juniata_gages_adjacency.zarr"),
            "streamflow": str(bundle_dir / "juniata_qprime.ic"),
            "observations": str(bundle_dir / "juniata_obs.ic"),
            "gages": str(bundle_dir / "juniata_gage.csv"),
            "statistics": bundle_dir / "statistics",
        },
        params={"save_path": str(bundle_dir.parent / "runs")},
        experiment={
            "batch_size": 1,
            "start_time": train_period[0],
            "end_time": train_period[1],
            "epochs": epochs,
            "rho": rho,
            "shuffle": True,
            "warmup": 5,
            "learning_rate": {1: 1e-3, max(2, epochs // 2): 5e-4},
        },
        kan={
            "hidden_size": 21,
            "num_hidden_layers": 2,
            "grid": 50,
            "k": 2,
            "input_var_names": [
                "SoilGrids1km_clay", "aridity", "meanelevation", "meanP", "NDVI",
                "meanslope", "log10_uparea", "SoilGrids1km_sand", "ETPOT_Hargr", "Porosity",
            ],
            "learnable_parameters": ["n", "q_spatial", "p_spatial"],
        },
        # `test_period` is threaded through by test() below (Dates rebuild).
    )


def _models(cfg: Config) -> tuple[kan, dmc, streamflow]:
    nn = kan(
        input_var_names=cfg.kan.input_var_names,
        learnable_parameters=cfg.kan.learnable_parameters,
        hidden_size=cfg.kan.hidden_size,
        num_hidden_layers=cfg.kan.num_hidden_layers,
        grid=cfg.kan.grid,
        k=cfg.kan.k,
        seed=cfg.seed,
        device=cfg.device,
    )
    return nn, dmc(cfg=cfg, device=cfg.device), streamflow(cfg)


def train(cfg: Config) -> Path:
    """Train the KAN on the single Juniata gauge; return last checkpoint path."""
    torch.manual_seed(cfg.seed)
    dataset = GeoDataset.get_dataset_class(cfg=cfg)
    nn, routing_model, flow = _models(cfg)
    lr = resolve_learning_rate(cfg.experiment.learning_rate, 1)
    optimizer = torch.optim.Adam(params=nn.parameters(), lr=lr)
    ckpt_dir = Path(cfg.params.save_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.experiment.epochs + 1):
        for g in optimizer.param_groups:
            g["lr"] = resolve_learning_rate(cfg.experiment.learning_rate, epoch)
        dataset.dates.calculate_time_period()          # random rho-day window
        batch = dataset.collate_fn(list(dataset.gage_ids))

        q_prime = flow(routing_dataclass=batch)        # <-- runoff-model seam
        params = nn(inputs=batch.normalized_spatial_attributes.to(cfg.device))
        out = routing_model(routing_dataclass=batch, spatial_parameters=params, streamflow=q_prime)
        daily = tau_trim_and_downsample(out["runoff"], cfg.params.tau)

        obs = torch.tensor(
            batch.observations.streamflow.values, device=cfg.device, dtype=torch.float32
        )[:, :-2]
        w = cfg.experiment.warmup
        loss = torch.nn.functional.l1_loss(daily[:, w:], obs[:, w:])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.0)
        optimizer.step()
        log.info(f"epoch {epoch}: loss {loss.item():.4f}")

    ckpt = ckpt_dir / f"juniata_epoch_{cfg.experiment.epochs}.pt"
    torch.save({"model_state_dict": nn.state_dict(), "epoch": cfg.experiment.epochs}, ckpt)
    return ckpt


def _eval_dataset_and_batch(cfg: Config, test_period: tuple[str, str]):
    eval_cfg = cfg.model_copy(deep=True)
    eval_cfg.mode = Mode.TESTING
    eval_cfg.experiment.start_time = test_period[0]
    eval_cfg.experiment.end_time = test_period[1]
    eval_cfg.experiment.rho = None
    dataset = GeoDataset.get_dataset_class(cfg=eval_cfg)
    batch = dataset.collate_fn([0])   # inference mode: pre-built dataclass
    return eval_cfg, dataset, batch


def test(
    cfg: Config, checkpoint: Path, test_period: tuple[str, str] = ("1995/10/01", "2010/09/30")
) -> xr.Dataset:
    """Evaluate a checkpoint over the full test period; return preds+obs+metrics."""
    eval_cfg, dataset, batch = _eval_dataset_and_batch(cfg, test_period)
    nn, routing_model, flow = _models(eval_cfg)
    state = torch.load(checkpoint, map_location=eval_cfg.device)
    nn.load_state_dict(state["model_state_dict"])
    nn.eval()
    with torch.no_grad():
        q_prime = flow(routing_dataclass=batch)
        params = nn(inputs=batch.normalized_spatial_attributes.to(eval_cfg.device))
        out = routing_model(routing_dataclass=batch, spatial_parameters=params, streamflow=q_prime)
        daily = tau_trim_and_downsample(out["runoff"], eval_cfg.params.tau).cpu().numpy()

    obs = batch.observations.streamflow.values[:, :-2]
    time = dataset.dates.daily_time_range[:-2]
    metrics = Metrics(pred=daily, target=obs)
    ds = xr.Dataset(
        {
            "predictions": (("gage", "time"), daily),
            "observations": (("gage", "time"), obs),
        },
        coords={"gage": ["01567000"], "time": time},
        attrs={
            "nse": float(np.nanmedian(metrics.nse)),
            "kge": float(np.nanmedian(metrics.kge)),
            "rmse": float(np.nanmedian(metrics.rmse)),
        },
    )
    return ds


def summed_qprime_baseline(
    cfg: Config, test_period: tuple[str, str] = ("1995/10/01", "2010/09/30")
) -> xr.Dataset:
    """No-routing baseline: daily sum of all upstream lateral inflows."""
    eval_cfg, dataset, batch = _eval_dataset_and_batch(cfg, test_period)
    flow = streamflow(eval_cfg)
    with torch.no_grad():
        q_prime = flow(routing_dataclass=batch)              # hourly (T, N)
        total = q_prime.sum(dim=1, keepdim=True).T           # (1, T)
        daily = tau_trim_and_downsample(total, tau=0).cpu().numpy()  # day-aligned
    time = dataset.dates.daily_time_range[:-2]
    obs = batch.observations.streamflow.values[:, :-2]
    metrics = Metrics(pred=daily, target=obs)
    return xr.Dataset(
        {"predictions": (("gage", "time"), daily)},
        coords={"gage": ["01567000"], "time": time},
        attrs={"nse": float(np.nanmedian(metrics.nse)), "kge": float(np.nanmedian(metrics.kge))},
    )


def main() -> None:
    """CLI: train, test, baseline, print the comparison table."""
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", type=Path, default=Path("examples/juniata/data"))
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    cfg = make_config(bundle_dir=a.bundle, device=a.device, epochs=a.epochs)
    ckpt = train(cfg)
    result = test(cfg, ckpt)
    baseline = summed_qprime_baseline(cfg)
    print(f"{'':>12} {'NSE':>8} {'KGE':>8}")
    print(f"{'routed':>12} {result.attrs['nse']:8.3f} {result.attrs['kge']:8.3f}")
    print(f"{'summed q\'':>12} {baseline.attrs['nse']:8.3f} {baseline.attrs['kge']:8.3f}")


if __name__ == "__main__":
    main()
```

Implementation notes the engineer must honor:
- `dataset.collate_fn` signatures differ between training mode (list of gage IDs) and inference mode (returns the pre-built dataclass; argument ignored) — mirror `scripts/test.py` if `[0]` mis-fires and check `BaseGeoDataset.collate_fn`.
- Obs pairing `[:, :-2]` and `tau_trim_and_downsample` follow the signed-tau convention (day i ↔ obs day i).
- The baseline uses `tau=0` (day-aligned sum, matching `summed_q_prime.py` semantics).
- If `Config`/`Dates` reject any field here (e.g., `learning_rate` key type), align with `tests/validation/test_configs.py` fixtures rather than changing the library.

- [ ] **Step 2: Append the smoke test**

Append to `tests/examples/test_juniata_bundle.py`:

```python
@needs_bundle
class TestSmoke:
    def test_one_epoch_train_and_metrics(self, tmp_path: Path) -> None:
        from examples.juniata.train_and_test import make_config, summed_qprime_baseline, test, train

        cfg = make_config(bundle_dir=BUNDLE, epochs=1, rho=30)
        cfg.params.save_path = tmp_path
        ckpt = train(cfg)
        assert ckpt.exists()
        result = test(cfg, ckpt, test_period=("1996/10/01", "1997/09/30"))
        baseline = summed_qprime_baseline(cfg, test_period=("1996/10/01", "1997/09/30"))
        assert np.isfinite(result.attrs["nse"])
        assert np.isfinite(baseline.attrs["nse"])
        assert result.predictions.shape == result.observations.shape
```

- [ ] **Step 3: Run the bundle + smoke tests**

Run: `uv run pytest tests/examples/ -v`
Expected: re-index tests PASS; bundle-contract tests PASS; smoke PASS (≈1–3 min on CPU: 213 reaches, 30-day window + 1-year eval).

- [ ] **Step 4: Run the CLI end-to-end**

Run: `uv run python -m examples.juniata.train_and_test --bundle examples/juniata/data --epochs 30`
Expected: 30 epoch losses trending down; final table prints routed and summed-Q' NSE/KGE (routed should be in the same range as or above the baseline; no hard assert — record the numbers in the README in Task 4).

- [ ] **Step 5: Commit**

```bash
git add examples/juniata/train_and_test.py tests/examples/test_juniata_bundle.py
git commit -m "feat(examples): Juniata plain-Python train/test module with summed-Q' baseline"
```

---

### Task 4: Notebook, README, wiki

**Files:**
- Create: `examples/juniata/make_notebook.py`, `examples/juniata/juniata_routing.ipynb` (generated), `examples/juniata/README.md`
- Modify: `wiki/examples.md`, `wiki/log.md`

**Interfaces:**
- Consumes: module functions from Task 3, bundle from Task 2.

- [ ] **Step 1: Write `make_notebook.py`**

Builder pattern: a list of `(cell_type, source)` tuples passed to nbformat. Full cell list (markdown sources abridged to their opening lines here ONLY where the remaining prose is the engineer's to write from the spec's §3 section outline — every code cell is complete):

```python
"""Generate juniata_routing.ipynb. Run: uv run --with nbformat python examples/juniata/make_notebook.py"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# From dMC-Juniata to DDR: differentiable routing on one catchment

The Juniata River at Newport, PA (USGS 01567000; 8,657 km²; 213 MERIT reaches)
— the basin where differentiable Muskingum-Cunge routing started. This notebook
walks the full physics and training chain on a laptop-sized bundle.

**Contents** — 1. The basin · 2. Muskingum-Cunge physics · 3. The network solve
· 4. Why differentiable · 5. Train & evaluate · 6. The road to end-to-end.""")

# --- 1. The basin ---
code("""from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.juniata.train_and_test import make_config
from ddr.validation.enums import GeoDataset

BUNDLE = Path("data")
cfg = make_config(bundle_dir=BUNDLE)
dataset = GeoDataset.get_dataset_class(cfg=cfg)
batch = dataset.collate_fn(list(dataset.gage_ids))
N = batch.adjacency_matrix.shape[0]
print(f"{N} reaches, gauge reach index {batch.outflow_idx[0][0]}")""")
code("""import rustworkx as rx
from rustworkx.visualization import mpl_draw

adj = batch.adjacency_matrix.to_dense().numpy()
g = rx.PyDiGraph()
g.add_nodes_from(range(N))
rows, cols = np.nonzero(adj)
g.add_edges_from_no_data([(int(c), int(r)) for r, c in zip(rows, cols)])
fig, ax = plt.subplots(figsize=(10, 7))
mpl_draw(g, ax=ax, node_size=12, arrow_size=4)
ax.set_title("Juniata reach network (edges point downstream)")""")
code("""obs = batch.observations.streamflow.values[0]
t = dataset.dates.daily_time_range
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t[: len(obs)], obs, lw=0.5)
ax.set_ylabel("Q (m³/s)"); ax.set_title("Observed discharge at Newport")""")
md("(§1 prose: why 5–10k km² is where routing physically acts — cite the "
   "area-stratified ddrs table: +0.135 median NSE at 5–10k km².)")

# --- 2. Physics ---
md(r"""## 2. Muskingum-Cunge physics

Storage routing from continuity $\frac{dS}{dt} = I - Q$ with
$S = K[XI + (1-X)Q]$ gives the update
$Q_{t+1} = c_1 I_{t+1} + c_2 I_t + c_3 Q_t + c_4 q'$ with

$$c_1 = \frac{\Delta t - 2KX}{D},\; c_2 = \frac{\Delta t + 2KX}{D},\;
c_3 = \frac{2K(1-X) - \Delta t}{D},\; c_4 = \frac{2\Delta t}{D},\;
D = 2K(1-X) + \Delta t$$

$c_1 + c_2 + c_3 = 1$ **exactly** — mass conservation holds for any $(K, X)$.
$K = L/c$ is the reach travel time; everything hinges on celerity $c$ and $X$.""")
code("""# Verify the mass identity on the actual implementation
from ddr.routing.mmc import MuskingumCunge
mc = MuskingumCunge(cfg, device="cpu")
c1, c2, c3, c4 = mc.calculate_muskingum_coefficients(
    length=torch.tensor([5000.0]), celerity=torch.tensor([1.2]), x=torch.tensor([0.4]))
print(c1 + c2 + c3)  # tensor([1.])""")
md(r"""### Trapezoid-exact celerity

Kinematic celerity is $c = dQ/dA$. For the trapezoid DDR builds
(Leopold & Maddock: $T = p\,d^{\,q}$), $c = v\,\beta$ with
$\beta = \frac{5}{3} - \frac{4}{3}\frac{A\sqrt{1+z^2}}{T\,P}$ — the classic
$5/3$ is the wide-rectangular limit and runs 22–27% high on real channels.""")
code("""def beta(b, y, z):
    T = b + 2 * z * y
    A = (b + T) * y / 2
    P = b + 2 * y * np.sqrt(1 + z**2)
    return 5 / 3 - (4 / 3) * A * np.sqrt(1 + z**2) / (T * P)

by = np.logspace(-2, 3, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for z in [0.0, 1.0, 2.0]:
    ax.semilogx(by, beta(by, 1.0, z), label=f"z={z}")
ax.axhline(5 / 3, ls="--", c="k", lw=0.7); ax.axhline(4 / 3, ls=":", c="gray", lw=0.7)
ax.set_xlabel("b / y"); ax.set_ylabel("β"); ax.legend()
ax.set_title("β is non-monotone in b/y and NOT bounded below by 4/3")""")
md(r"""### Cunge X: matching numerical to physical diffusion

The scheme's numerical diffusion is $D_{num} = cL(0.5 - X)$; the channel's
physical diffusivity is $D_{phys} = Q/(2TS)$. Setting them equal:
$X = \mathrm{clamp}\!\left(0.5\left(1 - \frac{Q}{T\,S\,c\,L}\right), 0, 0.5\right)$.
The legacy constant $X = 0.3$ traded diffusion accuracy for a wide stability
window $2X \le C_r \le 2(1-X)$ — DDR now computes $X$ per reach per timestep.""")

# --- 3. Network solve ---
md(r"""## 3. The network solve

Per timestep DDR solves $(I - c_1 N)\,Q_{t+1} = c_2 N Q_t + c_3 Q_t + c_4 q'$.
$N$ is the downstream adjacency; topological ordering makes $(I - c_1 N)$
**lower triangular**, so the solve is a single forward substitution.""")
code("""fig, ax = plt.subplots(figsize=(5, 5))
ax.spy(np.eye(N) + adj, markersize=1)
ax.set_title("(I − c₁N) sparsity — lower triangular")""")

# --- 4. Why differentiable ---
md("""## 4. Why differentiable

The KAN maps catchment attributes → {n, q_spatial, p_spatial} ∈ [0,1] →
physical bounds. The loss differentiates through the solve, the coefficients,
Cunge X, and β back into the KAN weights — one autograd chain.""")
code("""from ddr import dmc, kan, streamflow

nn = kan(input_var_names=cfg.kan.input_var_names,
         learnable_parameters=cfg.kan.learnable_parameters,
         hidden_size=cfg.kan.hidden_size, num_hidden_layers=cfg.kan.num_hidden_layers,
         grid=cfg.kan.grid, k=cfg.kan.k, seed=cfg.seed, device="cpu")
routing = dmc(cfg=cfg, device="cpu")
flow = streamflow(cfg)

dataset.dates.calculate_time_period()
b = dataset.collate_fn(list(dataset.gage_ids))
q_prime = flow(routing_dataclass=b)          # <-- future runoff model plugs in HERE
params = nn(inputs=b.normalized_spatial_attributes)
out = routing(routing_dataclass=b, spatial_parameters=params, streamflow=q_prime)
loss = out["runoff"].mean()
loss.backward()
g = [p.grad.abs().mean().item() for p in nn.parameters() if p.grad is not None]
print(f"{len(g)} KAN tensors received gradients; mean |grad| {np.mean(g):.2e}")""")

# --- 5. Train & evaluate ---
code("""from examples.juniata.train_and_test import summed_qprime_baseline, test, train

cfg = make_config(bundle_dir=BUNDLE, epochs=1)   # fast mode; raise to 30 for the README numbers
ckpt = train(cfg)
result = test(cfg, ckpt)
baseline = summed_qprime_baseline(cfg)
print(f"routed  NSE {result.attrs['nse']:.3f}  KGE {result.attrs['kge']:.3f}")
print(f"summed  NSE {baseline.attrs['nse']:.3f}  KGE {baseline.attrs['kge']:.3f}")""")
code("""fig, ax = plt.subplots(figsize=(12, 4))
sl = slice(0, 730)
ax.plot(result.time[sl], result.observations[0, sl], "k", lw=0.8, label="observed")
ax.plot(result.time[sl], result.predictions[0, sl], "C0", lw=0.8, label="DDR routed")
ax.plot(baseline.time[sl], baseline.predictions[0, sl], "C1", lw=0.6, ls="--", label="summed q'")
ax.legend(); ax.set_ylabel("Q (m³/s)")""")

# --- 6. Road to end-to-end ---
md("""## 6. The road to end-to-end

Q' entered this notebook through `flow(routing_dataclass=...)` — a reader of
precomputed runoff. The contract for replacing it with a **differentiable
runoff model** is exactly: return an hourly `(num_timesteps, num_divides)`
float32 tensor in m³/s that carries `requires_grad`. Then `loss.backward()`
reaches the runoff model's parameters through the routing physics — the full
end-to-end gradient chain. That toy model is the next project.""")

nb["cells"] = cells
nbf.write(nb, "examples/juniata/juniata_routing.ipynb")
print("wrote examples/juniata/juniata_routing.ipynb")
```

- [ ] **Step 2: Generate and execute the notebook**

```bash
uv run --with nbformat python examples/juniata/make_notebook.py
uv run --with nbclient --with nbformat --with matplotlib python -c "
import nbformat
from nbclient import NotebookClient
nb = nbformat.read('examples/juniata/juniata_routing.ipynb', as_version=4)
NotebookClient(nb, timeout=1200, kernel_name='python3').execute()
print('notebook executes clean')
"
```

Expected: `notebook executes clean` (execution from repo root; the notebook's `BUNDLE = Path("data")` assumes CWD `examples/juniata/` — set `resources={'metadata': {'path': 'examples/juniata'}}` in the NotebookClient call). Fix any cell that errors before proceeding.

- [ ] **Step 3: Write README.md**

`examples/juniata/README.md` content (complete):

```markdown
# Juniata single-catchment sample

DDR on one basin: the Juniata River at Newport, PA (USGS 01567000,
8,657 km², 213 MERIT reaches). Everything needed is in `data/` — no HPC,
S3, or external stores.

## Quickstart

    git clone <repo> && cd ddr
    uv sync --all-packages
    uv run python -m examples.juniata.train_and_test --bundle examples/juniata/data
    # then open examples/juniata/juniata_routing.ipynb

Training is ~30 optimizer steps (one random 90-day window per epoch on a
single gauge): it demonstrates learning and physically plausible parameter
fields, not a converged CONUS-grade model. Reference run (CPU, 30 epochs):
routed NSE <fill from Task 3 Step 4>, summed-q' baseline NSE <fill>.

## What's in the bundle

| File | Contents |
|---|---|
| `juniata_qprime.ic` | icechunk, `Qr(divide_id, time)` daily m³/s, 213 divides, 1980–2010 (dHBV2 UH retrospective) |
| `juniata_obs.ic` | icechunk, `streamflow(gage_id, time)` daily m³/s, USGS 01567000, 1980–2010 |
| `juniata_attributes.nc` | 10 KAN input attributes per COMID |
| `juniata_conus_adjacency.zarr` | binsparse COO subgraph + `length_m`, `slope`, `order` (compact 0..212 indexing) |
| `juniata_gages_adjacency.zarr` | single-gage COO group, same schema as the CONUS store |
| `juniata_gage.csv` | one-row gage metadata (gages_3000 schema) |

Normalization statistics are computed from these 213 catchments on first
run (basin-local z-scores) into `data/statistics/` (gitignored).

The bundle was written with the repo-locked icechunk (2.0.3) — install via
`uv sync`, not a standalone icechunk.

Regenerate (maintainer, needs the CONUS stores):
`uv run python examples/juniata/extract_bundle.py --out examples/juniata/data`

## Toward end-to-end gradients

`train_and_test.py` marks the seam: `q_prime = flow(routing_dataclass=batch)`.
A differentiable runoff model replacing `flow` must return an hourly
`(num_timesteps, num_divides)` float32 m³/s tensor with gradients attached —
nothing else changes.
```

Fill both `<fill …>` slots with the Task 3 Step 4 numbers before committing (a `<fill>` left in the committed README is a task failure).

- [ ] **Step 4: Update wiki**

Add to `wiki/examples.md` notebook table: `| examples/juniata/ | Self-contained single-catchment sample: bundle + plain-Python train/test + physics notebook |`. Append to `wiki/log.md`:
`## [<today>] create | examples/juniata single-catchment sample` with a two-sentence summary.

- [ ] **Step 5: Full suite + commit**

```bash
uv run pytest -q          # everything green, examples tests included
git add examples/juniata/ wiki/examples.md wiki/log.md
git commit -m "feat(examples): Juniata physics notebook, README, wiki entry"
```

---

## Verification after all tasks

1. `uv run pytest -q` green; `uv run pytest tests/examples/ -v` shows re-index + bundle + smoke all PASS (none skipped, since the bundle is present).
2. Fresh-eyes portability check: `git stash -u && git stash pop` is not sufficient — instead verify no absolute paths outside `/mnt/ssd1` defaults leaked into the module or notebook (`grep -rn "/mnt/ssd1\|/home/tbindas" examples/juniata/train_and_test.py examples/juniata/juniata_routing.ipynb` → only `extract_bundle.py` may reference them).
3. `du -sh examples/juniata/data` recorded in the PR description with the 50 MB gate result.
4. README reference numbers filled from a real 30-epoch run.
