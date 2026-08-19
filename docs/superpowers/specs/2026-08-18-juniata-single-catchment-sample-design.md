# Juniata single-catchment DDR sample — design

**Date:** 2026-08-18
**Audience:** a collaborating team learning how DDR works on one catchment,
building toward a full end-to-end gradient chain (differentiable runoff model
→ differentiable routing).

## Goal

A self-contained sample under `examples/juniata/` featuring the Juniata River
at Newport, PA (USGS 01567000, COMID 73005278, 8,657 km² — inside the
5,000–10,000 km² band where routing physically acts, and the dMC-Juniata
heritage basin). Three deliverables:

1. a one-time **extraction script** that slices the CONUS stores into a small
   portable data bundle;
2. a small-scale **train-and-test module** that runs real DDR (unmodified
   `Merit` dataset, `StreamflowReader`, `kan`, `dmc`) on the bundle;
3. a **Jupyter notebook** explaining the Muskingum-Cunge physics and the
   differentiable training loop on this basin.

The toy differentiable runoff model (replacing icechunk Q') is the NEXT
project, not this one. This sample documents the seam it will plug into.

## Decisions (settled during brainstorming)

- **Self-contained bundle** — collaborators have no access to
  `/mnt/ssd1` stores, S3, or HPC. Everything needed ships in
  `examples/juniata/data/` (or a release tarball; see Bundle).
- **Reuse the real library** — the bundle mimics the CONUS store layouts so
  the unmodified `Merit` dataset and readers run on it. No parallel
  "example" dataset class that would drift from the library.
- **Seam only, no abstraction** — `q_prime = flow(routing_dataclass)` in the
  train loop is the documented socket for the future runoff model. No
  `QPrimeProvider` protocol is introduced yet (YAGNI); the notebook and code
  comments state the contract: a `(num_timesteps, num_divides)` hourly
  tensor, gradient-capable, in m³/s.
- **Q' source: UH retrospective** (`merit_dhbv2_UH_retrospective.ic`) —
  matches the repo's canonical baseline config so sample results are
  comparable to the full-CONUS baseline run.

## 1. Data bundle

`examples/juniata/extract_bundle.py` — run once by the maintainer on the
machine that has the CONUS stores; not needed by collaborators. CLI:
`uv run python examples/juniata/extract_bundle.py --out examples/juniata/data`
with `--gage 01567000` default.

Steps:

1. Read the `01567000` group from `data/merit_gages_conus_adjacency.zarr`
   (per-gage COO subgraph, binsparse v3 layout) to get the upstream COMID
   set.
2. **`juniata_gages_adjacency.zarr`** — copy of that single gage group,
   same layout.
3. **`juniata_conus_adjacency.zarr`** — subset of
   `data/merit_conus_adjacency.zarr` (`indices_0`, `indices_1`, `values`,
   `order`, `length_m`, `slope`) restricted to the subgraph COMIDs.
   **Index contract risk:** `Merit`/`construct_network_matrix` may assume
   CONUS-wide index spaces. Implementation must verify how the per-gage COO
   in `gages_adjacency` references `conus_adjacency` (positional index vs
   COMID) and either re-index both consistently or keep original CONUS
   positional indices in the subset. This is the one place the "unmodified
   library" claim can break — it is a plan-level verification gate, and the
   fallback is preserving original index values with a sparse `order` array.
4. **`juniata_qprime.ic`** — local icechunk store, variable `Qr` with dims
   `(divide_id, time)`, sliced from `merit_dhbv2_UH_retrospective.ic` for
   the subgraph divides, daily, 1981-10-01 – 2010-09-30.
5. **`juniata_obs.ic`** — local icechunk store, same layout as
   `usgs_daily_observations`, gauge 01567000 only, same period.
6. **`juniata_attributes.nc`** — `merit_global_attributes_v2.nc` subset to
   the subgraph COMIDs and the 10 KAN input variables (same variable names
   and `COMID` coordinate).
7. **`juniata_gage.csv`** — the 01567000 row of `gages_3000.csv`, same
   header.
8. Print bundle size and a manifest.

Statistics (`set_statistics`) are computed on first dataset construction
from the bundle attributes — no statistics files ship.

**Size budget:** expected tens of MB (≈200–300 divides × 10,600 days ×
float32 ≈ 10–25 MB for Q' dominant). If the bundle exceeds **50 MB**, it is
NOT committed; instead `extract_bundle.py` output is uploaded as a GitHub
release asset and `examples/juniata/README.md` documents
`curl -L <release-url> | tar x`. Under 50 MB, it commits directly to
`examples/juniata/data/`.

**Pinned dependency note:** the bundle's icechunk stores are written by the
locked icechunk version; `examples/juniata/README.md` records that version
and instructs collaborators to `uv sync` from the repo lockfile rather than
installing icechunk independently.

## 2. Train-and-test module

`examples/juniata/train_and_test.py` — plain Python, no Hydra. The full
configuration is constructed visibly in code so a newcomer can read every
knob:

```python
def make_config(
    bundle_dir: Path,
    device: str = "cpu",
    epochs: int = 5,
    rho: int = 90,
    train_period: tuple[str, str] = ("1981/10/01", "1995/09/30"),
    test_period: tuple[str, str] = ("1995/10/01", "2010/09/30"),
) -> Config: ...
```

returning the same Pydantic `Config` the main scripts use (geodataset merit,
mode set per phase, batch_size 1, warmup 5, tau default 9, learning-rate
schedule `{1: 1e-3, 3: 5e-4}`, KAN identical to
`config/merit_training_config.yaml`).

```python
def train(cfg: Config) -> Path: ...  # returns last checkpoint path
def test(cfg: Config, checkpoint: Path) -> xr.Dataset: ...
def summed_qprime_baseline(cfg: Config) -> xr.Dataset: ...
```

- `train` mirrors `scripts/train.py`'s loop at batch_size=1 (single gauge):
  random rho-day windows, `tau_trim_and_downsample`, L1 loss with warmup
  skip, grad clip 1.0, checkpoints under `bundle_dir.parent / "runs/"`.
- `test` mirrors `scripts/test.py`: sequential full test period,
  `torch.no_grad()`, returns an `xr.Dataset` with `predictions`,
  `observations` and attaches `Metrics` values; obs pairing `[:, :-2]` per
  the signed-tau convention.
- `summed_qprime_baseline` sums the subgraph's Q' (no routing) to daily for
  the routing-gain comparison — small enough here to compute directly, no
  CuPy path.
- `python -m examples.juniata.train_and_test --bundle examples/juniata/data`
  runs train then test then baseline and prints the NSE/KGE table.
- The `q_prime = flow(routing_dataclass)` line carries the seam comment:
  the future differentiable runoff model replaces `flow` with a module
  returning the same gradient-capable hourly `(time, divides)` tensor.

Expectation management (README + notebook): single-gauge training is
~36 windows × 5 epochs ≈ 180 optimizer steps — enough to *demonstrate*
learning and beat/approach the summed-Q' baseline, not to produce
publication skill. The pedagogical target is a visibly improving hydrograph
and physically plausible n/q_spatial fields.

## 3. Notebook

`examples/juniata/juniata_routing.ipynb` — narrative "from dMC-Juniata to
DDR", targeted at hydrologists with DL background (full derivations, no
hand-waving). Sections:

1. **The basin** — load the bundle, plot the reach network (adjacency as a
   graph), drainage area, the obs hydrograph; why 5–10k km² is where routing
   matters (area-stratified evidence from the ddrs findings).
2. **Muskingum-Cunge physics** — storage routing from the continuity
   equation; K and X; the four coefficients with the exact
   `c1 + c2 + c3 = 1` mass identity; trapezoidal geometry from Leopold &
   Maddock (`top_width = p·depth^q`); Manning velocity; **trapezoid-exact
   celerity** `c = v·β`, `β = 5/3 − (4/3)·A·√(1+z²)/(T·P)` with the β-vs-b/y
   curve and the wide-rectangular 5/3 limit; **Cunge X**
   `X = clamp(0.5·(1 − Q/(T·S·c·L)), 0, 0.5)` as numerical-diffusion
   matching, contrasted with the legacy constant X = 0.3 and its stability
   trade (window `2X ≤ Cr ≤ 2(1−X)`).
3. **The network solve** — `(I − c1·N)Q_{t+1} = c2·N·Q_t + c3·Q_t + c4·q'`;
   show Juniata's actual lower-triangular sparse matrix pattern and one
   timestep's forward substitution.
4. **Why differentiable** — KAN attributes→parameters map, denormalization
   to physical bounds, loss → autograd; a live demo: one forward + backward,
   inspect gradients on KAN weights and confirm the chain
   loss → routing → celerity/X → parameters.
5. **Train & evaluate** — call the module's `train`/`test`/baseline; plot
   learned Manning's n and q_spatial per reach, hydrographs
   (pred vs obs vs summed-Q'), NSE/KGE table.
6. **The road end-to-end** — where Q' enters (`flow(...)`), the tensor
   contract the toy runoff model must satisfy, and what "full gradient
   chain" adds (gradients flowing past routing into runoff-model
   parameters).

Notebook executes top-to-bottom on the bundle on CPU in minutes (training
cell may be flagged `epochs=1` fast mode with a note to raise it).

## 4. Testing & docs

- `tests/examples/test_juniata_bundle.py` — bundle contract: readers open
  each piece, `Merit` (training mode) constructs a valid `RoutingDataclass`
  (adjacency square, attribute rows = 10, obs aligned). Skipped with a
  clear message when `examples/juniata/data/` is absent, so CI stays green
  without the bundle; the extraction itself gets an
  `@pytest.mark.integration` test on the maintainer machine.
- `examples/juniata/README.md` — fresh-machine path: clone → `uv sync` →
  (fetch bundle if tarballed) → run module → open notebook. Records the
  icechunk version pin and the expectation-management note.
- `wiki/examples.md` gets the new entry; `wiki/log.md` appended.
- Notebook committed with outputs stripped (nbstripout pre-commit already
  enforces this).

## Out of scope

- The toy differentiable runoff model and forcing-data extraction (next
  project; will consume the seam documented here).
- Lynker-hydrofabric variant of the sample.
- Any change to `src/ddr/` — if the index-contract verification (Bundle
  step 3) reveals a genuine blocker inside `Merit`, that becomes its own
  minimal, separately-reviewed change rather than being folded silently
  into this sample.

## Risks

- **Index contract (Bundle step 3)** — highest uncertainty; verified first
  in the implementation plan before any other work.
- **Icechunk local-store writability** — assumed fine (repo already carries
  local `.ic` stores); if a subset write proves awkward, fallback is zarr
  with the same `Qr(divide_id, time)` schema plus a thin reader shim — but
  that violates "unmodified library" and would be surfaced for approval
  first.
- **Bundle licensing/size** — USGS obs and MERIT-derived data are
  redistributable; size gate at 50 MB decides commit vs release asset.
