# Port ddrs corrections into DDR — design

**Date:** 2026-08-17
**Source of truth:** `~/projects/ddrs` — `.claude/PHYSICS-CORRECTIONS.md`,
`docs/2026-08-03-ddr-match-findings.md`, `docs/2026-08-06-tau-sweep-pilot-findings.md` §5i.

## Background

The Rust reimplementation (`ddrs`) gated a set of corrections behind
`params.ddr_match: bool` (`true` = bit-for-bit DDR parity, `false` = corrected).
Its production configs all run `ddr_match: false`. The tau=9 retrain
(2026-08-09, 1,841 gauges) reached median NSE 0.706–0.707, beating the
summed-Q' baseline (0.642); the shipped legacy configuration loses to it
(0.620). This spec ports the corrections into DDR as a **clean break**: no
compatibility flag, corrected behavior is the only path. Reproduction of old
results = old commit hashes. All pre-port checkpoints (legacy physics, legacy
tau) are invalidated.

## Scope — four staged changes, landed in order

Each stage is an independent, bisectable commit/PR with its own tests, so a
retrain after stage 3 and after stage 4 attributes the tau gain separately
from the physics gain (the matched control ddrs never ran).

### Stage 1 — `outflow_idx` mass-conservation fix (merit.py)

The MERIT collate builds a gauge's prediction by summing the gauge reach's
*upstream* columns, dropping the gauge reach's own local drainage (worst case
0.215× observed; 26 of 1,841 gauges below 0.5× baseline, all small basins).
The MC solve at the gauge reach already carries all upstream flow plus its own
lateral inflow, so the mass-conserving extraction is the gauge reach itself:

```python
outflow_idx.append(np.array([index_mapping[int(_idx)]]))
```

Apply at **every** `outflow_idx` construction site in `merit.py` (training
collate + inference-mode builders). The Lynker path validates against `toid`
and is untouched.

**Test:** steady-state gauge mass-conservation on a small synthetic network —
constant lateral inflow, gauge prediction converges to total upstream inflow
(mirrors ddrs `tests/gauge_mass_conservation.rs`).

**Verify during implementation:** `build_flow_scale_tensor` receives
`gage_compressed_indices` — confirm the DA-ratio scaling does not assume the
old upstream-sum semantics.

### Stage 2 — negative-solve counter (mmc.py)

`route_timestep` clamps the sparse-solve output (`clamp(solution, min=
discharge_lb)`), silently creating mass where the solution is negative. Count
`(solution < 0)` **before** the clamp, accumulate over the `route()` loop, and
log the per-forward rate. No behavior change. ddrs measured ~0.03–0.14% of
reach-timesteps under corrected physics; this counter is the stage-4 tripwire.

### Stage 3 — tau signed-at-zero convention

Port ddrs's 2026-08-08 convention exactly:

- `compute_daily_runoff` slices `[tau : -(24 - tau)]`. Pooled day *i* is
  scored against **obs day *i*** (tau = hours the routed output is advanced
  before daily scoring; dMC-Juniata's sign; tau=0 is day-aligned).
- Eval drops the **first** pooled day instead of the last, so the output zarr
  day axis is unchanged.
- Runtime tau ∈ [0, 24), **default 9** (was 3). Pydantic guard in
  `validation/configs.py`; config YAMLs updated.
- `train.py`'s inline slice (`[13 : -11+tau]`, currently inconsistent with
  `compute_daily_runoff`'s `[13+tau : -11+tau]`) is replaced by the shared
  helper. Callers updated: `train.py`, `test.py`, `train_and_test.py`,
  `router.py`, benchmarks if applicable.
- Legacy mapping, for the record: shipped tau=3 ≡ signed −8; legacy tau=3 cuts
  the same hour window as new tau=16 under the new pairing.

**Tests:** window arithmetic, legacy-equivalence pin (new 16 ≡ legacy 3 hour
window), day-count invariance.

**Caveat (docs):** tau=9 fits the flagship Q' sources; ddrs measured the
aorc2f-lumped store's optimum near −8. Per-config override is the escape.

### Stage 4 — corrected physics: celerity β + Cunge X

**β (trapezoid-exact celerity).** `c = v·5/3` is the wide-rectangular limit;
the solver builds a trapezoid. In `_get_trapezoid_velocity`:

```
β = 5/3 − (4/3)·A·√(1+z²) / (T·P)      c = clamp(v)·β
```

with `A = cross_sectional_area`, `P = wetted_perimeter`, `T = top_width`,
`z = side_slope` from the geometry dict — the *internal* trapezoid values,
pre-Lynker-override, so celerity stays consistent with the geometry that
produced the velocity (matches ddrs, which is MERIT-only). Measured channels
give β ≈ 1.30–1.36 (the hardcoded 5/3 was 22–27% high). β is non-monotone in
b/y with minimum ≈ 1.07, not bounded below by 4/3.

**Cunge X (replaces static x).** Computed per timestep where celerity is
available:

```
X = clamp( 0.5·(1 − Q/(T·S·c·L)), 0, 0.5 )
```

with `Q = self._discharge_t`, `S` = clamped slope, `L` = length.
`calculate_muskingum_coefficients` keeps its signature but receives the
dynamic X. PyTorch autograd provides the backward — none of ddrs's
hand-derived gradient machinery is needed.

**Dead-code removal (orphaned by this change):** `RoutingDataclass.x`,
merit's fixed-0.3 fill, Lynker's `zarr_muskingum_x` read, `self.x_storage`.
Sweep the downstream checklist: `examples/`, `benchmarks/`, `scripts/`,
`config/`.

**Tests (mirroring ddrs gates):** β limits (→5/3 as b/y→∞; →4/3 as b→0 at
fixed z; non-monotone minimum ≈1.07), β vs finite-difference dQ/dA to ~1e-6,
X clamp bounds, `c1+c2+c3 = 1` exactly, `torch.autograd.gradcheck` (float64)
through the β and X paths. Gradcheck fixture must use long (~5000 m) reaches:
ddrs found short reaches saturate the X clamp and let the tests pass vacuously.

## Deliberately NOT ported (measured NO-GOs in ddrs)

- `enforce_positivity` (K floor + X cap): provably zero negative solves but
  replaces Cunge X almost everywhere (median X 0.4976 → 0.0794); skill impact
  unmeasured.
- Reach subdivision (variable Δx): built and measured — `c1 < 0` gets worse;
  2× network for a 35% negative-solve reduction.
- Courant sub-stepping: cannot work (K spans 44×; global Δt cannot compress
  the spread).
- Slope-floor 1e-3 removal: exact invariance (`n/√S` ratio), deferred in ddrs
  too; requires its own retrain campaign.
- Leakance: closed NO-GO in ddrs.

## Cross-cutting

- **CLAUDE.md** statements corrected in the stage that invalidates them:
  celerity 5/3 constant, Muskingum-x rows of the MERIT/Lynker table, tau
  boundary-trimming section.
- **Verification beyond unit tests:** after stages 3–4 a MERIT retrain should
  reproduce the *direction* of the ddrs result (routing beats the summed-Q'
  baseline). Retrains are user-run, not PR gates.
- **Risk:** Cunge X ≈ 0.49 nearly everywhere collapses the non-negative-
  coefficient window (2X ≤ Cr ≤ 2(1−X)) to ~1.4% wide; negative Muskingum
  coefficients rise (artifacts — initial dips/recession oscillation), while
  negative *solves* stayed ~0.1% in ddrs. The stage-2 counter is the
  tripwire: a retrain showing a rate far above ~0.1% is a stop-and-
  investigate signal.
- **Assumption:** DDR's trapezoid geometry chain matches ddrs S1–S14 (ddrs
  maintained an ABSOLUTE MATCH sandbox against DDR, so the shared geometry is
  bit-compatible by construction).
