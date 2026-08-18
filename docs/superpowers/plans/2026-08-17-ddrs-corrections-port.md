# ddrs Corrections Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the four ddrs `ddr_match` corrections into DDR as a clean break: gauge-reach `outflow_idx` extraction, negative-solve instrumentation, signed-at-zero tau convention (default 9), and corrected physics (trapezoid-exact celerity β + Cunge-derived Muskingum X).

**Architecture:** Four independent, bisectable commits on branch `port-ddrs-corrections`, landed in order. Stages 1–2 are behavior-preserving-adjacent (bug fix + instrumentation); stage 3 changes temporal alignment; stage 4 changes the routing physics and removes the static Muskingum-x plumbing. PyTorch autograd supplies all backward passes — no hand-derived gradients.

**Tech Stack:** Python 3.11+, PyTorch, Pydantic v2, pytest, `uv` for all commands.

**Spec:** `docs/superpowers/specs/2026-08-17-ddrs-corrections-port-design.md`

## Global Constraints

- Run everything with `uv` (`uv run pytest ...`). Never bare `python`/`pip`.
- Line length 110; ruff + mypy strict (`disallow_untyped_defs`); NumPy docstrings. Pre-commit runs on `git commit` — if it modifies files, `git add` them and commit again.
- Clean break: no compatibility flags. Old behavior is reproducible only via old commit hashes.
- Do NOT port: `enforce_positivity`, reach subdivision, Courant sub-stepping, slope-floor removal, leakance (measured NO-GOs / deferred in ddrs).
- New physics constants come verbatim from ddrs `.claude/PHYSICS-CORRECTIONS.md`: `β = 5/3 − (4/3)·A·√(1+z²)/(T·P)`, `X = clamp(0.5·(1 − Q/(T·S·c·L)), 0, 0.5)`.
- Unit tests must not require HPC data stores (`@pytest.mark.integration` is deselected by default).

---

### Task 1: Gauge-reach `outflow_idx` extraction (merit.py)

The MERIT collate currently sums a gauge's *upstream* columns, dropping the gauge reach's own local drainage (worst gauges predict 0.215× observed). The MC solve at the gauge reach already carries all upstream flow plus its own lateral inflow, so the mass-conserving extraction is the gauge reach itself. ddrs reference: `.claude/PHYSICS-CORRECTIONS.md` §"Outside the forward chain".

**Files:**
- Modify: `src/ddr/geodatazoo/merit.py:226-235` (training collate) and `src/ddr/geodatazoo/merit.py:468-477` (inference gages mode)
- Test: `tests/geodatazoo/test_merit_outflow.py` (create)

**Interfaces:**
- Produces: `_gage_outflow_indices(gage_idx, index_mapping) -> list[np.ndarray]` — module-level function in `merit.py`, one single-element array per gauge (the gauge reach's compressed index).
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/geodatazoo/test_merit_outflow.py`:

```python
"""Tests for the mass-conserving gauge outflow extraction in the MERIT dataset."""

import numpy as np

from ddr.geodatazoo.merit import _gage_outflow_indices


class TestGageOutflowIndices:
    def test_returns_gauge_reach_itself(self) -> None:
        # CONUS index -> compressed index
        index_mapping = {100: 0, 200: 1, 300: 2}
        gage_idx = np.array([300])

        result = _gage_outflow_indices(gage_idx, index_mapping)

        # The gauge reach itself, NOT its upstream columns: the MC solve at the
        # gauge reach already carries upstream flow plus its own lateral inflow.
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], np.array([2]))

    def test_one_entry_per_gauge_in_order(self) -> None:
        index_mapping = {10: 0, 20: 1, 30: 2, 40: 3}
        gage_idx = np.array([40, 10])

        result = _gage_outflow_indices(gage_idx, index_mapping)

        assert len(result) == 2
        np.testing.assert_array_equal(result[0], np.array([3]))
        np.testing.assert_array_equal(result[1], np.array([0]))

    def test_headwater_gauge(self) -> None:
        # A headwater gauge (no upstream reaches) maps to its own reach — same
        # rule, no special case needed anymore.
        index_mapping = {7: 0}
        result = _gage_outflow_indices(np.array([7]), index_mapping)
        np.testing.assert_array_equal(result[0], np.array([0]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/geodatazoo/test_merit_outflow.py -v`
Expected: FAIL with `ImportError: cannot import name '_gage_outflow_indices'`

- [ ] **Step 3: Implement**

In `src/ddr/geodatazoo/merit.py`, add a module-level function (near the top, after imports):

```python
def _gage_outflow_indices(gage_idx: np.ndarray, index_mapping: dict[int, int]) -> list[np.ndarray]:
    """Compressed outflow index for each gauge — the gauge reach itself.

    A USGS gauge measures all drainage above it, and the Muskingum-Cunge solve
    at the gauge reach already carries everything upstream plus the reach's own
    lateral inflow, so extracting the gauge reach is the mass-conserving
    choice. Summing the upstream columns instead (pre-2026-08-17 behavior)
    dropped the gauge reach's local drainage — up to ~78% of a small basin.

    Parameters
    ----------
    gage_idx : np.ndarray
        CONUS-index of the gauge reach for each gauge in the batch.
    index_mapping : dict[int, int]
        CONUS index → compressed (batch) index.

    Returns
    -------
    list[np.ndarray]
        One single-element array per gauge, in batch order.
    """
    return [np.array([index_mapping[int(_idx)]]) for _idx in gage_idx]
```

Then at BOTH construction sites (`merit.py:226-235` and `merit.py:468-477`), replace the ten-line loop:

```python
        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            original_col_indices = coo.col[local_gage_inflow_idx]
            if len(original_col_indices) > 0:
                compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            else:
                compressed_col_indices = np.array([index_mapping[int(_idx)]])
            outflow_idx.append(compressed_col_indices)
```

with:

```python
        outflow_idx = _gage_outflow_indices(_gage_idx, index_mapping)
```

- [ ] **Step 4: Verify flow_scale semantics still hold**

Read `build_flow_scale_tensor` in `src/ddr/io/readers.py:299` and its consumer `src/ddr/routing/mmc.py:303-304`. Confirm: flow_scale multiplies `q_prime` at the *gauge segment* (partial-area correction on lateral inflow). Under the old upstream-sum extraction this scaling never reached the gauge prediction (the gauge reach wasn't summed); under the fix it does — this is the correction becoming functional, not a regression. Add one sentence noting this to the commit message.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest tests/geodatazoo tests/routing tests/io -q`
Expected: all PASS (no existing test pins the upstream-sum behavior; if one does, update it to the gauge-reach semantics and say so in the commit).

- [ ] **Step 6: Commit**

```bash
git add src/ddr/geodatazoo/merit.py tests/geodatazoo/test_merit_outflow.py
git commit -m "fix(merit): extract gauge reach itself in outflow_idx — mass-conserving

Ported from ddrs (.claude/PHYSICS-CORRECTIONS.md): summing upstream columns
dropped the gauge reach's own local drainage (0.215x observed on the worst
small basins; 26/1841 gauges below 0.5x baseline). The MC solve at the gauge
reach already carries upstream flow plus its own lateral inflow. Side effect:
partial-area flow_scale corrections now actually reach MERIT gauge
predictions.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Negative-solve counter (mmc.py)

`route_timestep` clamps the sparse-solve output to `discharge_lb`, silently creating mass where the solution went negative. Count violations before the clamp and log the per-forward rate. ddrs measured ~0.03–0.14% of reach-timesteps under corrected physics; this counter is the stage-4 tripwire.

**Files:**
- Modify: `src/ddr/routing/mmc.py` (`forward()` ~line 365, `route_timestep()` ~line 556)
- Test: `tests/routing/test_mmc.py` (append)

**Interfaces:**
- Produces: `MuskingumCunge.neg_solve_count: int`, `MuskingumCunge.neg_solve_total: int` — reset at each `forward()`, updated every `route_timestep()`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Append to `tests/routing/test_mmc.py`:

```python
class TestNegativeSolveCounter:
    """The solve-output clamp silently creates mass; count violations first."""

    def test_counter_counts_negative_solutions(self) -> None:
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        streamflow = create_mock_streamflow(num_timesteps=6, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        def mock_solver(*args: Any, **kwargs: Any) -> torch.Tensor:
            # Two negative entries per timestep
            return torch.tensor([-1.0, 2.0, -0.5, 3.0])

        with patch("ddr.routing.mmc.triangular_sparse_solve", side_effect=mock_solver):
            mc.forward()

        # 5 routed timesteps (num_timesteps - 1), 2 negatives of 4 reaches each
        assert mc.neg_solve_total == 5 * 4
        assert mc.neg_solve_count == 5 * 2

    def test_counter_resets_between_forwards(self) -> None:
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        streamflow = create_mock_streamflow(num_timesteps=6, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        mc.forward()
        first_total = mc.neg_solve_total
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)
        mc.forward()

        assert mc.neg_solve_total == first_total  # reset, not accumulated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/routing/test_mmc.py::TestNegativeSolveCounter -v`
Expected: FAIL with `AttributeError: ... has no attribute 'neg_solve_total'`

- [ ] **Step 3: Implement**

In `MuskingumCunge.__init__` (near the other state attrs, ~line 218):

```python
        # Negative-solve instrumentation: the S28-style clamp below rewrites
        # negative solve output to discharge_lb, creating mass. Count first.
        self.neg_solve_count: int = 0
        self.neg_solve_total: int = 0
```

In `forward()`, immediately before the `for timestep in tqdm(...)` loop (~line 415):

```python
        self.neg_solve_count = 0
        self.neg_solve_total = 0
```

and immediately after the loop completes (before `return output`):

```python
        if self.neg_solve_total > 0 and self.neg_solve_count > 0:
            rate = 100.0 * self.neg_solve_count / self.neg_solve_total
            log.info(
                f"negative solve output: {self.neg_solve_count}/{self.neg_solve_total} "
                f"reach-timesteps ({rate:.4f}%) clamped to discharge_lb"
            )
```

(`mmc.py` already has `log = logging.getLogger(...)` — verify; if the module logger has a different name, use it.)

In `route_timestep()`, between the solve and the clamp (~line 556):

```python
        # Count negatives BEFORE the clamp rewrites them (mass creation)
        self.neg_solve_count += int((solution < 0).sum().item())
        self.neg_solve_total += solution.numel()

        # Clamp solution to physical bounds
        q_t1 = torch.clamp(solution, min=self.discharge_lb)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/routing/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ddr/routing/mmc.py tests/routing/test_mmc.py
git commit -m "feat(routing): count negative solve output before the discharge clamp

The clamp to discharge_lb silently creates mass and hides Courant
instability. ddrs measured ~0.03-0.14% of reach-timesteps negative under
corrected physics; this counter is the tripwire for the Cunge-X port.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Signed-at-zero tau convention, default 9

Port ddrs's 2026-08-08 convention (`docs/2026-08-06-tau-sweep-pilot-findings.md` §5i): slice `[tau : -(24-tau)]`, pooled day *i* scored against obs day *i*, runtime tau ∈ [0, 24), default 9. The shipped legacy tau=3 ≡ signed −8 (mis-set in the wrong direction); the fix alone was worth +0.086 median NSE in ddrs. Legacy pin: new tau=16 cuts the same hour window as legacy tau=3.

**Window arithmetic (derived from `Dates`, `dataclasses.py:110-121`):** `daily_time_range` has D days; `hourly_time_range` is `inclusive="left"` → H = 24·(D−1) hours. New slice length = H − 24 = 24·(D−2) → D−2 pooled days, exactly. Pooled day j covers hours `[tau+24j, tau+24(j+1))` = calendar day j advanced tau hours → pairs with **obs day j**. So predictions align with `obs[:, :-2]` and `daily_time_range[:-2]` (legacy paired with `[1:-1]`). Legacy train.py sliced `[13 : -11+tau]` (length H−24+tau, not divisible by 24 — the downsample smeared a fractional day); this task removes that inconsistency.

**Files:**
- Modify: `src/ddr/scripts_utils.py:18-42`, `src/ddr/validation/configs.py:116-119`, `scripts/train.py:78-91`, `scripts/test.py:76-78`, `scripts/train_and_test.py:97-99`, `scripts/router.py:169-170`, `benchmarks/src/ddr_benchmarks/benchmark.py:787-796`, `CLAUDE.md` (tau + boundary-trimming sections)
- Test: `tests/scripts/test_scripts_utils.py` (rewrite `TestComputeDailyRunoff`)

**Interfaces:**
- Produces: `tau_trim_and_downsample(hourly_predictions: torch.Tensor, tau: int) -> torch.Tensor` (keeps autograd; used by train.py); `compute_daily_runoff(hourly_predictions: torch.Tensor, tau: int) -> np.ndarray` (numpy wrapper, same name as before). Both raise `ValueError` unless `0 <= tau < 24`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Replace `TestComputeDailyRunoff` in `tests/scripts/test_scripts_utils.py` (keep the other classes; add `tau_trim_and_downsample` to the import):

```python
class TestTauTrimAndDownsample:
    """Signed-at-zero tau convention: slice [tau : -(24-tau)], day i ↔ obs day i."""

    def test_shape(self) -> None:
        # 5 gages, 240 hours (10 hourly-days) → slice removes 24 h → 9 days
        hourly = torch.rand(5, 240)
        result = tau_trim_and_downsample(hourly, tau=9)
        assert result.shape == (5, 9)

    def test_tau_zero_is_day_aligned(self) -> None:
        # Day 0 = hours [0, 24). Encode the day index in the data.
        hourly = torch.arange(72, dtype=torch.float32).unsqueeze(0) // 24
        result = tau_trim_and_downsample(hourly, tau=0)
        assert torch.allclose(result[0], torch.tensor([0.0, 1.0]))

    def test_tau_advances_window(self) -> None:
        # With tau=6, pooled day 0 covers hours [6, 30): 18 h of day-0 value
        # (0.0) and 6 h of day-1 value (1.0) → mean 6/24 = 0.25.
        hourly = torch.arange(72, dtype=torch.float32).unsqueeze(0) // 24
        result = tau_trim_and_downsample(hourly, tau=6)
        assert torch.allclose(result[0], torch.tensor([0.25, 1.25]))

    def test_legacy_equivalence_pin(self) -> None:
        # ddrs findings §5i: legacy tau=3 cut hours [16:-8]; the new convention
        # cuts the same hour window at tau=16.
        hourly = torch.rand(3, 240)
        new = tau_trim_and_downsample(hourly, tau=16)
        legacy_sliced = hourly[:, 16:-8]
        legacy = downsample(legacy_sliced, rho=legacy_sliced.shape[1] // 24)
        assert torch.allclose(new, legacy)

    def test_preserves_gradients(self) -> None:
        hourly = torch.rand(2, 120, requires_grad=True)
        result = tau_trim_and_downsample(hourly, tau=9)
        result.sum().backward()
        assert hourly.grad is not None

    def test_rejects_out_of_range_tau(self) -> None:
        hourly = torch.rand(1, 96)
        with pytest.raises(ValueError):
            tau_trim_and_downsample(hourly, tau=24)
        with pytest.raises(ValueError):
            tau_trim_and_downsample(hourly, tau=-1)


class TestComputeDailyRunoff:
    """Numpy wrapper around tau_trim_and_downsample."""

    def test_returns_numpy(self) -> None:
        hourly = torch.rand(2, 120)
        result = compute_daily_runoff(hourly, tau=9)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 4)

    def test_flat_input_daily_mean(self) -> None:
        hourly = torch.ones(1, 72)
        result = compute_daily_runoff(hourly, tau=9)
        assert np.allclose(result, 1.0, atol=1e-4)
```

Add the imports at the top of the file: `import pytest`, `from ddr.io.functions import downsample`, and extend the `ddr.scripts_utils` import with `tau_trim_and_downsample`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/scripts/test_scripts_utils.py -v`
Expected: FAIL with `ImportError: cannot import name 'tau_trim_and_downsample'`

- [ ] **Step 3: Implement in scripts_utils.py**

Replace `compute_daily_runoff` (`src/ddr/scripts_utils.py:18-42`) with:

```python
def tau_trim_and_downsample(
    hourly_predictions: torch.Tensor,
    tau: int,
) -> torch.Tensor:
    """Slice hourly predictions on the signed-at-zero tau convention, pool to daily.

    tau is the number of hours the routed output is advanced before daily
    scoring (dMC-Juniata's sign; tau=0 is day-aligned): the slice is
    ``[tau : -(24 - tau)]`` and pooled day *i* aligns with calendar day *i* of
    the batch window. Ported from ddrs (tau-sweep findings §5i); the legacy
    ``[13+tau : -11+tau]`` convention with shipped tau=3 was signed −8 —
    mis-set in the wrong direction. Legacy pin: new tau=16 cuts the same hour
    window as legacy tau=3.

    Preserves autograd — used directly in the training loss path.

    Parameters
    ----------
    hourly_predictions : torch.Tensor
        Hourly discharge, shape (num_gages, num_hours).
    tau : int
        Hours of advance, ``0 <= tau < 24``. Default in config is 9 (measured
        optimum for the flagship Q' sources; per-source optima differ — the
        aorc2f-lumped store measured ≈ −8, outside the runtime range).

    Returns
    -------
    torch.Tensor
        Daily discharge, shape (num_gages, num_days).
    """
    if not 0 <= tau < 24:
        raise ValueError(f"tau must be in [0, 24), got {tau}")
    sliced = hourly_predictions[:, tau : -(24 - tau)]
    num_days = sliced.shape[1] // 24
    return downsample(sliced, rho=num_days)


def compute_daily_runoff(
    hourly_predictions: torch.Tensor,
    tau: int,
) -> np.ndarray:
    """Numpy wrapper around :func:`tau_trim_and_downsample` for eval scripts.

    Parameters
    ----------
    hourly_predictions : torch.Tensor
        Hourly discharge, shape (num_gages, num_hours).
    tau : int
        Hours of advance, ``0 <= tau < 24``.

    Returns
    -------
    np.ndarray
        Daily discharge, shape (num_gages, num_days).
    """
    return tau_trim_and_downsample(hourly_predictions, tau).numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/scripts/test_scripts_utils.py -v`
Expected: PASS.

- [ ] **Step 5: Update config default and validation**

`src/ddr/validation/configs.py:116-119`:

```python
    tau: int = Field(
        default=9,
        ge=0,
        lt=24,
        description=(
            "Signed-at-zero timing advance: routed output is advanced tau hours "
            "before daily pooling, pooled day i scores against obs day i. "
            "Default 9 is the measured CONUS optimum (ddrs tau sweep, 2026-08-08). "
            "Legacy [13+tau : -11+tau] convention removed 2026-08-17."
        ),
    )
```

Then `grep -rn "tau" config/*.yaml` — no config currently sets `tau:` (verified 2026-08-17), so the new default applies everywhere; if any YAML has appeared since, update its value to 9 or delete the line.

- [ ] **Step 6: Update the four scripts and benchmarks**

`scripts/train.py:78-91` — replace the inline slice and obs pairing. Old:

```python
                num_days = len(dmc_output["runoff"][0][13 : (-11 + cfg.params.tau)]) // 24
                daily_runoff = ddr_functions.downsample(
                    dmc_output["runoff"][:, 13 : (-11 + cfg.params.tau)],
                    rho=num_days,
                )
```

New (add `from ddr.scripts_utils import tau_trim_and_downsample` to the imports; keep the existing import line's other names):

```python
                daily_runoff = tau_trim_and_downsample(dmc_output["runoff"], cfg.params.tau)
```

Old obs pairing:

```python
                )[:, 1:-1]  # Cutting off days to match with realigned timesteps
```

New:

```python
                )[:, :-2]  # Pooled day i ↔ obs day i; last 2 obs days have no pooled window
```

`scripts/test.py:77-78` and `scripts/train_and_test.py:98-99` — old:

```python
    daily_obs = observations[:, 1:-1]
    time_range = dataset.dates.daily_time_range[1:-1]
```

new:

```python
    daily_obs = observations[:, :-2]
    time_range = dataset.dates.daily_time_range[:-2]
```

`scripts/router.py:170` — old `time_range = dataset.dates.daily_time_range[1:-1]`, new `time_range = dataset.dates.daily_time_range[:-2]`.

`benchmarks/src/ddr_benchmarks/benchmark.py:787-796` — replace both `(13 + cfg.params.tau) : (-11 + cfg.params.tau)` slices and the `[13 : ...]` day count with `compute_daily_runoff(...)` / `tau_trim_and_downsample(...)` calls on the same tensors, and shift its obs pairing from `[1:-1]` to `[:-2]` if present at those lines (read the surrounding function first; keep its structure otherwise).

- [ ] **Step 7: Update CLAUDE.md**

In `CLAUDE.md`: (a) **Params** section: `tau: int = 9` — signed-at-zero timing advance (was 3, legacy convention). (b) **Boundary Trimming** section: replace the `output[:, (13+tau):(-11+tau)]` description with the `[tau : -(24−tau)]` convention, pooled day i ↔ obs day i. (c) **Training Pipeline** step 5: `daily = downsample(output[:, tau:-(24-tau)])`.

- [ ] **Step 8: Run the affected suites**

Run: `uv run pytest tests/scripts tests/validation tests/routing -q`
Expected: PASS. `tests/validation/test_configs.py` may pin `tau == 3` — if so, update to 9 (that pin is the old default, not a behavior contract).

- [ ] **Step 9: Commit**

```bash
git add src/ddr/scripts_utils.py src/ddr/validation/configs.py scripts/train.py scripts/test.py scripts/train_and_test.py scripts/router.py benchmarks/src/ddr_benchmarks/benchmark.py tests/scripts/test_scripts_utils.py tests/validation CLAUDE.md
git commit -m "feat(tau)!: signed-at-zero tau convention, default 9

Slice [tau : -(24-tau)], pooled day i scored against obs day i (dMC-Juniata
sign; tau=0 day-aligned). Shipped legacy tau=3 was signed -8 — mis-set in the
wrong direction; ddrs measured the fix alone at +0.086 median NSE, flipping
routing from losing to the summed-Q' baseline to beating it. Also unifies
train.py's inline slice (was [13 : -11+tau], a fractional-day window) with
the shared helper. BREAKING: pre-port checkpoints trained on the legacy
window; config tau values do not carry across.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Corrected physics — celerity β + Cunge X, remove static x

Two changes in the MC core, from ddrs `.claude/PHYSICS-CORRECTIONS.md` S17/S19: (1) celerity `c = v·β` with the trapezoid-exact `β = 5/3 − (4/3)·A·√(1+z²)/(T·P)` (the hardcoded 5/3 is the wide-rectangular limit, 22–27% high on real channels); (2) Muskingum X computed per timestep from Cunge's diffusion matching, `X = clamp(0.5·(1 − Q/(T·S·c·L)), 0, 0.5)`, replacing the static per-reach x (MERIT: constant 0.3; Lynker: zarr). The static-x plumbing becomes dead and is removed. Autograd differentiates both paths (X depends on `_discharge_t`, so gradients flow through time — matching ddrs's gq_t path).

Known risk (spec): Cunge X ≈ 0.49 nearly everywhere collapses the non-negative-coefficient window; negative *coefficients* rise but negative *solves* stayed ~0.1% in ddrs. The Task-2 counter reports the rate every forward.

**Files:**
- Modify: `src/ddr/routing/mmc.py` (`_get_trapezoid_velocity` ~lines 100-170, `calculate_muskingum_coefficients` 460-485, `route_timestep` 487-559, `_set_network_context` 289, `__init__` 218)
- Modify (x removal): `src/ddr/geodatazoo/dataclasses.py:217-218,257`, `src/ddr/geodatazoo/merit.py:262,313,388,426,504`, `src/ddr/geodatazoo/lynker_hydrofabric.py:78,89,291,349-350,425,464,543`, `src/ddr/geodatazoo/base_geodataset.py:228`, `tests/routing/test_utils.py` (MockRoutingDataclass.x), `tests/routing/test_mmc.py` (x_storage assertions), `CLAUDE.md`
- Test: `tests/routing/test_celerity_beta.py` (create), `tests/routing/test_cunge_x.py` (create)

**Interfaces:**
- Produces: `_get_trapezoid_velocity` — same signature, but returned celerity uses β; `calculate_muskingum_coefficients(self, length, celerity, x)` — same arity, `x` is now the per-timestep Cunge X tensor.
- Consumes: `neg_solve_count`/`neg_solve_total` from Task 2 (reporting only).

- [ ] **Step 1: Write the failing β tests**

Create `tests/routing/test_celerity_beta.py`:

```python
"""Trapezoid-exact kinematic celerity: c = v·β, β = 5/3 − (4/3)·A·√(1+z²)/(T·P).

Ported from ddrs (celerity_beta gates). β limits: → 5/3 as b/y → ∞ (wide
rectangular), → 4/3 as b → 0 at fixed z. β is NON-monotone in b/y and reaches
≈1.07 for narrow sections — it is not bounded below by 4/3.
"""

import torch


def _beta(bottom_width: torch.Tensor, depth: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Reference β for a trapezoid parameterized by (b, y, z)."""
    top_width = bottom_width + 2.0 * z * depth
    area = (bottom_width + top_width) * depth / 2.0
    wetted_perimeter = bottom_width + 2.0 * depth * torch.sqrt(1.0 + z**2)
    return 5.0 / 3.0 - (4.0 / 3.0) * area * torch.sqrt(1.0 + z**2) / (top_width * wetted_perimeter)


class TestBetaLimits:
    def test_wide_rectangular_limit(self) -> None:
        # b/y → ∞, z → 0: β → 5/3
        beta = _beta(torch.tensor(1e6), torch.tensor(1.0), torch.tensor(0.0))
        assert torch.isclose(beta, torch.tensor(5.0 / 3.0), atol=1e-4)

    def test_triangular_limit(self) -> None:
        # b → 0 at fixed z: A = z·y², P = 2y√(1+z²), T = 2zy
        # → β = 5/3 − (4/3)·(z·y²·√(1+z²))/(2zy·2y√(1+z²)) = 5/3 − 1/3 = 4/3
        beta = _beta(torch.tensor(1e-8), torch.tensor(2.0), torch.tensor(1.5))
        assert torch.isclose(beta, torch.tensor(4.0 / 3.0), atol=1e-4)

    def test_narrow_section_below_four_thirds(self) -> None:
        # β is non-monotone in b/y; near b/y ≈ 2 at z = 0 it dips well below 4/3
        beta = _beta(torch.tensor(2.0), torch.tensor(1.0), torch.tensor(0.0))
        assert beta < 4.0 / 3.0

    def test_beta_matches_finite_difference_dQ_dA(self) -> None:
        # c = dQ/dA must equal v·β. Manning: v = n⁻¹ R^(2/3) √S, Q = v·A.
        # Perturb depth; dQ/dA = (dQ/dy)/(dA/dy).
        n, s, b, z = 0.03, 1e-3, torch.tensor(5.0), torch.tensor(1.2)

        def q_and_a(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            tw = b + 2.0 * z * y
            area = (b + tw) * y / 2.0
            wp = b + 2.0 * y * torch.sqrt(1.0 + z**2)
            v = (1.0 / n) * (area / wp) ** (2.0 / 3.0) * s**0.5
            return v * area, area

        y0 = torch.tensor(1.7, dtype=torch.float64)
        eps = 1e-6
        q_hi, a_hi = q_and_a(y0 + eps)
        q_lo, a_lo = q_and_a(y0 - eps)
        dq_da_fd = (q_hi - q_lo) / (a_hi - a_lo)

        q0, a0 = q_and_a(y0)
        v0 = q0 / a0
        beta = _beta(b.double(), y0, z.double())
        assert torch.isclose(v0 * beta, dq_da_fd, rtol=1e-6)


class TestRoutingUsesBeta:
    def test_route_timestep_celerity_is_not_five_thirds(self) -> None:
        """The routed celerity must reflect β, not the hardcoded 5/3."""
        from typing import Any
        from unittest.mock import patch

        from ddr.routing.mmc import MuskingumCunge
        from tests.routing.test_utils import (
            create_mock_config,
            create_mock_routing_dataclass,
            create_mock_spatial_parameters,
            create_mock_streamflow,
        )

        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        captured: dict[str, Any] = {}
        original = MuskingumCunge.calculate_muskingum_coefficients

        def spy(self: Any, length: Any, celerity: Any, x: Any) -> Any:
            captured["celerity"] = celerity
            # discharge as it was when this celerity was computed
            captured["discharge"] = self._discharge_t.clone()
            return original(self, length, celerity, x)

        with patch.object(MuskingumCunge, "calculate_muskingum_coefficients", spy):
            mc.forward()

        from ddr.geometry.trapezoidal import compute_trapezoidal_geometry

        # Recompute the Manning velocity at the SAME discharge the captured
        # call used (forward() advances _discharge_t after each step)
        geom = compute_trapezoidal_geometry(
            n=mc.n,
            p_spatial=mc.p_spatial,
            q_spatial=mc.q_spatial,
            discharge=captured["discharge"],
            slope=mc.slope,
            depth_lb=float(mc.depth_lb),
            bottom_width_lb=float(mc.bottom_width_lb),
        )
        # β < 5/3 strictly for any finite trapezoid: celerity must be < v·5/3
        v = torch.clamp(geom["velocity"], min=mc.velocity_lb, max=torch.tensor(15.0))
        assert (captured["celerity"] < v * 5.0 / 3.0).all()
        assert (captured["celerity"] > v * 1.0).all()  # β > 1 on physical channels
```

- [ ] **Step 2: Write the failing Cunge-X tests**

Create `tests/routing/test_cunge_x.py`:

```python
"""Cunge-derived Muskingum X: X = clamp(0.5·(1 − Q/(T·S·c·L)), 0, 0.5).

Matches the scheme's numerical diffusion D_num = c·L·(0.5−X) to the channel's
physical diffusivity D_phys = Q/(2·T·S). Replaces the static per-reach x
(MERIT constant 0.3 / Lynker zarr). Fixture warning from ddrs: short reaches
saturate the clamp at 0 and pass vacuously — use ≥5000 m where X must be
interior.
"""

from typing import Any
from unittest.mock import patch

import torch

from ddr.routing.mmc import MuskingumCunge
from tests.routing.test_utils import (
    create_mock_config,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)


def _cunge_x(
    q: torch.Tensor, t: torch.Tensor, s: torch.Tensor, c: torch.Tensor, length: torch.Tensor
) -> torch.Tensor:
    return torch.clamp(0.5 * (1.0 - q / (t * s * c * length)), min=0.0, max=0.5)


class TestCungeXFormula:
    def test_diffusion_matching_interior(self) -> None:
        # Interior X: D_num = c·L·(0.5−X) must equal D_phys = Q/(2·T·S)
        q, t, s, c, length = (
            torch.tensor(50.0),
            torch.tensor(30.0),
            torch.tensor(1e-3),
            torch.tensor(1.5),
            torch.tensor(5000.0),
        )
        x = _cunge_x(q, t, s, c, length)
        assert 0.0 < x < 0.5
        d_num = c * length * (0.5 - x)
        d_phys = q / (2.0 * t * s)
        assert torch.isclose(d_num, d_phys, rtol=1e-6)

    def test_clamps_to_zero_for_diffusive_channels(self) -> None:
        # Large Q over a short flat reach → raw X negative → clamp at 0
        x = _cunge_x(
            torch.tensor(500.0),
            torch.tensor(10.0),
            torch.tensor(1e-3),
            torch.tensor(1.0),
            torch.tensor(500.0),
        )
        assert x == 0.0

    def test_never_exceeds_half(self) -> None:
        x = _cunge_x(
            torch.tensor(1e-4),
            torch.tensor(50.0),
            torch.tensor(0.01),
            torch.tensor(3.0),
            torch.tensor(10000.0),
        )
        assert x <= 0.5


class TestRoutingUsesCungeX:
    def _spy_forward(self) -> dict[str, Any]:
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        # Long reaches so X is interior, not clamp-saturated (ddrs gotcha)
        hydrofabric.length = torch.full((4,), 5000.0)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        captured: dict[str, Any] = {}
        original = MuskingumCunge.calculate_muskingum_coefficients

        def spy(self: Any, length: Any, celerity: Any, x: Any) -> Any:
            captured["x"] = x
            captured["coeffs"] = original(self, length, celerity, x)
            return captured["coeffs"]

        with patch.object(MuskingumCunge, "calculate_muskingum_coefficients", spy):
            mc.forward()
        return captured

    def test_x_is_dynamic_and_bounded(self) -> None:
        captured = self._spy_forward()
        x = captured["x"]
        assert x.shape == (4,)  # per-reach, computed per timestep
        assert (x >= 0.0).all() and (x <= 0.5).all()

    def test_coefficient_identity(self) -> None:
        # c1 + c2 + c3 = 1 exactly, for ANY (K, X) — mass conservation
        captured = self._spy_forward()
        c1, c2, c3, _c4 = captured["coeffs"]
        assert torch.allclose(c1 + c2 + c3, torch.ones_like(c1), atol=1e-6)

    def test_gradients_flow_through_x(self) -> None:
        # X depends on discharge and geometry; the KAN parameters must receive
        # gradients through the X path (ddrs verified this branch falsifiable).
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hydrofabric = create_mock_routing_dataclass(num_reaches=4)
        hydrofabric.length = torch.full((4,), 5000.0)
        streamflow = create_mock_streamflow(num_timesteps=4, num_reaches=4)
        spatial_params = create_mock_spatial_parameters(num_reaches=4)
        for v in spatial_params.values():
            v.requires_grad_(True)
        mc.setup_inputs(hydrofabric, streamflow, spatial_params)

        output = mc.forward()
        output.sum().backward()
        for name, v in spatial_params.items():
            assert v.grad is not None, f"no gradient reached spatial parameter {name!r}"
            assert torch.isfinite(v.grad).all()
```

- [ ] **Step 3: Run new tests to verify they fail**

Run: `uv run pytest tests/routing/test_celerity_beta.py tests/routing/test_cunge_x.py -v`
Expected: `TestBetaLimits`/`TestCungeXFormula` PASS (they test reference formulas), `TestRoutingUsesBeta`/`TestRoutingUsesCungeX` FAIL (routing still uses 5/3 and `x_storage`; the spy signature `(length, celerity, x)` also matches the current positional call).

- [ ] **Step 4: Implement β in `_get_trapezoid_velocity`**

In `src/ddr/routing/mmc.py` (~lines 160-168), replace:

```python
    # Compute celerity from velocity (routing-specific, not in geometry module)
    v = geom["velocity"]
    c_ = torch.clamp(v, min=velocity_lb, max=torch.tensor(15.0, device=v.device))
    c = c_ * 5 / 3
    return c, top_width, side_slope
```

with:

```python
# Kinematic celerity c = dQ/dA = v·β for the trapezoid actually built above.
# β from the INTERNAL geometry (pre data-override) so celerity stays
# consistent with the section that produced the velocity. The old constant
# 5/3 is the wide-rectangular limit — 22-27% high on real channels (ddrs).
v = geom["velocity"]
c_ = torch.clamp(v, min=velocity_lb, max=torch.tensor(15.0, device=v.device))
beta = 5.0 / 3.0 - (4.0 / 3.0) * geom["cross_sectional_area"] * torch.sqrt(1.0 + geom["side_slope"] ** 2) / (
    geom["top_width"] * geom["wetted_perimeter"]
)
c = c_ * beta
return c, top_width, side_slope
```

- [ ] **Step 5: Implement Cunge X in `route_timestep` and rename coefficient params**

`calculate_muskingum_coefficients` (mmc.py:460-485): rename parameters `velocity` → `celerity` and `x_storage` → `x` (the old names were wrong — the value passed has always been the celerity, and x is now per-timestep). Body: replace all `velocity`/`x_storage` occurrences with `celerity`/`x`; update the docstring (`celerity : torch.Tensor — kinematic wave celerity`, `x : torch.Tensor — Cunge-derived Muskingum weighting, [0, 0.5]`).

In `route_timestep` (mmc.py:506-532): remove `or self.x_storage is None` from the guard, and replace:

```python
        # Calculate routing coefficients
        c_1, c_2, c_3, c_4 = self.calculate_muskingum_coefficients(self.length, velocity, self.x_storage)
```

with:

```python
        # Cunge X: match numerical diffusion c·L·(0.5−X) to physical
        # diffusivity Q/(2·T·S). Clamped to [0, 0.5]; denominators are safe —
        # slope/top_width/discharge are already clamped to positive floors.
        x_cunge = torch.clamp(
            0.5 * (1.0 - self._discharge_t / (self.top_width * self.slope * velocity * self.length)),
            min=0.0,
            max=0.5,
        )
        c_1, c_2, c_3, c_4 = self.calculate_muskingum_coefficients(self.length, velocity, x_cunge)
```

(`velocity` here is the celerity returned by `_get_trapezoid_velocity`; `self.top_width` was just set by that call — the data-override value on Lynker, which is the actual channel. Ordering note: `self.top_width` must be assigned before this line; it is, at mmc.py:518.)

- [ ] **Step 6: Remove the static-x plumbing**

All references, enumerated (re-verify with `grep -rn "x_storage" src/ tests/` and `grep -rn '\bx=' src/ddr/geodatazoo/`):

- `mmc.py:218` — delete `self.x_storage: torch.Tensor | None = None`
- `mmc.py:289` — delete `self.x_storage = routing_dataclass.x.to(...)`
- `src/ddr/geodatazoo/dataclasses.py:217-218` (docstring) and `:257` (field) — delete `x`
- `src/ddr/geodatazoo/merit.py:313` — delete the `flowpath_tensors["x"] = torch.full_like(...)` fill; delete `x=flowpath_tensors["x"],` kwargs at lines 262, 388, 426, 504
- `src/ddr/geodatazoo/lynker_hydrofabric.py:78` (`self.zarr_muskingum_x = ...`), `:89` (its use in the NaN-fill block — read the block, remove only the muskingum_x entry), `:349-350` (`"x": fill_nans(...)`), and `x=flowpath_tensors["x"],` kwargs at 291, 425, 464, 543
- `src/ddr/geodatazoo/base_geodataset.py:228` — remove the `"x"` bullet from the docstring
- `tests/routing/test_utils.py` — remove `self.x = ...` lines from `MockRoutingDataclass` (both the init and the clamp)
- `tests/routing/test_mmc.py:38` region — remove `assert mc.x_storage is not None` and `assert_tensor_properties(mc.x_storage, (10,))`
- `tests/geodatazoo/test_dataclasses.py` — `grep -n '\bx\b'`; remove any `x=`/`.x` usage the same way

Then sweep the downstream checklist (per CLAUDE.md): `grep -rn "x_storage\|muskingum_x\|\bx=flowpath" examples/ benchmarks/ scripts/ src/` — fix any remaining site the greps surface. (`benchmarks` DiffRoute's own `x` parameter at `benchmark.py:201` is DiffRoute's Muskingum x, NOT DDR's — leave it.)

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS, including the new β/X tests. Watch for stragglers pinning 5/3 or `x_storage` in `tests/routing/test_torch_mc.py`, `tests/bmi/`, `tests/dataset/` — update them to the new semantics (the BMI computes velocity only, not celerity, so it should be untouched).

- [ ] **Step 8: Update CLAUDE.md**

- Architecture list item 3 + "Key Constants" bullet: celerity is `c = v·β`, `β = 5/3 − (4/3)·A·√(1+z²)/(T·P)` (trapezoid-exact; wide-rectangular limit 5/3 removed 2026-08-17).
- MERIT vs Lynker table: replace the "Muskingum x" row (`Fixed 0.3` / `From zarr`) with `Cunge-derived per timestep (both)`.
- `routing/mmc.py` module-map row: mention Cunge X and the negative-solve counter.

- [ ] **Step 9: Commit**

```bash
git add src/ddr/routing/mmc.py src/ddr/geodatazoo/ tests/ CLAUDE.md
git commit -m "feat(routing)!: trapezoid-exact celerity and Cunge-derived Muskingum X

c = v·β with β = 5/3 − (4/3)·A·√(1+z²)/(T·P) — the hardcoded 5/3 is the
wide-rectangular limit, 22-27% high on real channels. X is now computed per
timestep from Cunge's diffusion matching, clamp(0.5·(1 − Q/(T·S·c·L)), 0,
0.5), replacing the static per-reach x (MERIT constant 0.3, Lynker zarr) —
that plumbing is removed. Ported from ddrs ddr_match:false physics
(.claude/PHYSICS-CORRECTIONS.md). BREAKING: pre-port checkpoints were
trained against the legacy physics. Known trade: Cunge X≈0.5 narrows the
non-negative-coefficient window; the negative-solve counter reports the
clamp rate every forward (~0.1% expected, ddrs-measured).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Verification after all tasks

1. `uv run pytest -q` — full unit suite green.
2. `uv run ruff check . && uv run ruff format --check .` (pre-commit covers this, but verify).
3. Grep gates: `grep -rn "5 / 3\|x_storage\|13 + tau\|(-11 + tau)" src/ scripts/ benchmarks/src/` → no hits outside docs/git history.
4. User-run (not a gate): MERIT retrain; expect routing to beat the summed-Q' baseline directionally (ddrs: 0.706 vs 0.642 median NSE) and the negative-solve rate near ~0.1%. A retrain at the Task-3 commit vs the Task-4 commit attributes tau vs physics — the control ddrs never ran.
