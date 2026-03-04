# Level Pool Reservoir Routing

DDR models reservoir storage and release via a **level pool** formulation integrated directly into the Muskingum-Cunge sparse solve. Reservoir reaches are treated as lumped storage nodes with weir + orifice outflow, while all other reaches continue to use standard MC routing. The key innovation is a **single-solve RHS override** that eliminates within-timestep lag and prevents numerical blowup.

## Motivation

Muskingum-Cunge routing has no mechanism for storage/release. Reservoir-dominated gages (e.g., 06259000 NSE=-17.31, 01019000 NSE=-0.29) fail catastrophically because MC can only reshape timing and attenuation of lateral inflow — it cannot hold water across days or weeks as a reservoir does.

Level pool routing addresses this by treating ~46,000 MERIT reaches intersecting HydroLAKES waterbodies (13.3% of CONUS network) as storage nodes with physically-based outflow equations.

## Physical Setup

A reservoir reach stores water at pool elevation $H$ and releases through two outlets:

```
                  inflow (Q_in)
                      |
                      v
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  <-- pool elevation H
    |                              |
    |     Lake surface area A_s    |
    |                              |
    |          RESERVOIR           |
    |                              |
====|==============================|====  <-- weir crest (WE)
    |                              |
    |        stored volume         |      Q_weir (over weir crest)
    |                              |  --> when H > WE
    |                              |
----|------------------------------|----  <-- orifice center (OE)
    |          ///O///             |  --> Q_orifice (through orifice)
    |                              |      when H > OE
====================================
```

**Reference frame:**

- Elevations are absolute (from HydroLAKES Elevation field)
- Weir crest sits at 75% of full pool: `WE = elevation - 0.25 * depth`
- Orifice center sits at lake bottom: `OE = elevation - depth`

## Outflow Equations

### Weir Discharge

Water flowing over the weir crest (broad-crested weir equation):

```
Q_weir = C_w * W_L * max(H - WE, 0)^(3/2)
```

Where:

- `C_w` = weir discharge coefficient (default 0.4, broad-crested)
- `W_L` = effective weir length [m] (1% of shoreline length)
- `H - WE` = head over weir crest [m]

Active only when pool elevation exceeds the weir crest (`H > WE`).

### Orifice Discharge

Water flowing through the low-level outlet (orifice equation):

```
Q_orifice = C_o * O_a * sqrt(2g * max(H - OE, 0) + eps)
```

Where:

- `C_o` = orifice discharge coefficient (default 0.6)
- `O_a` = orifice cross-sectional area [m^2] (back-calculated from average discharge)
- `H - OE` = head above orifice center [m]
- `g` = 9.81 m/s^2
- `eps` = 1e-8 (numerical stability)

Active only when pool elevation exceeds the orifice elevation (`H > OE`).

### Total Outflow

```
Q_out = max(Q_weir + Q_orifice, discharge_lb)
```

The lower bound (`discharge_lb`) ensures non-negative, physically bounded outflow.

## Pool Elevation Update

Forward Euler timestep with mass balance, computed **after** the sparse solve:

```
Q_in = (N @ Q_{t+1})[res] + q_prime[res]    # routed upstream + local lateral
dH = dt * (Q_in - Q_out) / (A_s + eps)
H_new = clamp(H + dH, min=OE, max=WE + D)
```

Where:

- `dt` = 3600 s (1 hour, same as MC routing timestep)
- `Q_in` = inflow to reservoir [m^3/s] (from solved upstream discharge + lateral inflow)
- `Q_out` = total outflow [m^3/s] (from `_level_pool_outflow()`)
- `A_s` = lake surface area [m^2]
- `eps` = 1e-8 (prevents division by zero for tiny lakes)
- `OE` = orifice elevation (lake bottom, lower clamp)
- `WE` = weir crest elevation (upper clamp base)
- `D` = `WE - OE` = 0.75 * depth (clamp headroom = one full depth above weir)

### Stability Clamp

The explicit forward Euler update is conditionally stable. The stability criterion is:

```
dt * dQ_out/dH / A_s < 2
```

For small reservoirs (small `A_s`) or large heads (large `dQ_out/dH`), this criterion is violated — the pool overshoots on correction, oscillates, and blows up to infinity, producing NaN through `inf - inf` in subsequent timesteps.

The pool elevation is therefore clamped after each update to the physically reasonable range `[OE, WE + D]`:

- **Lower bound** (`OE`): Pool cannot drain below the lake bottom.
- **Upper bound** (`WE + D`): Pool cannot rise more than one full depth above the weir crest. This is physically generous (no natural reservoir rises this much) but prevents the Euler instability from producing overflow values.

The clamp uses `torch.maximum` / `torch.minimum`, which have subgradient 0 at the boundary and 1 elsewhere. When pool is within bounds, gradients flow normally. When pool hits a bound, the gradient is killed — this is acceptable because the system is in an unphysical regime and there is no meaningful learning signal from further elevation changes.

This is verified by `test_no_nan_small_reservoir_large_inflow` which runs 500 timesteps with a tiny reservoir (1000 m^2) and large inflow (500 m^3/s), confirming no NaN/inf in either the forward or backward pass.

### Mass Conservation

The formulation is exactly mass-conservative when pool is within clamp bounds:

```
A_s * (H_new - H_old) = dt * (Q_in - Q_out)
```

When the clamp activates, mass is not strictly conserved — the excess (or deficit) is implicitly absorbed. This trade-off prevents numerical blowup while preserving mass conservation during normal operation.

This is verified by `test_step_mass_balance` in the test suite.

## Behavior Regimes

| Pool Elevation | Active Outlets | Behavior |
|----------------|---------------|----------|
| `H < OE` | None | Q_out = discharge_lb (near-zero) |
| `OE < H < WE` | Orifice only | Slow baseflow release |
| `H > WE` | Orifice + Weir | Rapid release (weir dominates at high head) |

### Pool Rising (Filling)

When `Q_in > Q_out`, `dH > 0` and the pool level rises. This occurs during flood events when upstream inflow exceeds the reservoir's outflow capacity.

### Pool Falling (Draining)

When `Q_out > Q_in`, `dH < 0` and the pool level drops. This occurs during dry periods when the orifice continues to drain stored water.

## Parameter Derivation

Parameters are derived from HydroLAKES-MERIT intersection data (see `references/build_reservoir_params.py`).

### Aggregation Per COMID

Multiple lakes can intersect a single COMID. Aggregation rules:

| Parameter | Aggregation | Units |
|-----------|------------|-------|
| `lake_area_m2` | Sum of Lake_area | km^2 -> m^2 |
| `depth_avg_m` | Area-weighted mean of Depth_avg | m |
| `elevation_m` | Area-weighted mean of Elevation | m |
| `dis_avg_m3s` | Sum of Dis_avg | m^3/s |
| `shore_len_m` | Sum of Shore_len | km -> m |

### Derived Parameters

| Parameter | Formula | Physical Meaning |
|-----------|---------|-----------------|
| `weir_elevation` | `elevation - 0.25 * depth` | Weir crest at 75% of full pool |
| `orifice_elevation` | `elevation - depth` | Bottom of lake |
| `weir_coeff` | 0.4 (fixed) | Broad-crested weir default |
| `weir_length` | `max(1.0, shore_len * 0.01)` | 1% of shoreline length |
| `orifice_coeff` | 0.6 (fixed) | Standard orifice default |
| `orifice_area` | `dis_avg / (C_o * sqrt(2g * 0.5 * depth) + 1e-8)` | Back-calculated from average Q |
| `initial_pool_elevation` | `elevation - 0.5 * depth` | Half-full (CSV only, not used at runtime) |

### Orifice Area Back-Calculation

The orifice area is not directly observable. It is back-calculated from the average discharge reported by HydroLAKES, assuming steady-state conditions at half-depth:

```
Q_avg = C_o * O_a * sqrt(2g * h_mid)
O_a   = Q_avg / (C_o * sqrt(2g * h_mid) + eps)
```

Where `h_mid = 0.5 * depth` (head at the midpoint).

## Integration with MC Routing (Single-Solve RHS Override)

Reservoir outflow is encoded directly into the sparse linear system so that a **single solve** produces correct reservoir outflow and propagates it to downstream reaches within the same timestep:

```
1. Compute b vector (RHS):
   b = C2*(N@Q_t) + C3*Q_t + C4*q_prime      [standard MC for all reaches]
   b[res_mask] = _level_pool_outflow(H_t)      [override reservoir rows]

2. Compute A matrix (LHS):
   c_1_ = -C1                                  [standard MC for all reaches]
   c_1_[res_mask] = 0                           [identity rows for reservoirs]
   A = I + diag(c_1_) @ N

3. Solve: A @ Q_{t+1} = b
   → Q_{t+1}[res] = b[res] = outflow           [identity row → direct assignment]
   → Q_{t+1}[downstream] uses correct outflow   [forward substitution propagates]

4. Update pool state (forward Euler with stability clamp):
   Q_in = (N @ Q_{t+1})[res] + q_prime[res]
   dH = dt * (Q_in - Q_out) / A_s
   H_{t+1} = clamp(H_t + dH, min=OE, max=WE + D)
```

**Why this works:**
- Setting `c_1_[res] = 0` makes the matrix row an identity: `A[res,:] = [0,...,1,...,0]`
- The solve produces `Q_{t+1}[res] = b[res] = outflow` directly
- Forward substitution processes rows top-down; downstream MC rows naturally use the correct reservoir outflow
- Cost: exactly 1 sparse solve per timestep (same as before), plus 1 `matmul` for inflow computation

**Why the old post-solve approach failed:**
- MC solve treated reservoirs as prismatic channels → physically meaningless discharge
- MC-solved discharge as "inflow" to level pool → inflow >> reservoir capacity → forward Euler blowup (dH = 34 m/hr) → pool elevation explosion → NaN
- Downstream reaches in the same timestep already used the wrong MC value during forward substitution (within-timestep lag)

## Pool Initialization (Equilibrium Orifice Inversion)

Pool elevation is initialized at **equilibrium** where orifice outflow matches the hotstart discharge, rather than using the HydroLAKES `initial_pool_elevation` (which was inconsistent with flow conditions and caused forward Euler blowup):

```
h_eq = Q_hotstart^2 / (2g * (C_o * A_o)^2)
H_init = min(OE + h_eq, WE)                   # capped at weir elevation
```

The hotstart discharge at reservoir rows is correct for this inversion because `compute_hotstart_discharge()` solves `(I-N) @ Q = q_prime[0]` — the same topological accumulation that the `c_1_[res]=0` override produces.

## State Management

Pool elevation (`_pool_elevation_t`) follows the same carry-state semantics as discharge (`_discharge_t`):

| Context | `carry_state` | Behavior |
|---------|--------------|----------|
| Training | `False` | Reset to equilibrium pool elevation every batch (random time windows) |
| Inference, batch 0 | `False` | Reset to equilibrium pool elevation (cold start) |
| Inference, batch i > 0 | `True` | Preserve pool elevation from previous batch |

## Differentiability

All operations use standard PyTorch ops — no custom autograd functions needed:

- `torch.clamp(H - WE, min=0)` — subgradient at boundary
- `torch.pow(head, 1.5)` — smooth for positive head
- `torch.sqrt(head + eps)` — eps prevents zero-gradient at H = OE
- `torch.maximum` / `torch.minimum` — pool elevation clamp (subgradient at boundary)
- Forward Euler is a simple linear combination

Gradients flow from the loss through:
```
loss -> Q[downstream] -> sparse solve -> b[res] = outflow -> pool_elevation_t -> previous timestep
loss -> Q[downstream] -> sparse solve -> A_values -> MC coefficients -> Manning's n -> KAN
pool_elev_{t+1} = clamp(pool_elev_t + dt*(inflow - outflow)/area)   [temporal chain through pool state]
```

## Data Source

**HydroLAKES v1.0** intersected with **MERIT Hydro Vectorized Flowlines**:

- 8 shapefiles: `RIV_lake_intersection_{71-78}.shp`
- Location: `/projects/mhpi/data/hydroLakes/merit_intersected_data/`
- Total records: 80,597
- Unique COMIDs with reservoirs: 46,095 (13.3% of CONUS MERIT reaches)

## Configuration

```yaml
params:
  use_reservoir: true

data_sources:
  reservoir_params: data/merit_reservoir_params.csv
```

Validation: `use_reservoir: true` requires `reservoir_params` to be set; otherwise a `ValueError` is raised.

## Implementation

- **Outflow physics**: `src/ddr/routing/mmc.py` -- `_level_pool_outflow()`
- **Equilibrium init**: `src/ddr/routing/mmc.py` -- `_compute_equilibrium_pool_elevation()`
- **Routing integration**: `src/ddr/routing/mmc.py` -- `MuskingumCunge.route_timestep()` (RHS override + coefficient zeroing + pool update)
- **State init**: `src/ddr/routing/mmc.py` -- `MuskingumCunge._init_pool_elevation_state()`
- **Data loading**: `src/ddr/geodatazoo/merit.py` -- `Merit._build_reservoir_tensors()`
- **Output**: `src/ddr/routing/torch_mc.py` -- `dmc.forward()` returns `pool_elevation` in output dict
- **Config**: `src/ddr/validation/configs.py` -- `params.use_reservoir`, `data_sources.reservoir_params`
- **Preprocessing**: `references/build_reservoir_params.py` -- builds CSV from HydroLAKES shapefiles
- **Tests**: `tests/routing/test_reservoir.py` -- 15 tests covering outflow physics, mass balance, gradients, sparse-solve integration, equilibrium initialization, peak attenuation, NaN stability

## Changes from Retention Module

The level pool reservoir module replaces the previous linear retention module:

| Aspect | Retention (removed) | Reservoir (current) |
|--------|-------------------|-------------------|
| Config flag | `use_retention` | `use_reservoir` |
| KAN parameter | `alpha` in [0, 1] | None (no learned parameters) |
| Physics | `Q_out = alpha * Q_t1` | Weir + orifice outflow |
| State | `_storage_t` (linear storage) | `_pool_elevation_t` (pool elevation) |
| Data requirement | None | HydroLAKES CSV |
| Physical basis | Ad hoc | Standard reservoir hydraulics |
| Integration | Post-solve override | Single-solve RHS override |
| Initialization | Fixed from CSV | Equilibrium orifice inversion |
