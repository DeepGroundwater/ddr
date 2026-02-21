# Level Pool Reservoir Routing

DDR models reservoir storage and release via a **level pool** formulation that replaces Muskingum-Cunge routing on reaches intersecting HydroLAKES waterbodies. Reservoir reaches are treated as lumped storage nodes with weir + orifice outflow, while all other reaches continue to use standard MC routing.

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
- Initial pool elevation at half-full: `H_init = elevation - 0.5 * depth`

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

Forward Euler timestep with mass balance:

```
dH = dt * (Q_in - Q_out) / (A_s + eps)
H_new = H + dH
```

Where:

- `dt` = 3600 s (1 hour, same as MC routing timestep)
- `Q_in` = inflow to reservoir [m^3/s] (MC-routed discharge arriving at this reach)
- `Q_out` = total outflow [m^3/s]
- `A_s` = lake surface area [m^2]
- `eps` = 1e-8 (prevents division by zero for tiny lakes)

### Mass Conservation

The formulation is exactly mass-conservative:

```
A_s * (H_new - H_old) = dt * (Q_in - Q_out)
```

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
| `initial_pool_elevation` | `elevation - 0.5 * depth` | Half-full initial condition |

### Orifice Area Back-Calculation

The orifice area is not directly observable. It is back-calculated from the average discharge reported by HydroLAKES, assuming steady-state conditions at half-depth:

```
Q_avg = C_o * O_a * sqrt(2g * h_mid)
O_a   = Q_avg / (C_o * sqrt(2g * h_mid) + eps)
```

Where `h_mid = 0.5 * depth` (head at the midpoint, consistent with the half-full initial condition).

## Integration with MC Routing

Reservoir routing is applied **after** the MC solve at each timestep:

```
1. MC solve: (I - C1*N) @ Q_{t+1} = C2*(N@Q_t) + C3*Q_t + C4*q_prime  [all reaches]
2. Reservoir override:
   - For reservoir reaches (mask=True):
     Q_out, H_new = level_pool_step(Q_{t+1}[mask], H_t[mask], params...)
     Q_{t+1}[mask] = Q_out       # Replace MC discharge with reservoir outflow
     H_t[mask] = H_new           # Update pool elevation state
   - For non-reservoir reaches: unchanged
```

Key implementation detail: `Q_{t+1}.clone()` and `H_t.clone()` are called before indexed assignment to preserve the autograd graph for backpropagation.

## State Management

Pool elevation (`_pool_elevation_t`) follows the same carry-state semantics as discharge (`_discharge_t`):

| Context | `carry_state` | Behavior |
|---------|--------------|----------|
| Training | `False` | Reset to `initial_pool_elevation` every batch (random time windows) |
| Inference, batch 0 | `False` | Reset to `initial_pool_elevation` (cold start) |
| Inference, batch i > 0 | `True` | Preserve pool elevation from previous batch |

## Differentiability

All operations use standard PyTorch ops — no custom autograd functions needed:

- `torch.clamp(H - WE, min=0)` — subgradient at boundary
- `torch.pow(head, 1.5)` — smooth for positive head
- `torch.sqrt(head + eps)` — eps prevents zero-gradient at H = OE
- Forward Euler is a simple linear combination

Gradients flow from the loss through:
```
loss -> daily_runoff -> hourly Q -> Q[res_mask] = Q_out -> _level_pool_step
     -> outflow -> weir/orifice equations -> pool_elevation
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

- **Physics functions**: `src/ddr/routing/mmc.py` -- `_level_pool_outflow()`, `_level_pool_step()`
- **Routing integration**: `src/ddr/routing/mmc.py` -- `MuskingumCunge.forward()` (reservoir block after MC solve)
- **State init**: `src/ddr/routing/mmc.py` -- `MuskingumCunge._init_pool_elevation_state()`
- **Data loading**: `src/ddr/geodatazoo/merit.py` -- `Merit._build_reservoir_tensors()`
- **Output**: `src/ddr/routing/torch_mc.py` -- `dmc.forward()` returns `pool_elevation` in output dict
- **Config**: `src/ddr/validation/configs.py` -- `params.use_reservoir`, `data_sources.reservoir_params`
- **Preprocessing**: `references/build_reservoir_params.py` -- builds CSV from HydroLAKES shapefiles
- **Tests**: `tests/routing/test_reservoir.py` -- 11 tests covering physics, gradients, mass balance, integration

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
