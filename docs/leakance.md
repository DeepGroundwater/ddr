# Leakance: Groundwater-Surface Water Exchange

DDR models groundwater-surface water exchange via a **leakance term** ($\zeta$) that modifies the Muskingum-Cunge routing equation. This captures gaining and losing stream behavior driven by the hydraulic head difference between the stream surface and the regional water table.

## Physical Setup

The leakance formulation uses depth to water table from the ground surface (`d_gw`) as its groundwater state variable, following standard hydrogeology convention (Ma et al. 2026, Maxwell et al.).

```
        ground surface (datum = 0)
════════     ════════
        │   │           <-- h_bed = top_width / (2 * side_slope)
        │~~~│           <-- depth (flow depth from Manning's)
        │   │
════════╧═══╧════════   <-- channel bed
        . . . . . . .
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  <-- water table (d_gw below ground surface)
```

**Reference frame:**

- Ground surface is the datum (elevation = 0)
- Channel bed sits at elevation `-(h_bed)` below ground surface
- Stream surface sits at elevation `-(h_bed) + depth`
- Water table sits at elevation `-(d_gw)`

## Head Difference

The hydraulic head difference driving exchange is:

```
dh = stream_surface - water_table
   = (-h_bed + depth) - (-d_gw)
   = depth - h_bed + d_gw
```

Where:

- `depth` = flow depth from inverted Manning's equation [m]
- `h_bed` = channel incision depth from trapezoidal geometry [m]
- `d_gw` = depth to water table from ground surface [m]

The channel incision depth is estimated from existing trapezoidal channel geometry:

```
h_bed = top_width / (2 * side_slope)
```

## Sign Convention

| Condition | dh Sign | Zeta Sign | Stream Type |
|-----------|---------|-----------|-------------|
| Large `d_gw` (deep water table) | Positive | Positive | **Losing** (water leaves stream) |
| Small `d_gw` (shallow water table) | Negative | Negative | **Gaining** (water enters stream) |

### Losing Stream (deep water table)

```
        ground surface
════════     ════════
        │~~~│           <-- stream surface (high)
        │   │
════════╧═══╧════════   <-- channel bed
        .   .   .   .
        .   .   .   .   <-- unsaturated zone
        .   .   .   .
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  <-- water table (far below)

dh > 0  =>  zeta > 0  =>  water LOST from stream to aquifer
```

### Gaining Stream (shallow water table)

```
~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  <-- water table (at/above ground surface)
════════     ════════   <-- ground surface
        │~~~│           <-- stream surface (low)
        │   │
════════╧═══╧════════   <-- channel bed

dh < 0  =>  zeta < 0  =>  water GAINED by stream from aquifer
```

## Zeta Equation

The full leakance term computed in `_compute_zeta()`:

```
zeta = A_wetted * K_D * (depth - h_bed + d_gw)
```

Where:

- `A_wetted = width * length` = wetted streambed area [m^2]
- `width = (p_spatial * depth)^q_spatial` = power-law width [m]
- `K_D` = hydraulic exchange rate [1/s] (static, per-reach; learned by KAN)
- `d_gw` = depth to water table from ground surface [m] (time-varying, daily; learned by LSTM)

## Modified Routing Equation

The leakance term enters the Muskingum-Cunge routing as:

```
b = C2 * (N @ Q_t) + C3 * Q_t + C4 * q_prime - zeta
```

Positive zeta (losing) reduces the right-hand side, decreasing downstream discharge.
Negative zeta (gaining) increases it, adding groundwater baseflow.

## Parameter Ranges

| Parameter | Range | Log-space | Description |
|-----------|-------|-----------|-------------|
| `K_D` | [1e-8, 1e-6] | No | Hydraulic exchange rate (1/s) |
| `d_gw` | [0.01, 300.0] | Yes | Depth to water table from ground surface (m) |

The `d_gw` range spans shallow alluvial aquifers (0.01 m) to deep bedrock settings (300 m), consistent with CONUS water table depth observations (Fan et al., 2013; Maxwell et al.).

## Changes Since Song et al. (2025)

The original leakance implementation (Song et al. 2025, Eq. 12-14) defined `d_gw` as the **height of the water table above the channel bed**, with the channel bed as the reference datum:

```
Old equation:  zeta = A_wetted * K_D * (depth - d_gw)
Old range:     d_gw in [-2.0, 2.0] m  (linear space)
Old reference: channel bed = 0
```

```
        │~~~│   <-- depth (flow depth, relative to channel bed)
        │   │
════════╧═══╧════  <-- channel bed (datum = 0)
        .   .
~ ~ ~ ~ ~ ~ ~ ~ ~  <-- water table at d_gw above channel bed
```

In that formulation:

- **Positive `d_gw`** (e.g. +1.0 m) = water table above channel bed. If `d_gw > depth`, then `depth - d_gw < 0` giving a gaining stream.
- **Negative `d_gw`** (e.g. -1.0 m) = water table below channel bed. `depth - d_gw` is always positive, giving a losing stream.

### Problems

1. **Negative `d_gw` is nonsensical as a "height"** -- the parameter was used as a signed offset relative to the channel bed rather than a true physical quantity.
2. **Range [-2.0, 2.0] m is far too narrow** -- real CONUS water table depths span 0 to 300+ meters below ground surface (Fan et al. 2013, Maxwell et al.).
3. **Wrong reference frame** -- hydrogeology convention measures water table depth from the ground surface, not from the channel bed. The channel bed's position relative to the ground surface was never accounted for.

### Current formulation

The updated equation explicitly computes where the channel bed sits relative to the ground surface using `h_bed = top_width / (2 * side_slope)` from existing trapezoidal geometry, then builds the head difference from a common datum (ground surface = 0):

```
New equation:  zeta = A_wetted * K_D * (depth - h_bed + d_gw)
New range:     d_gw in [0.01, 300.0] m  (log space)
New reference: ground surface = 0
```

This makes `d_gw` a proper physical quantity with a physically meaningful range, and the sign convention follows naturally from the geometry rather than from an arbitrary signed offset.

## Implementation

- **Core math**: `src/ddr/routing/mmc.py` -- `_compute_zeta()`
- **KAN prediction**: `src/ddr/nn/kan.py` -- produces static `K_D` (per-reach) from catchment attributes
- **LSTM prediction**: `src/ddr/nn/leakance_lstm.py` -- produces daily `d_gw` from forcings + attributes
- **Config**: `src/ddr/validation/configs.py` -- `params.use_leakance`, `params.parameter_ranges`
- **Daily-to-hourly mapping**: In `MuskingumCunge.forward()`, `day_idx = (timestep - 1) // 24` (for `d_gw` only)
