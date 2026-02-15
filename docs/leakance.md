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
- `K_D` = hydraulic exchange rate [1/s] (learned by LSTM)
- `d_gw` = depth to water table from ground surface [m] (learned by LSTM)

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

## Implementation

- **Core math**: `src/ddr/routing/mmc.py` -- `_compute_zeta()`
- **LSTM prediction**: `src/ddr/nn/leakance_lstm.py` -- produces daily `K_D` and `d_gw` from forcings + attributes
- **Config**: `src/ddr/validation/configs.py` -- `params.use_leakance`, `params.parameter_ranges`
- **Daily-to-hourly mapping**: In `MuskingumCunge.forward()`, `day_idx = (timestep - 1) // 24`
