# MERIT Geometry Predictions

Predicted trapezoidal channel geometry for **346,321 CONUS reaches** in the
MERIT-Hydro river network, computed from real accumulated discharge over
**water year 2000** (1999-10-01 to 2000-09-30). The NetCDF
(`merit_geometry_predictions.nc`) contains no geometries, so it can be shared
freely. Join on `COMID` to attach reach or catchment shapes from the
MERIT-Hydro shapefiles.

## Output variables

| Variable               | Units      | Description |
|------------------------|------------|-------------|
| `COMID`                | —          | MERIT-Hydro reach identifier (join key) |
| `n`                    | m^(-1/3) s | Manning's roughness (learned, bounds [0.015, 0.25]) |
| `p_spatial`            | —          | Leopold & Maddock width coefficient (learned, bounds [1, 200], log-space) |
| `q_spatial`            | —          | Width-depth exponent (learned, bounds [0, 1]; 0 = rectangular, 1 = triangular) |
| `slope`                | m/m        | Channel bed slope |
| `depth_{min,max,median,mean}` | m   | Flow depth statistics across WY2000 |
| `top_width_{min,max,median,mean}` | m | Water-surface width statistics across WY2000 |
| `discharge_{min,max,median,mean}` | m^3/s | Accumulated discharge statistics across WY2000 |

Each `_min/_max/_median/_mean` suffix is a per-reach temporal statistic
computed from 366 daily values across the water year.

## Discharge source

Discharge is computed from real lateral inflows (Q' from the dHBV2
unit-hydrograph retrospective) accumulated through the MERIT network topology
using `compute_hotstart_discharge`. This solves `(I - N) @ Q = Q'` per day,
where N is the adjacency matrix, giving each reach the sum of all upstream
lateral inflows.

The KAN parameters (n, p, q) are discharge-independent — they depend only on
catchment attributes. Recompute geometry for a different flow scenario by
calling `compute_trapezoidal_geometry` with new Q values.

## How depth is derived from n, p, q

The key equation inverts Manning's formula for a trapezoidal channel with
Leopold & Maddock scaling (q' = q + 1e-6):

```
depth  =  [ Q * n * (q' + 1) / (p * sqrt(S_0)) ] ^ [ 3 / (5 + 3q') ]
```

Once depth (d) is known:

```
top_width  = p * d^q'
```

```
      ◄──────── top_width ────────►
      ┌──────────────────────────────┐  water surface
     ╱              area              ╲   depth
    ╱                                  ╲
   └────────────────────────────────────┘  bed
    ◄─────── bottom_width ──────────►
```

See `geometry_predictor.ipynb` for the full workflow.
