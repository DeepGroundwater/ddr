# Mathematical Reference: DDR Routing Equations

This document provides a complete mathematical reference for all equations implemented in `src/ddr/routing/mmc.py`. It is organized to complement the Methods section of the DDR paper (Bindas et al.), which introduces Muskingum-Cunge as graph message passing and riverbed leakage. This document adds the full derivation chain from channel hydraulics through reservoir routing.

## 1. Channel Hydraulics

### 1.1 Trapezoidal Channel Geometry

DDR models each reach as a trapezoidal channel with learned geometry. Given top width $w_t$, side slope $z$ (horizontal : vertical), and flow depth $d$:

```
         <---- w_t ---->
         _______________
        /      d       \
       /  z:1   |  z:1  \
      /---------|--------\
      <--- w_b --->
```

Bottom width:

$$w_b = w_t - 2\,z\,d$$

Cross-sectional flow area:

$$A = \frac{(w_t + w_b)\,d}{2}$$

Wetted perimeter:

$$P_w = w_b + 2\,d\,\sqrt{1 + z^2}$$

Hydraulic radius:

$$R = \frac{A}{P_w}$$

**Implementation**: `_get_trapezoid_velocity()` in `mmc.py:91-160`.

### 1.2 Depth Inversion

Flow depth is not measured directly. It is inverted from discharge $Q_t$ using the power-law depth-discharge relationship derived from Manning's equation:

$$d = \left(\frac{Q_t\, n\, (q_s + 1)}{p_s\, S_0^{1/2}}\right)^{3/(5 + 3\,q_s)}$$

Where:

- $n$ = Manning's roughness coefficient [s/m^(1/3)]
- $S_0$ = channel bed slope [m/m]
- $p_s$ = spatial parameter $p$ (a constant, default 21)
- $q_s$ = spatial parameter $q$ (learned, range [0, 1]; 0 = rectangular, 1 = triangular)

Depth is clamped to `depth_lb` = 0.01 m to prevent zero-depth singularities.

**Implementation**: `mmc.py:133-141`.

### 1.3 Manning's Equation (Velocity)

Flow velocity from Manning's equation for uniform flow in an open channel:

$$v = \frac{1}{n}\, R^{2/3}\, S_0^{1/2}$$

Velocity is clamped to the range [`velocity_lb`, 15.0] m/s. The kinematic wave celerity is then:

$$c = \frac{5}{3}\, v$$

This is the speed at which a flood wave propagates downstream. The factor 5/3 comes from the kinematic wave approximation for wide channels.

**Implementation**: `mmc.py:157-160`. The function returns $c$ (celerity), not $v$ (velocity).

## 2. Muskingum-Cunge Routing Coefficients

The Muskingum method parameterizes channel storage as a linear function of inflow and outflow via two parameters: travel time $k$ and weighting factor $x$.

### 2.1 Wave Travel Time

$$k = \frac{\Delta x}{c}$$

Where $\Delta x$ is the reach length [m] and $c$ is the kinematic wave celerity [m/s].

### 2.2 Routing Coefficients

The four Muskingum-Cunge coefficients are derived from the continuity equation discretized with the Muskingum storage relation:

$$C_1 = \frac{\Delta t - 2\,k\,x}{2\,k\,(1-x) + \Delta t}$$

$$C_2 = \frac{\Delta t + 2\,k\,x}{2\,k\,(1-x) + \Delta t}$$

$$C_3 = \frac{2\,k\,(1-x) - \Delta t}{2\,k\,(1-x) + \Delta t}$$

$$C_4 = \frac{2\,\Delta t}{2\,k\,(1-x) + \Delta t}$$

Where $\Delta t$ = 3600 s (1-hour timestep, hardcoded) and $x$ is the storage weighting factor.

**Conservation property**: $C_1 + C_2 + C_3 = 1$ always holds, ensuring mass conservation in the routing.

**Implementation**: `calculate_muskingum_coefficients()` in `mmc.py:771-796`.

### 2.3 Key Difference from Classical Muskingum

In classical Muskingum, $k$ and $x$ are calibrated constants. In DDR, $k$ is computed dynamically at every timestep from Manning's equation:

$$k = \frac{\Delta x}{c(Q_t, n, w_t, z, S_0, q_s)}$$

This makes $k$ (and therefore $C_1$-$C_4$) differentiable functions of Manning's $n$ and channel geometry, which are learned by the KAN. Gradients flow: loss $\to$ $Q_{t+1}$ $\to$ $C_i$ $\to$ $k$ $\to$ $c$ $\to$ $v$ $\to$ $n$ $\to$ KAN weights.

## 3. Matrix Routing System

### 3.1 Per-Reach Equation (GNN View)

As derived in the Methods section (Eq. 2), writing out row $i$ of the routing system:

$$Q_{i,t+1} = \sum_{j \in \mathcal{N}(i)} \left[ C_{1,i}\, I_{j,t+1} + C_{2,i}\, I_{j,t} \right] + C_{3,i}\, Q_{i,t} + C_{4,i}\, q'_{i,t}$$

The message function $\psi(j \to i) = C_{1,i}\, I_{j,t+1} + C_{2,i}\, I_{j,t}$ is the routed contribution from upstream neighbor $j$. Aggregation is summation — directly enforcing mass conservation.

### 3.2 Matrix Form

Over all $N$ reaches simultaneously (Eq. 3 from Methods):

$$\underbrace{(\mathbf{I} - \mathbf{C}_1\, \mathbf{N})}_{\mathbf{A}}\, \mathbf{Q}_{t+1} = \mathbf{C}_2\,(\mathbf{N}\,\mathbf{Q}_t) + \mathbf{C}_3\,\mathbf{Q}_t + \mathbf{C}_4\,\mathbf{q}'_t$$

Where:

- $\mathbf{Q}_t \in \mathbb{R}^N$ — discharge vector [m^3/s]
- $\mathbf{q}'_t \in \mathbb{R}^N$ — lateral inflow from land-surface model [m^3/s]
- $\mathbf{N} \in \{0,1\}^{N \times N}$ — sparse adjacency matrix
- $\mathbf{C}_1, \ldots, \mathbf{C}_4 \in \mathbb{R}^{N \times N}$ — diagonal matrices of per-reach coefficients

Because reaches are topologically ordered, $\mathbf{A}$ is lower-triangular and the system is solved via forward substitution in $O(|\mathcal{E}|)$ time.

**Implementation**: `route_timestep()` in `mmc.py:798-944`.

### 3.3 Sparse Solve and Differentiability

The forward substitution is implemented as a custom `torch.autograd.Function` (`TriangularSparseSolver` in `utils.py`). The backward pass solves the transposed system $\mathbf{A}^T \nabla_b = \nabla_{\text{out}}$ via back-substitution, and computes $\nabla_A = -\nabla_b[\text{rows}] \cdot x[\text{cols}]$ where $x$ is the (unclamped) forward solution.

## 4. Hotstart Initialization

Cold-start initialization solves for topological accumulation of lateral inflows:

$$(\mathbf{I} - \mathbf{N})\, \mathbf{Q}_0 = \mathbf{q}'_0$$

This gives each reach the sum of all upstream lateral inflows — a physically reasonable steady-state initialization. It is equivalent to the routing system with $C_1 = 1$, $C_2 = C_3 = 0$, $C_4 = 1$ (pure accumulation, no attenuation).

The result is clamped to `discharge_lb` to ensure non-negative discharge.

**Implementation**: `compute_hotstart_discharge()` in `mmc.py:42-83`.

## 5. Riverbed Leakage

As derived in the Methods section (Eq. 5), the leakance term modifies the effective lateral inflow:

$$\zeta_{i,t} = A_{b,i}\, \frac{K}{D}_i\, \left(d_{i,t} - h_{\text{bed},i} + d_{\text{gw},i,t}\right)$$

Where:

- $A_{b,i} = w_i \cdot \Delta x_i$ = wetted streambed area [m^2]
- $w_i = (p_s \cdot d_i)^{q_s}$ = power-law width from depth [m]
- $K/D$ = hydraulic exchange rate [1/s] (Cosby PTF + KAN delta correction)
- $d_{i,t}$ = flow depth from inverted Manning's equation [m]
- $h_{\text{bed},i} = w_t / (2z)$ = channel incision depth from trapezoidal geometry [m]
- $d_{\text{gw},i,t}$ = depth to water table from ground surface [m] (LSTM, time-varying)

The head difference $\Delta h = d - h_{\text{bed}} + d_{\text{gw}}$ uses the ground surface as datum. Positive $\zeta$ = losing stream; negative = gaining stream.

The modified routing equation (Eq. 6 from Methods):

$$\mathbf{A}\, \mathbf{Q}_{t+1} = \mathbf{C}_2\,(\mathbf{N}\,\mathbf{Q}_t) + \mathbf{C}_3\,\mathbf{Q}_t + \mathbf{C}_4\,(\mathbf{q}'_t - \boldsymbol{\zeta}_t)$$

An optional binary gate $g_i \in \{0, 1\}$ (learned via straight-through estimator) enables or disables leakance per reach: $\zeta_i \leftarrow g_i \cdot \zeta_i$.

**Implementation**: `_compute_zeta()` in `mmc.py:163-223`.

### 5.1 Hydraulic Conductivity (Cosby PTF + KAN Delta)

The riverbed hydraulic exchange rate $K/D$ is derived from the Cosby et al. (1984) pedotransfer function with a learned correction:

$$\log_{10}(K_s) = -0.60 + 0.0126 \cdot \text{sand\%} - 0.0064 \cdot \text{clay\%} + \log_{10}(7.056 \times 10^{-6})$$

The last term converts from inches/hr to m/s. The KAN predicts a delta correction $\delta_{K/D}$ in log-space:

$$K/D = 10^{\log_{10}(K_s) + \delta_{K/D}}$$

Where $\delta_{K/D} \in [-3, 1]$ allows the learned conductivity to span 3 orders of magnitude below to 1 order of magnitude above the Cosby prior.

## 6. Level Pool Reservoir Routing

Reservoir reaches are modeled as lumped storage nodes with physically-based outflow, integrated directly into the sparse solve via a single-solve RHS override.

### 6.1 Outflow Equations

A reservoir at pool elevation $H$ releases through two outlets:

**Weir discharge** (broad-crested weir, active when $H > H_{\text{weir}}$):

$$Q_{\text{weir}} = C_w\, W_L\, \max(H - H_{\text{weir}},\, 0)^{3/2}$$

**Orifice discharge** (active when $H > H_{\text{orifice}}$):

$$Q_{\text{orifice}} = C_o\, A_o\, \sqrt{2g\, \max(H - H_{\text{orifice}},\, 0) + \epsilon}$$

**Total outflow**:

$$Q_{\text{out}} = \max(Q_{\text{weir}} + Q_{\text{orifice}},\, Q_{\text{lb}})$$

Where:

| Symbol | Value | Description |
|--------|-------|-------------|
| $C_w$ | 0.4 | Weir discharge coefficient (broad-crested) |
| $W_L$ | $\max(1, 0.01 \cdot L_{\text{shore}})$ | Effective weir length [m] |
| $C_o$ | 0.6 | Orifice discharge coefficient |
| $A_o$ | back-calculated | Orifice cross-sectional area [m^2] |
| $g$ | 9.81 | Gravitational acceleration [m/s^2] |
| $\epsilon$ | $10^{-8}$ | Numerical stability for sqrt gradient |
| $Q_{\text{lb}}$ | $10^{-4}$ | Discharge lower bound [m^3/s] |

**Implementation**: `_level_pool_outflow()` in `mmc.py:226-274`.

### 6.2 Single-Solve RHS Override

Reservoir outflow is encoded directly into the sparse linear system so that a single solve produces correct outflow at reservoir reaches and propagates it to downstream MC reaches within the same timestep. No post-solve override or within-timestep lag.

**Step 1 — RHS override**: Replace $b$ at reservoir rows with level-pool outflow:

$$b_i = \begin{cases} C_{2,i}\, I_{i,t} + C_{3,i}\, Q_{i,t} + C_{4,i}\, (q'_{i,t} - \zeta_{i,t}) & i \notin \mathcal{R} \\ Q_{\text{out},i}(H_{i,t}) & i \in \mathcal{R} \end{cases}$$

**Step 2 — Coefficient zeroing**: Set $C_{1,i} = 0$ for reservoir rows, making $\mathbf{A}$ have identity rows at reservoir indices:

$$A_{ij} = \begin{cases} \delta_{ij} & i \in \mathcal{R} \\ \delta_{ij} - C_{1,i}\, N_{ij} & i \notin \mathcal{R} \end{cases}$$

**Step 3 — Solve**: $\mathbf{A}\, \mathbf{Q}_{t+1} = \mathbf{b}$. Forward substitution produces:

- $Q_{i,t+1} = b_i = Q_{\text{out},i}(H_{i,t})$ at reservoir rows (identity row)
- Downstream MC rows naturally use the correct reservoir outflow during forward substitution

**Step 4 — Pool update** (forward Euler with stability clamp):

$$Q_{\text{in},i} = (\mathbf{N}\, \mathbf{Q}_{t+1})_i + q'_{i,t} \qquad \text{(routed upstream + local lateral)}$$

$$\Delta H_i = \frac{\Delta t\, (Q_{\text{in},i} - Q_{\text{out},i})}{A_{s,i} + \epsilon}$$

$$H_{i,t+1} = \text{clamp}\!\left(H_{i,t} + \Delta H_i,\;\; H_{\text{orifice},i},\;\; H_{\text{weir},i} + D_i\right)$$

Where $D_i = H_{\text{weir},i} - H_{\text{orifice},i}$ (= 0.75 $\times$ depth). The clamp prevents explicit Euler instability for small reservoirs (see Section 6.5).

**Implementation**: `route_timestep()` in `mmc.py:871-942`.

### 6.3 Equilibrium Pool Initialization

Pool elevation is initialized at equilibrium where orifice outflow matches the hotstart discharge. Inverting the orifice equation:

$$Q = C_o\, A_o\, \sqrt{2g\, h} \quad \Longrightarrow \quad h_{\text{eq}} = \frac{Q_{\text{hotstart}}^2}{2g\, (C_o\, A_o)^2}$$

$$H_{\text{init}} = \min\!\left(H_{\text{orifice}} + h_{\text{eq}},\;\; H_{\text{weir}}\right)$$

The weir cap ensures the pool doesn't start above the weir crest, which would be physically unreasonable for equilibrium conditions.

The hotstart discharge at reservoir rows is correct for this inversion because `compute_hotstart_discharge()` solves $(\mathbf{I} - \mathbf{N})\, \mathbf{Q} = \mathbf{q}'_0$ — the same topological accumulation that the $C_{1,i} = 0$ override produces.

**Implementation**: `_compute_equilibrium_pool_elevation()` in `mmc.py:277-317`.

### 6.4 Orifice Area Back-Calculation

The orifice area $A_o$ is not directly observable. It is back-calculated from HydroLAKES average discharge, assuming steady-state at half-depth:

$$Q_{\text{avg}} = C_o\, A_o\, \sqrt{2g\, h_{\text{mid}}} \quad \Longrightarrow \quad A_o = \frac{Q_{\text{avg}}}{C_o\, \sqrt{2g\, h_{\text{mid}}} + \epsilon}$$

Where $h_{\text{mid}} = 0.5 \times \text{depth}$.

### 6.5 Forward Euler Stability

The explicit Euler pool update is conditionally stable. For stability, the Courant-like criterion must hold:

$$\frac{\Delta t}{A_s} \cdot \frac{\partial Q_{\text{out}}}{\partial H} < 2$$

The outflow sensitivity to pool elevation is dominated by the weir term:

$$\frac{\partial Q_{\text{out}}}{\partial H} \approx \frac{3}{2}\, C_w\, W_L\, (H - H_{\text{weir}})^{1/2} + \frac{C_o\, A_o\, g}{\sqrt{2g\,(H - H_{\text{orifice}}) + \epsilon}}$$

For small reservoirs (small $A_s$), even modest head produces a ratio $> 2$, causing the pool to overshoot on correction, oscillate with growing amplitude, and blow up to $\pm\infty$. The subsequent `inf - inf = NaN` propagates through the sparse solve to all downstream reaches.

The stability clamp (Section 6.2, Step 4) bounds pool elevation to $[H_{\text{orifice}}, H_{\text{weir}} + D]$, preventing the oscillation from producing overflow values while preserving physically meaningful dynamics within bounds.

### 6.6 Reservoir Reference Elevations

Derived from HydroLAKES fields:

| Elevation | Formula | Physical Meaning |
|-----------|---------|-----------------|
| $H_{\text{weir}}$ | `elevation - 0.25 * depth` | Weir crest at 75% of full pool |
| $H_{\text{orifice}}$ | `elevation - depth` | Lake bottom (orifice center) |
| $D$ | $H_{\text{weir}} - H_{\text{orifice}}$ | = 0.75 $\times$ depth |

### 6.7 Behavior Regimes

| Pool Elevation | Active Outlets | Behavior |
|----------------|---------------|----------|
| $H < H_{\text{orifice}}$ | None | $Q_{\text{out}} = Q_{\text{lb}}$ (near-zero) |
| $H_{\text{orifice}} < H < H_{\text{weir}}$ | Orifice only | Slow baseflow release |
| $H > H_{\text{weir}}$ | Orifice + Weir | Rapid release (weir dominates at high head) |

## 7. Gradient Flow

The full routing system is end-to-end differentiable. Gradients flow through three main paths:

**Path 1 — Manning's n (wave timing)**:
```
loss -> Q_{t+1} -> sparse solve -> A_values -> C_1
     -> k = dx/c -> c = (5/3)v -> v = (1/n) R^{2/3} S_0^{1/2} -> n -> KAN weights
```

**Path 2 — Leakance (baseflow)**:
```
loss -> Q_{t+1} -> sparse solve -> b -> C_4 * (q' - zeta)
     -> zeta = A_b * K/D * (d - h_bed + d_gw) -> d_gw -> LSTM weights
```

**Path 3 — Reservoir (storage/release)**:
```
loss -> Q_{t+1}[downstream] -> sparse solve -> b[res] = Q_out(H_t)
     -> H_t -> H_{t-1} -> ... -> H_0 (temporal chain through pool state)
```

The temporal chain through pool state creates an unrolled RNN-like gradient path. The stability clamp (Section 6.5) prevents this chain from producing exploding gradients.

## 8. State Management

| State Variable | Reset Condition | Carry Condition |
|---------------|----------------|-----------------|
| $\mathbf{Q}_t$ (discharge) | Training: every batch. Inference: batch 0. | Inference: batch $i > 0$ |
| $\mathbf{H}_t$ (pool elevation) | Training: every batch. Inference: batch 0. | Inference: batch $i > 0$ |

Reset initializes from hotstart (topological accumulation for discharge, equilibrium orifice inversion for pool elevation). Carry preserves the state from the previous batch for sequential inference.

## 9. Complete Timestep Algorithm

```
INPUT: Q_t, H_t, q'_t, network N, learned parameters {n, q_s, w_t, z, K_D_delta, d_gw, gate}

1. Depth inversion:       d = f(Q_t, n, q_s, S_0)
2. Manning's velocity:    v = (1/n) R^{2/3} S_0^{1/2}
3. Wave celerity:         c = (5/3) v
4. Muskingum coefficients: C_1...C_4 = f(dx, c, x)
5. Upstream inflow:       I_t = N @ Q_t
6. Leakance (if enabled): zeta = A_b * K/D * (d - h_bed + d_gw)
7. RHS construction:      b = C_2 * I_t + C_3 * Q_t + C_4 * (q' - zeta)

8. Reservoir override (if enabled):
   a. Compute outflow:    Q_out = level_pool_outflow(H_t)
   b. Override RHS:       b[res] = Q_out
   c. Zero coefficients:  C_1[res] = 0

9. Sparse solve:          A @ Q_{t+1} = b   (forward substitution)
10. Clamp discharge:      Q_{t+1} = max(Q_{t+1}, Q_lb)

11. Pool update (if reservoirs):
    a. Compute inflow:    Q_in = (N @ Q_{t+1})[res] + q'[res]
    b. Forward Euler:     dH = dt * (Q_in - Q_out) / A_s
    c. Stability clamp:   H_{t+1} = clamp(H_t + dH, H_orifice, H_weir + D)

OUTPUT: Q_{t+1}, H_{t+1}
```

## 10. Implementation Map

| Equation | Function | Location |
|----------|----------|----------|
| Depth inversion (1.2) | `_get_trapezoid_velocity()` | `mmc.py:133-141` |
| Manning's velocity (1.3) | `_get_trapezoid_velocity()` | `mmc.py:157-160` |
| Muskingum coefficients (2.2) | `calculate_muskingum_coefficients()` | `mmc.py:771-796` |
| Hotstart (4) | `compute_hotstart_discharge()` | `mmc.py:42-83` |
| Leakance (5) | `_compute_zeta()` | `mmc.py:163-223` |
| Cosby PTF (5.1) | `_cosby_log10_ks()` | `mmc.py:32-39` |
| Reservoir outflow (6.1) | `_level_pool_outflow()` | `mmc.py:226-274` |
| Equilibrium init (6.3) | `_compute_equilibrium_pool_elevation()` | `mmc.py:277-317` |
| Full timestep (9) | `route_timestep()` | `mmc.py:798-944` |
| Time loop (forward) | `forward()` | `mmc.py:657-754` |
| Sparse solve | `TriangularSparseSolver` | `utils.py` |
