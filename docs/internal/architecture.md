# DDR Internal Architecture Guide

This document is the internal technical reference for the Distributed Differentiable Routing (DDR) codebase. It explains the physics, the code structure, how SWOT / references fit in, and how models are trained and tested.

---

## Table of Contents

1. [What DDR Does](#what-ddr-does)
2. [Repository Layout](#repository-layout)
3. [The Physics: Muskingum-Cunge Routing](#the-physics-muskingum-cunge-routing)
4. [Neural Network Modules](#neural-network-modules)
5. [SWOT, SWORD, and the References Directory](#swot-sword-and-the-references-directory)
6. [Geospatial Datasets (geodatazoo)](#geospatial-datasets)
7. [Data I/O Pipeline](#data-io-pipeline)
8. [Training Pipeline](#training-pipeline)
9. [Testing Pipeline](#testing-pipeline)
10. [Configuration System](#configuration-system)
11. [Loss Functions and Metrics](#loss-functions-and-metrics)
12. [Key Equations Reference](#key-equations-reference)

---

## What DDR Does

DDR is an **end-to-end differentiable river routing framework**. It solves the problem: *given lateral inflow predictions (Q') at every unit catchment, route water downstream through a river network and learn the optimal physical parameters from observed streamflow at USGS gauges.*

The key innovation is that the entire pipeline — from catchment attributes through a neural network, through the Muskingum-Cunge physics equations, to the routed discharge — is differentiable. Gradients flow backward through the routing physics into the neural network, so the NN learns physical parameters (Manning's roughness, channel geometry) solely from downstream streamflow observations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DDR End-to-End Pipeline                         │
│                                                                        │
│  Catchment Attributes        Lateral Inflow (Q')                       │
│  (glaciers, aridity,         (from dHBV2.0 or                          │
│   elevation, precip,          other land model)                        │
│   upstream area)                   │                                   │
│        │                           │                                   │
│        ▼                           ▼  (optional)                       │
│  ┌──────────┐              ┌──────────────┐                            │
│  │ Spatial   │              │ Temporal     │                            │
│  │ KAN       │              │ Phi-KAN      │                            │
│  │ (learns n,│              │ (bias correct│                            │
│  │  q, p)    │              │  Q' → Q'*)   │                            │
│  └─────┬─────┘              └──────┬───────┘                            │
│        │ [0,1] → physical          │                                   │
│        │ bounds via                 │                                   │
│        │ denormalization            │                                   │
│        ▼                           ▼                                   │
│  ┌─────────────────────────────────────────────┐                       │
│  │         Muskingum-Cunge Router (dMC)        │                       │
│  │                                             │                       │
│  │  For each timestep t:                       │                       │
│  │    1. Manning's eq → velocity               │                       │
│  │    2. Muskingum coefficients (c1,c2,c3,c4)  │                       │
│  │    3. Sparse triangular solve               │                       │
│  │       (I - c1*N) Q(t+1) = b(t)             │                       │
│  └──────────────────┬──────────────────────────┘                       │
│                     │                                                  │
│                     ▼                                                  │
│            Routed Discharge Q(t) at gauges                             │
│                     │                                                  │
│                     ▼                                                  │
│         Loss(Q_pred, Q_obs)  ← USGS observations                      │
│                     │                                                  │
│                     ▼                                                  │
│         ∂Loss/∂θ via autograd  ───────────────► Update KAN + φ-KAN     │
│         (gradients flow through                  weights               │
│          MC routing equations)                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Layout

```
ddr/
├── src/ddr/                       # Core library
│   ├── routing/
│   │   ├── mmc.py                 # MuskingumCunge class — all routing math
│   │   ├── torch_mc.py            # dmc — PyTorch nn.Module wrapper
│   │   └── utils.py               # PatternMapper, sparse solvers, denormalize
│   ├── nn/
│   │   ├── kan.py                 # Spatial KAN (predicts n, q_spatial, p_spatial)
│   │   └── temporal_phi_kan.py    # Temporal φ-KAN (bias correction of Q')
│   ├── geodatazoo/
│   │   ├── base_geodataset.py     # Abstract base class (torch Dataset)
│   │   ├── merit.py               # MERIT Hydro dataset implementation
│   │   ├── lynker_hydrofabric.py  # Lynker Hydrofabric v2.2 implementation
│   │   └── dataclasses.py         # Gauge, Dates, RoutingDataclass
│   ├── io/
│   │   ├── readers.py             # COO matrices, zarr stores, USGS observations
│   │   ├── builders.py            # Network graph construction (rustworkx)
│   │   ├── statistics.py          # Normalization statistics (attribute/streamflow)
│   │   └── functions.py           # Downsampling (hourly → daily)
│   ├── validation/
│   │   ├── configs.py             # Pydantic config classes (Config, Params, etc.)
│   │   ├── enums.py               # Mode, PhiInputs, BiasLossFn, GeoDataset
│   │   ├── losses.py              # mass_balance_loss, kge_loss, huber_loss
│   │   ├── metrics.py             # NSE, KGE, RMSE, FDC, etc.
│   │   └── plots.py               # Visualization utilities
│   ├── scripts_utils.py           # Checkpoint loading, LR schedule, downsampling
│   └── __init__.py                # Package exports: dmc, kan, streamflow, etc.
│
├── scripts/
│   ├── train.py                   # Training loop (Hydra entry point)
│   ├── test.py                    # Evaluation on held-out periods
│   ├── router.py                  # Full-network routing (all catchments)
│   └── summed_q_prime.py          # Pre-routing Q' diagnostic
│
├── config/                        # Hydra YAML configs
│   ├── example_config.yaml
│   ├── training_config.yaml
│   ├── test_config.yaml
│   └── router_config.yaml
│
├── references/                    # Gage-to-river mapping and SWOT geometry
│   ├── gage_info/                 # CSV files (GAGES-II, CAMELS, gages_3000, dhbv2)
│   ├── geo_io/                    # Scripts to build gage references + SWOT geometry
│   ├── analysis/                  # Drainage area mismatch diagnostics
│   └── dhbv2_merit/               # dHBV2.0 lateral inflow download/conversion
│
├── engine/                        # ddr-engine: geospatial data prep
├── benchmarks/                    # ddr-benchmarks: comparison against T-Route etc.
├── examples/                      # Jupyter notebooks (canonical + scratch)
├── data/                          # Adjacency matrices, statistics, diagrams
├── model/                         # Pre-trained checkpoints
└── tests/                         # Unit tests
```

---

## The Physics: Muskingum-Cunge Routing

### What Is River Routing?

A land surface model (e.g. dHBV2.0) predicts how much rainfall becomes runoff at each small catchment (unit catchment). This runoff enters the river network as **lateral inflow** (Q'). But water doesn't appear instantly at a downstream gauge — it propagates as a flood wave through the channel network, attenuated and delayed by the channel geometry and roughness.

**Routing** simulates this wave propagation. The Muskingum-Cunge method is a well-established, computationally efficient approach that models each river segment as a prism of storage with parameters controlling wave speed and attenuation.

### Manning's Equation

The fundamental relationship between flow and channel properties:

```
v = (1/n) × R^(2/3) × S^(1/2)
```

Where:
- `v` = flow velocity (m/s)
- `n` = Manning's roughness coefficient (dimensionless, **learned by KAN**)
- `R` = hydraulic radius = cross-sectional area / wetted perimeter (m)
- `S` = channel slope (m/m, from geospatial attributes)

Manning's n is the primary parameter the neural network learns. It encodes the friction of the channel bed — smooth concrete channels have n ≈ 0.013, rocky mountain streams n ≈ 0.05+.

### Trapezoidal Channel Geometry

DDR models channel cross-sections as trapezoids using the **at-a-station hydraulic geometry** power law (Leopold & Maddock, 1953):

```
top_width = p × depth^q
```

Where:
- `p` (p_spatial) = width coefficient (default 21, optionally **learned by KAN**)
- `q` (q_spatial) = width-depth exponent (**learned by KAN**)
  - q = 0 → rectangular channel (width independent of depth)
  - q = 1 → triangular channel (width proportional to depth)

From the discharge and these parameters, DDR solves for depth:

```
depth = [ (Q × n × (q+1)) / (p × S^0.5) ] ^ (3 / (5 + 3q))
```

Then derives the full trapezoid geometry:

```
                      ← top_width →
                     ╱              ╲
                    ╱                ╲
        side_slope ╱                  ╲ side_slope
                  ╱                    ╲
                 ╱                      ╲
                 ←── bottom_width ──────→
                          ↕ depth

    side_slope = top_width × q / (2 × depth)
    bottom_width = top_width - 2 × side_slope × depth
    area = (top_width + bottom_width) × depth / 2
    wetted_perimeter = bottom_width + 2 × depth × sqrt(1 + side_slope²)
    R = area / wetted_perimeter
```

### SWOT / Lynker Data Override

When observed channel geometry is available (from SWOT satellite or Lynker Hydrofabric), DDR overrides the power-law-derived values:

- **Full override**: observed data has no NaN → use observed (Lynker full coverage)
- **Partial override**: observed data has some NaN → blend observed + derived (SWOT partial coverage)
- **No data**: all NaN or None → use power-law derived values (MERIT without SWOT)

This is implemented in `_apply_data_override()` in `mmc.py`.

### Celerity and Muskingum Coefficients

From Manning's velocity, DDR computes the wave **celerity** (speed at which the flood wave front moves):

```
celerity = v × (5/3)
```

The factor 5/3 comes from kinematic wave theory for wide channels.

The Muskingum routing coefficients control how the flood wave is routed through each segment:

```
K = length / celerity          (travel time through segment)
Δt = 3600 s                    (1-hour timestep)

c1 = (Δt - 2Kx) / (2K(1-x) + Δt)     — weight on future upstream inflow
c2 = (Δt + 2Kx) / (2K(1-x) + Δt)     — weight on current upstream inflow
c3 = (2K(1-x) - Δt) / (2K(1-x) + Δt) — weight on current outflow
c4 = 2Δt / (2K(1-x) + Δt)            — weight on lateral inflow (Q')
```

Where `x` is the Muskingum storage coefficient (from geospatial data, not learned).

### Network Solve (Each Timestep)

At each timestep, DDR solves a lower-triangular sparse linear system:

```
(I - c1 × N) × Q(t+1) = c2 × (N × Q(t)) + c3 × Q(t) + c4 × Q'(t)
                         └─────────────── b(t) ─────────────────────┘
```

Where:
- `I` = identity matrix
- `N` = sparse adjacency matrix (network topology — which reach flows into which)
- `Q(t)` = discharge at all segments at time t
- `Q'(t)` = lateral inflow at time t

The system is lower-triangular because reaches are topologically sorted (upstream before downstream), so it can be solved efficiently with a forward substitution (no matrix inversion needed).

### Hot-Start Initialization

Before the first timestep, DDR initializes discharge by solving:

```
(I - N) × Q(0) = Q'(0)
```

This gives each segment the accumulated sum of all upstream lateral inflows at t=0 — a physically reasonable starting state.

---

## Neural Network Modules

### Spatial KAN (`src/ddr/nn/kan.py`)

The Spatial KAN predicts physical routing parameters from static catchment attributes.

**Architecture:**
```
Input (n_features)
    │
    ▼
Linear Layer → hidden_size      (Kaiming init)
    │
    ▼
KAN Hidden Layer(s)             (B-spline activations, pykan library)
    │
    ▼
Linear Layer → n_parameters     (Xavier init, small gain)
    │
    ▼
Sigmoid → [0, 1]                (then denormalized to physical bounds)
    │
    ▼
Dict: {"n": tensor, "q_spatial": tensor, ...}
```

**Input features** (typical): `glaciers, aridity, meanelevation, meanP, log_uparea`

**Output parameters** (typical): `n` (Manning's roughness), `q_spatial` (width-depth exponent), optionally `p_spatial` (width coefficient)

The sigmoid output is denormalized to physical bounds:
- `n`: [0.02, 0.2] (smooth to rough channels)
- `q_spatial`: [0, 1] (rectangular to triangular)
- `p_spatial`: [1, 100] (narrow to wide)

Log-space denormalization is available for parameters with skewed distributions.

### Temporal φ-KAN (`src/ddr/nn/temporal_phi_kan.py`)

The Temporal φ-KAN learns an **additive bias correction** to the lateral inflow Q':

```
Q'_corrected = Q' + δ(z, context)
```

Where `z = (Q' - μ) / σ` is the z-score-normalized Q' and context depends on the mode:

| Mode     | Input Dim | Inputs                    |
|----------|-----------|---------------------------|
| STATIC   | 1         | z only                    |
| MONTHLY  | 3         | z, sin(2π·month/12), cos(2π·month/12) |
| FORCING  | 2         | z, forcing variable       |
| RANDOM   | 3         | z, rand1, rand2           |

**Key design choices:**
- **Residual connection**: At initialization the KAN outputs ≈ 0, so `Q'_corrected ≈ Q'`. The network only needs to learn the correction δ, not the full signal.
- **Gradient checkpointing**: The flat (T×N) input is processed in chunks of 8192 with `torch.utils.checkpoint` to save ~4 GiB of GPU memory.
- **Interpretability**: By Kolmogorov-Arnold theory, each learned function is a 1D B-spline curve that can be plotted and inspected.

**Why bias correction?**
The lateral inflow Q' from the land model (dHBV2.0) has systematic biases — e.g., consistently overestimating runoff in arid regions or underestimating snowmelt timing. The φ-KAN corrects these biases before routing so that the routing parameters (n, q) don't have to compensate for upstream errors. This breaks equifinality: φ owns volume (AUC), MC owns timing.

---

## SWOT, SWORD, and the References Directory

### What These Acronyms Mean

- **SWOT** (Surface Water and Ocean Topography) — NASA/CNES satellite launched Dec 2022. Measures river width, water surface elevation, and slope from space using a Ka-band radar interferometer. Coverage is partial: the satellite has a 21-day repeat cycle and only observes rivers wider than ~50-100m.

- **SWORD** (SWOT River Database) v16 — A pre-mission database of river reaches worldwide with prior geometry (width, max width, slope, water surface elevation) derived from Landsat and other sources. Complete coverage for all mapped rivers, not just those observed by SWOT.

- **EIV fits** (Errors-In-Variables) — Statistical fits that estimate channel cross-section shape (side slope) from SWOT observations. Only available for reaches that SWOT has actually observed (partial coverage).

### How References Work

The `references/` directory contains:

1. **Gage reference CSVs** (`references/gage_info/`) — Mapping USGS streamflow gauges to MERIT Hydro COMIDs (river reach identifiers). Key columns:
   - `STAID` — USGS station ID (zero-padded 8 digits)
   - `COMID` — MERIT river reach this gauge is assigned to
   - `DRAIN_SQKM` — Observed drainage area from USGS
   - `COMID_DRAIN_SQKM` — Modeled drainage area from MERIT
   - `ABS_DIFF` — Absolute difference between the two (used to filter bad matches)
   - `FLOW_SCALE` — Scaling factor for partial-catchment gauges (when gauge drainage area ≠ COMID drainage area)

2. **SWOT geometry builder** (`references/geo_io/build_swot_geometry.py`) — Joins SWORD v16 reach geometry to MERIT COMIDs via a crosswalk table:
   ```
   SWORD v16 reaches  ──┐
                         ├── crosswalk ──► MERIT COMIDs with top_width, side_slope
   SWOT EIV fits     ──┘
   ```
   - `top_width` comes from SWORD (near-complete coverage)
   - `side_slope` comes from SWOT EIV fits (partial coverage, NaN elsewhere)
   - When multiple SWORD reaches map to one COMID, the one with the largest flow accumulation (main channel) wins

3. **dHBV2.0 lateral inflow** (`references/dhbv2_merit/`) — Notebooks to download and convert the pre-computed lateral inflow predictions that DDR routes.

### Data Flow: SWOT → Routing

```
SWORD v16 (all NA reaches)
    │
    ├── width → top_width (clamped [1, 5000] m)
    │
    └── EIV fits (SWOT-observed reaches only)
            │
            └── m_1 / 2 → side_slope (clamped [0.5, 50], NaN if unobserved)
                    │
                    ▼
        ┌─────────────────────────┐
        │  swot_merit_geometry.nc │  (COMID-indexed NetCDF)
        └───────────┬─────────────┘
                    │
                    ▼
           RoutingDataclass
           .top_width, .side_slope
                    │
                    ▼
         _get_trapezoid_velocity()
           ├── has data? → override derived value
           └── NaN?      → use power-law derived value
```

---

## Geospatial Datasets

DDR supports two river network datasets, abstracted behind a common `BaseGeoDataset` interface:

### MERIT Hydro (`geodatazoo/merit.py`)

- Global vector-based river network
- ~37 km² average catchment resolution over CONUS
- Uses COMID as reach identifier
- Primary dataset for national-scale DDR experiments

### Lynker Hydrofabric v2.2 (`geodatazoo/lynker_hydrofabric.py`)

- NOAA-OWP river network for CONUS
- Complete top_width and side_slope from Lynker (no NaN → full data override)
- Different attribute naming conventions

### RoutingDataclass

The core data container passed through the pipeline:

```python
@dataclass
class RoutingDataclass:
    adjacency_matrix: torch.Tensor      # Sparse COO network topology
    spatial_attributes: xr.Dataset      # Raw catchment attributes
    normalized_spatial_attributes: Tensor  # Z-scored for KAN input
    length: Tensor                      # Channel length per segment
    slope: Tensor                       # Channel slope per segment
    top_width: Tensor | None            # SWOT/Lynker observed (NaN = no data)
    side_slope: Tensor | None           # SWOT/Lynker observed (NaN = no data)
    x: Tensor                           # Muskingum storage coefficient
    observations: xr.Dataset | None     # USGS gauge streamflow
    divide_ids: list                    # Catchment identifiers
    outflow_idx: list[np.ndarray]       # Ragged array: which segments drain to each gauge
    gage_catchment: list[str]           # Gauge IDs
    flow_scale: Tensor | None           # Partial-catchment scaling factors
```

### Dates Class

Manages time indexing for both training (random rho-day windows) and testing (sequential windows):

- **Training**: Randomly samples a `rho`-day window per batch using a seeded generator. Creates both daily and hourly time ranges for the window.
- **Testing**: Creates sequential non-overlapping windows covering the full evaluation period.
- **Monthly tensors**: Pre-computes sin/cos of month for the φ-KAN.

---

## Data I/O Pipeline

### Data Sources

| Data | Format | Reader |
|------|--------|--------|
| Network adjacency | zarr (sparse COO) | `read_coo()` |
| Catchment attributes | xarray/zarr Icechunk | `_load_attributes()` |
| Lateral inflow (Q') | xarray/zarr Icechunk | `streamflow` class |
| USGS observations | Icechunk | `IcechunkUSGSReader` |
| SWOT geometry | NetCDF | Standard xarray |
| Meteorological forcings | Icechunk | `ForcingsReader` |

### Network Construction

For each training batch (set of gauges):

1. Load full CONUS adjacency matrix (sparse COO)
2. Build directed graph with `rustworkx`
3. For each gauge in the batch, find all upstream reaches
4. Take the union of all upstream reach sets
5. Extract the subnetwork adjacency matrix
6. Topologically sort reaches (upstream first)
7. Return as `RoutingDataclass`

### Normalization

- **Catchment attributes**: Z-score normalized per feature using dataset-wide statistics (cached after first computation)
- **Lateral inflow (Q')**: Per-basin z-score for φ-KAN input using pre-computed mean/std
- **Forcings**: Similarly normalized per-variable

---

## Training Pipeline

Entry point: `scripts/train.py` → `train()` function

### Training Loop (per epoch, per mini-batch)

```
1. Sample batch of gauges via DataLoader
       │
       ▼
2. Collate → RoutingDataclass
   (build subnetwork, load attributes, observations)
       │
       ▼
3. Forward KAN
   attributes (N, n_features) → spatial_params {"n": (N,), "q_spatial": (N,), ...}
       │
       ▼
4. (Optional) Forward φ-KAN on daily Q'
   Q'_daily (T_d, N) → Q'_corrected_daily (T_d, N)
   Then interpolate daily → hourly via repeat_interleave(24)
       │
       ▼
5. Forward dMC routing
   (network, Q', spatial_params) → routed discharge (G, T_h)
   where G = # gauges, T_h = # hourly timesteps
       │
       ▼
6. Downsample hourly → daily
   routed discharge (G, T_h) → daily discharge (G, T_d)
       │
       ▼
7. Filter NaN observations, slice off warmup days
       │
       ▼
8. Compute loss:
   Without φ-KAN: huber_loss(pred, obs)
   With φ-KAN:    λ × mass_balance_loss + (1-λ) × routing_loss
       │
       ▼
9. loss.backward()
   Gradients flow: loss → downsample → MC routing → KAN
                                                  → φ-KAN
       │
       ▼
10. Gradient clipping (max_norm=1.0) + optimizer.step()
       │
       ▼
11. Log metrics (NSE, RMSE, KGE), save checkpoint, plot validation
```

### Key Training Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 64 | Number of gauges per mini-batch |
| `epochs` | 5 | Total training epochs |
| `rho` | 365 | Training window length (days) |
| `warmup` | 3 | Burn-in days before loss computation |
| `learning_rate` | {1: 0.005, 3: 0.001} | Per-epoch LR schedule |
| `lambda_mass` | 0.5 | Weight for mass balance vs. routing loss |

### Why Warmup?

The first few days of routing use cold-start initialization. The discharge state hasn't converged to the physically correct profile yet, so including those days in the loss would inject noise. `warmup` days are routed but excluded from loss computation.

---

## Testing Pipeline

Entry point: `scripts/test.py`

Testing is similar to training but:
- No gradient computation (`torch.no_grad()`)
- Sequential (non-overlapping) time windows instead of random sampling
- `carry_state=True` — discharge state carries across windows for physical continuity
- Computes full metric suite (NSE, KGE, RMSE, FDC RMSE, etc.)
- Saves predictions to `model_test.zarr`

### Routing Mode

`scripts/router.py` routes over all target catchments (not just gauge locations). Used for producing wall-to-wall CONUS discharge estimates. Saves to `chrout.zarr`.

### Pre-Routing Diagnostic

`scripts/summed_q_prime.py` evaluates the lateral inflow Q' before routing. This helps diagnose whether poor performance is due to the land model (bad Q') or the routing (bad parameters).

---

## Configuration System

DDR uses [Hydra](https://hydra.cc/) for configuration management with Pydantic validation.

### Config Hierarchy

```yaml
mode: training              # training | testing | routing
geodataset: merit           # merit | lynker_hydrofabric
device: 7                   # GPU index or "cpu"
seed: 42

data_sources:
  attributes: <icechunk store path>
  conus_adjacency: ./data/hydrofabric_v2.2_conus_adjacency.zarr
  gages_adjacency: ./data/hydrofabric_v2.2_gages_conus_adjacency.zarr
  streamflow: <lateral inflow icechunk path>
  observations: <USGS observations icechunk path>
  gages: ./references/gage_info/dhbv2_gages.csv
  swot_geometry: ./data/swot_merit_geometry.nc  # optional

experiment:
  batch_size: 64
  epochs: 5
  learning_rate: {1: 0.005, 3: 0.001}
  rho: 365
  warmup: 3
  max_area_diff_sqkm: 50    # filter bad gage-to-COMID matches

kan:
  hidden_size: 11
  num_hidden_layers: 1
  input_var_names: [glaciers, aridity, meanelevation, meanP, log_uparea]
  learnable_parameters: [n, q_spatial]
  grid: 3
  k: 3

bias:                        # optional φ-KAN bias correction
  enabled: false
  phi_inputs: MONTHLY
  phi_hidden_size: 3
  lambda_mass: 0.5
  loss_fn: HUBER

params:
  attribute_minimums:
    discharge: 0.001
    slope: 0.00001
    velocity: 0.1
    depth: 0.001
    bottom_width: 0.05
  parameter_ranges:
    n: [0.02, 0.2]
    q_spatial: [0.0, 1.0]
    p_spatial: [1.0, 100.0]
```

---

## Loss Functions and Metrics

### Training Losses (`validation/losses.py`)

**Huber Loss** (default without φ-KAN):
```
L = mean over (G, T):
    |e| < δ  → 0.5 × e²
    |e| ≥ δ  → δ × (|e| - 0.5δ)
```
Quadratic for small errors (strong gradients), linear for large errors (robust to flood outliers).

**Mass Balance Loss** (φ-KAN direct gradient):
```
L = mean over G: |Σ_t pred(g,t) - Σ_t obs(g,t)|
```
Total volume at gauge must equal total volume injected upstream. Gives φ-KAN gradients that bypass the MC routing entirely (MC conserves mass, so total routed volume = total input volume).

**KGE Loss** (optional routing loss):
```
KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
  r = correlation between pred and obs
  α = σ_pred / σ_obs  (variability ratio)
  β = μ_pred / μ_obs  (bias ratio)
Loss = mean over G: sqrt((r-1)² + (α-1)² + (β-1)² + ε)
```

**Combined Loss** (with φ-KAN):
```
L = λ × mass_balance_loss + (1-λ) × routing_loss
```
Where routing_loss is Huber, KGE, or MSE depending on `bias.loss_fn`.

### Evaluation Metrics (`validation/metrics.py`)

| Metric | Formula | Range | Perfect |
|--------|---------|-------|---------|
| NSE | 1 - Σ(pred-obs)² / Σ(obs-μ_obs)² | (-∞, 1] | 1 |
| KGE | 1 - sqrt((r-1)² + (α-1)² + (β-1)²) | (-∞, 1] | 1 |
| RMSE | sqrt(mean((pred-obs)²)) | [0, ∞) | 0 |
| PBIAS | 100 × Σ(pred-obs) / Σ(obs) | (-∞, ∞) | 0 |
| FDC RMSE | RMSE of flow duration curves | [0, ∞) | 0 |
| FLV | Volume error at low flows (Q < Q70) | (-∞, ∞) | 0 |
| FHV | Volume error at high flows (Q > Q2) | (-∞, ∞) | 0 |

---

## Key Equations Reference

### Manning's Equation
```
v = (1/n) × R^(2/3) × S^(1/2)
```

### Hydraulic Geometry Power Law
```
top_width = p × depth^q
```

### Depth from Discharge
```
depth = [ (Q × n × (q+1)) / (p × S^0.5) ] ^ (3 / (5 + 3q))
```

### Celerity
```
c = v × (5/3)
```

### Muskingum Coefficients
```
K = length / celerity
c1 = (Δt - 2Kx) / (2K(1-x) + Δt)
c2 = (Δt + 2Kx) / (2K(1-x) + Δt)
c3 = (2K(1-x) - Δt) / (2K(1-x) + Δt)
c4 = 2Δt / (2K(1-x) + Δt)
```

### Network Routing (per timestep)
```
(I - c1·N) × Q(t+1) = c2·(N·Q(t)) + c3·Q(t) + c4·Q'(t)
```

### φ-KAN Bias Correction
```
Q'_corrected = (z + δ(z, context)) × σ + μ     where z = (Q' - μ) / σ
```

### Combined Training Loss
```
L = λ × |Σ_t Q'_corrected - Σ_t Q_obs| + (1-λ) × Huber(Q_routed, Q_obs)
```
