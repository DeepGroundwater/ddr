# DDR — Distributed Differentiable Routing

AI-agent context file. Committed to version control so every coding assistant
(Claude, Copilot, Cursor, etc.) gets the same codebase orientation.

---

## Architecture

DDR couples a **Kolmogorov-Arnold Network (KAN)** with **differentiable
Muskingum-Cunge (MC) routing** to learn spatially varying river-routing
parameters end-to-end via PyTorch autograd.

1. **KAN** ingests normalized catchment attributes → sigmoid outputs in [0,1]
   → denormalized to physical parameter bounds (Manning's *n*, *q_spatial*,
   optionally *p_spatial*).
2. **Leopold & Maddock power law** converts discharge to channel geometry:
   `top_width = p_spatial * depth^(q_spatial + 1e-6)`.
3. **MC routing** solves the linearized Saint-Venant equations on a trapezoidal
   channel cross-section using a sparse CSR matrix solve at **dt = 3600s**
   (1-hour timestep, hardcoded in `mmc.py:192`).
4. Gradients flow from the loss (L1/MAE) back through the routing physics into
   the KAN weights. Only KAN weights are learned; routing physics are
   differentiable but not parameterized.

### End-to-End Data Flow

```
Catchment Attributes ──► KAN ──► {n, q_spatial} [0,1]
                                       │
                              denormalize to physical bounds
                                       │
                                       ▼
Lateral Inflow (Q') ──► StreamflowReader ──► hourly tensor
                                                  │
                         ┌────────────────────────┘
                         ▼
              MuskingumCunge.route()
              ├─ _get_trapezoid_velocity()    (Leopold & Maddock geometry)
              ├─ calculate_muskingum_coefficients()  (c1, c2, c3, c4)
              └─ triangular_sparse_solve()    (CSR linear system per timestep)
                         │
                         ▼
              Hourly routed discharge (num_segments, num_hours)
                         │
                    downsample()  ──► F.interpolate(mode="area")
                         │
                         ▼
              Daily predictions ──► L1 Loss vs USGS obs ──► backprop ──► KAN
```

---

## Module Map

### Core Library (`src/ddr/`)

| Path | Role |
|---|---|
| `routing/mmc.py` | `MuskingumCunge` engine — sparse matrix solve, trapezoid velocity, Muskingum coefficients. Key: `route()` loops timesteps, `route_timestep()` solves one step, `setup_inputs()` denormalizes NN params and cold-starts discharge. |
| `routing/torch_mc.py` | PyTorch `nn.Module` wrapper (`dmc` class). Manages `MuskingumCunge` lifecycle, exposes `forward()` for autograd. |
| `nn/kan.py` | KAN neural network. Architecture: `Linear → KAN layers → Linear → Sigmoid`. Input: normalized attributes `(batch, n_attrs)`. Output: `dict[str, Tensor]` of [0,1] parameter predictions. |
| `io/readers.py` | `StreamflowReader` — reads lateral inflows from icechunk stores via `isel`. Daily stores are nearest-neighbor interpolated to hourly via `np.repeat(..., 24)`. Also: `IcechunkUSGSReader`, `AttributesReader`, `read_ic()`, gage filtering functions, `build_flow_scale_tensor()`. |
| `io/functions.py` | `downsample()` — hourly → daily via `F.interpolate(mode="area")`. |
| `io/builders.py` | `construct_network_matrix()` — unions per-gage COO subgraphs from zarr into a single adjacency matrix. `_build_network_graph()` — creates rustworkx DiGraph from CONUS adjacency zarr. |
| `geodatazoo/base_geodataset.py` | `BaseGeoDataset(ABC)` — PyTorch Dataset. Training: `__getitem__` returns gage IDs, `collate_fn` calls `_collate_gages()`. Inference: pre-builds `RoutingDataclass` once in `__init__`. |
| `geodatazoo/merit.py` | `Merit(BaseGeoDataset)` — MERIT-Hydro dataset. Uses integer COMIDs, NetCDF attributes, Leopold & Maddock geometry (no observed top_width/side_slope). Muskingum x fixed at 0.3. |
| `geodatazoo/lynker_hydrofabric.py` | `LynkerHydrofabric(BaseGeoDataset)` — Lynker v2.2 dataset. Uses `"cat-{id}"` divide IDs, icechunk attributes, observed trapezoid geometry, Muskingum x from zarr. |
| `geodatazoo/dataclasses.py` | `RoutingDataclass` — batch container (adjacency matrix, attributes, physical tensors, observations). `Dates` — temporal window management; `calculate_time_period()` for random batch sampling, `set_batch_time()` for hourly/daily ranges. |
| `validation/configs.py` | Pydantic config models: `Config`, `DataSources`, `Params`, `Kan`, `ExperimentConfig`, `AttributeMinimums`. Also `validate_config()` entry point. |
| `validation/enums.py` | `GeoDataset` enum (merit, lynker_hydrofabric) with `get_dataset_class()` factory. `Mode` enum (training, testing, routing). |
| `validation/metrics.py` | `Metrics` — Pydantic model that auto-computes 13+ metrics on init: NSE, KGE, RMSE, bias, FLV, FHV, pbias, correlation, FDC RMSE, etc. |
| `validation/plots.py` | `plot_time_series()`, `plot_cdf()`, `plot_box_fig()`, `plot_drainage_area_boxplots()`, `plot_gauge_map()`, `plot_routing_hydrograph()`. Publication-quality matplotlib. |
| `validation/utils.py` | `save_state()` — checkpoint serialization (KAN weights, optimizer, RNG states, epoch/mini_batch). |
| `geometry/` | `compute_trapezoidal_geometry()` — Manning's equation inversion + Leopold & Maddock. `GeometryPredictor` — standalone channel geometry estimator wrapping trained KAN. |
| `bmi/ddr_bmi.py` | BMI interface for ngen framework. Runs KAN once at init, routes per `update_until()` call. Supports sub-stepping with constant/linear interpolation. |
| `scripts_utils.py` | `compute_daily_runoff()` — boundary trimming + downsample. `load_checkpoint()`, `resolve_learning_rate()`, `safe_mean()`, `safe_percentile()`. |

### Public API (`src/ddr/__init__.py`)

```python
from .routing.torch_mc import dmc        # Differentiable routing model
from .nn import kan                       # KAN neural network
from .io.readers import StreamflowReader as streamflow  # Data reader
from .io import functions as ddr_functions              # Utilities
from . import validation                  # Config, Metrics, plotting
```

---

## Config System

### Flow

```
Hydra YAML (config/) → OmegaConf DictConfig → validate_config() → Pydantic Config
```

1. **Hydra** loads YAML from `config/` via `@hydra.main(config_path="../config")`.
   User selects config with `--config-name=<name>`.
2. **OmegaConf** resolves `${oc.env:VAR_NAME,default}` interpolation.
3. **`validate_config()`** calls `OmegaConf.to_container(cfg, resolve=True)`,
   instantiates `Config(**dict)`, sets seeds, saves config to output dir.

### Key Config Models

**Config** (root):
- `name`, `geodataset`, `mode`, `device`, `seed`, `np_seed`, `s3_region`
- `data_sources: DataSources`, `params: Params`, `kan: Kan`, `experiment: ExperimentConfig`

**DataSources**:
- `streamflow` — icechunk store path (local or S3)
- `is_hourly: bool = False` — if True, StreamflowReader indexes hourly directly
- `observations` — USGS icechunk store
- `attributes` — catchment attributes (NetCDF for MERIT, icechunk for Lynker)
- `gages`, `gages_adjacency` — gauge metadata CSV and zarr adjacency store
- `conus_adjacency`, `geospatial_fabric_gpkg`, `statistics`
- `target_catchments: list[str] | None` — optional explicit catchment list

**Params**:
- `parameter_ranges: dict[str, list[float]]` — physical bounds (e.g., `n: [0.015, 0.25]`)
- `log_space_parameters: list[str]` — parameters denormalized in log-space
- `attribute_minimums: dict[str, float]` — clamping floors for routing stability
- `tau: int = 3` — timezone/boundary adjustment (hours trimmed from routing output)
- `save_path: Path`

**Kan**:
- `input_var_names: list[str]` — catchment attribute names (determines input_size)
- `learnable_parameters: list[str]` — output parameter names (e.g., `["n", "q_spatial"]`)
- `hidden_size`, `num_hidden_layers`, `grid`, `k`

**ExperimentConfig**:
- `rho: int | None` — days per training batch (None = full period for inference)
- `warmup: int = 3` — days excluded from loss (routing spin-up)
- `learning_rate: dict[int, float]` — epoch → LR schedule (e.g., `{1: 0.005, 3: 0.001}`)
- `batch_size`, `epochs`, `shuffle`, `start_time`, `end_time`
- `checkpoint: Path | None` — trained model to load

### Environment Variables

Configs use `${oc.env:VAR,default}` for portable paths:
- `DDR_VERSION` — auto-set from `ddr._version.__version__`
- `DDR_DATA_DIR` — data directory root
- `DDR_STREAMFLOW_STORE`, `DDR_OBS_STORE`, `DDR_ATTRIBUTES_STORE`

### Hydra Output Structure

```
output/{name}/{YYYY-MM-DD_HH-MM-SS}/
├── .hydra/config.yaml          # Resolved config snapshot
├── pydantic_config.yaml        # Validated config
├── plots/                      # Generated visualizations
├── saved_models/               # Checkpoints: _{name}_epoch_{e}_mb_{mb}.pt
└── model_test.zarr             # Test predictions (if testing)
```

---

## Training Pipeline (`scripts/train.py`)

```
@hydra.main → validate_config() → Config
  │
  ├─ Dataset: GeoDataset.get_dataset_class(cfg) → Merit or LynkerHydrofabric
  ├─ Sampler: RandomSampler (shuffled)
  ├─ DataLoader: batch_size gages, collate_fn=dataset.collate_fn, drop_last=True
  │
  ├─ Models: flow=StreamflowReader(cfg), nn=kan(cfg), routing_model=dmc(cfg)
  ├─ Optimizer: Adam, LR from schedule, gradient clipping norm=1.0
  │
  └─ Training loop:
     FOR each epoch:
       resolve_learning_rate(schedule, epoch)
       FOR each batch (routing_dataclass):
         1. dates.calculate_time_period()         # random rho-day window
         2. q_prime = flow(routing_dataclass)      # hourly lateral inflow
         3. params = nn(normalized_attributes)     # KAN → {n, q_spatial}
         4. output = dmc(routing_dataclass, q_prime, params)  # route @ 1hr
         5. daily = downsample(output[:, 13+tau:-11+tau])     # hourly → daily
         6. loss = L1(daily[warmup:], observations[warmup:])  # skip spin-up
         7. loss.backward() → grad_clip → optimizer.step()
         8. save_state() periodically
```

### Checkpoint Structure (`.pt` files)

```python
{
    "model_state_dict": {KAN weights},
    "optimizer_state_dict": {Adam state},
    "rng_state", "cuda_rng_state", "data_generator_state",
    "epoch": int, "mini_batch": int,
}
```

Resume: `load_checkpoint()` restores KAN weights + optimizer + RNG states.

---

## Testing Pipeline (`scripts/test.py`)

Same model setup as training, but:
- `SequentialSampler` (no shuffling, deterministic order)
- `torch.no_grad()` — no gradient computation
- `model.eval()` mode
- Accumulates hourly predictions across all batches into a single array
- Downsamples to daily, computes `Metrics(pred, target)` over full eval period
- Saves predictions + observations to `model_test.zarr`

`scripts/train_and_test.py` — runs training then auto-discovers latest
checkpoint and evaluates on a different time period.

---

## Routing Script (`scripts/router.py`)

Forward routing with a trained checkpoint over a large domain:
- Supports three modes: target_catchments, gages, or all segments
- Uses `carry_state=True` across batches for temporal continuity
- Outputs `chrout.zarr` with `(catchment_ids, time)` dimensions
- Generates summary hydrograph plot

---

## Validation Script (`scripts/summed_q_prime.py`)

Baseline evaluation — unrouted sum of upstream lateral inflows:
- Pre-collects all upstream divides across all gauges (from `gages_adjacency`)
- Loads only needed divides via `isel` (not full CONUS)
- For hourly stores: `isel` + numpy reshape + mean → daily (avoids lazy resample)
- GPU-accelerated per-gauge `nansum` via CuPy
- Compares against USGS daily observations using `Metrics`

---

## Temporal Resolution Handling

### StreamflowReader (`src/ddr/io/readers.py`)

- **Daily store** (`is_hourly=False`): Reads daily values via `isel`, then
  `np.repeat(..., 24)[:, :n_hourly]` for nearest-neighbor → hourly.
- **Hourly store** (`is_hourly=True`): Reads hourly values directly via `isel`
  with `(timestamps - store_start).total_seconds() // 3600` index computation.
- Time offset: computed from store's actual start date vs 1980/01/01 origin.

### Boundary Trimming

Routing output is trimmed before downsampling:
`output[:, (13+tau):(-11+tau)]` where `tau` (default 3) handles timezone
alignment. This removes ~13 hours of spin-up and ~8 hours of edge effects.

### Downsampling

`F.interpolate(data.unsqueeze(1), size=(rho,), mode="area")` — area-weighted
averaging preserves total discharge flux.

---

## Dataset Layer (`src/ddr/geodatazoo/`)

### Batch Construction (Training)

```
DataLoader.__getitem__(idx) → gage_ids[idx] (string STAID)
  │
  collate_fn(batch_of_gage_ids)
  ├─ dates.calculate_time_period()           # random rho-day window
  └─ _collate_gages(batch)
     ├─ construct_network_matrix()           # union per-gage COO subgraphs
     ├─ compress indices to active segments  # CONUS index → batch index
     ├─ _build_common_tensors()
     │  ├─ CSR adjacency tensor
     │  ├─ attributes (z-score normalized)
     │  ├─ flowpath tensors (length, slope, x, top_width, side_slope)
     │  └─ fill NaNs with per-attribute batch mean
     ├─ build_flow_scale_tensor()            # partial-area gage corrections
     └─ create_hydrofabric_observations()    # select obs for batch gages
         │
         ▼
     RoutingDataclass (adjacency, attrs, physical tensors, obs, dates)
```

### Inference (Testing/Routing)

Pre-builds `RoutingDataclass` once in `__init__()`. Three modes (priority order):
1. **target_catchments** — upstream ancestor traversal via rustworkx
2. **gages** — subgraph union from gages_adjacency
3. **all segments** — full CONUS adjacency

`collate_fn` returns the pre-built dataclass; only the time window updates.

### MERIT vs Lynker Differences

| | MERIT | Lynker Hydrofabric |
|---|---|---|
| Divide IDs | Integer COMIDs | `"cat-{id}"` strings |
| Attributes | NetCDF (`xr.open_mfdataset`) | Icechunk (`read_ic`) |
| Geometry | Leopold & Maddock only | Observed top_width, side_slope |
| Muskingum x | Fixed 0.3 | From zarr (per-segment) |
| Area field | `log10_uparea` | `log_uparea` |

---

## Workspace (Monorepo)

Three packages managed by `uv`:

| Package | Directory | Purpose |
|---|---|---|
| `ddr` | `.` (root) | Core routing library |
| `ddr-engine` | `engine/` | Geospatial data prep: geodataset → zarr adjacency matrices |
| `ddr-benchmarks` | `benchmarks/` | Evaluation framework: DDR vs DiffRoute comparison |

```bash
uv sync --all-packages
```

### Engine (`engine/`)

Builds sparse adjacency matrices from external hydrological datasets:
- `core/converters.py` — pluggable ID conversion (MERIT integers, Lynker `wb-{n}` strings)
- `core/zarr_io.py` — COO matrix I/O following binsparse v3 spec
- `merit/` — MERIT-specific graph construction from shapefiles
- `lynker_hydrofabric/` — Lynker v2.2 construction from GeoPackages (SQLite + Polars)
- Uses rustworkx for topological sort and cycle detection

### Benchmarks (`benchmarks/`)

Standardized comparison framework:
- `benchmark.py` — runs DDR + DiffRoute on identical inputs, computes metrics
- `diffroute_adapter.py` — converts zarr COO → NetworkX DiGraph for DiffRoute
- `validation/` — `BenchmarkConfig` Pydantic model, `DiffRouteConfig`
- Outputs: `benchmark_results.zarr`, CDF/box/map plots, per-gage hydrographs

---

## Downstream Call-Site Checklist

When modifying `src/ddr/` interfaces (constructor signatures, `forward()`
return types, config fields), **always check and update**:

1. **`examples/`** — Notebooks that instantiate `kan()`, `dmc()`, load configs.
2. **`benchmarks/`** — Own `kan()`/`dmc()` instantiation and evaluation loops.
3. **`scripts/`** — All training/testing/routing scripts.
4. **`config/`** — YAML files referencing changed field names.

```bash
grep -r "kan(" examples/ benchmarks/ scripts/
```

---

## Testing

```bash
uv run pytest                    # Unit tests (no data dependencies)
uv run pytest -m integration     # Integration tests (requires HPC data)
```

- Unit tests in `tests/` mirror `src/ddr/` structure.
- Integration tests marked with `@pytest.mark.integration`, deselected by default.
- Key test files:
  - `tests/io/test_readers.py` — StreamflowReader offset, hourly/daily paths
  - `tests/scripts/test_summed_q_prime.py` — hourly collapse, pre-filtered divides, CuPy
  - `tests/routing/test_mmc.py`, `test_torch_mc.py` — routing correctness
  - `tests/nn/test_kan.py` — KAN forward pass
  - `tests/validation/test_configs.py` — config validation

---

## Code Quality

| Tool | Config |
|---|---|
| **Linter** | ruff — rules: F, E, W, I, D, B, Q, TID, C4, BLE, UP, RUF100 |
| **Formatter** | ruff format |
| **Type checker** | mypy (strict: `disallow_untyped_defs = true`) |
| **Docstrings** | NumPy convention (`tool.ruff.lint.pydocstyle`) |
| **Line length** | 110 |
| **Pre-commit** | ruff check+format, mypy, nbstripout, trailing-whitespace, end-of-file-fixer, check-yaml |

All config in `pyproject.toml`. Pre-commit hooks in `.pre-commit-config.yaml`.

---

## Key Constants & Conventions

- Python **>=3.11, <3.14**.
- PyTorch with CUDA 13.0 index (configurable in `pyproject.toml`).
- **Routing timestep**: `dt = 3600.0s` (hardcoded in `mmc.py:192`). Always routes
  at hourly resolution regardless of input data frequency.
- **Sparse CSR tensors** for the routing matrix solve — expect PyTorch beta
  warnings (suppressed in pytest config).
- **Parameter denormalization**: KAN outputs [0,1] → physical bounds.
  Linear: `val = sigmoid * (max - min) + min`.
  Log-space (for `p_spatial`): `val = exp(sigmoid * (log_max - log_min) + log_min)`.
- **Celerity factor**: wave speed = velocity * 5/3.
- `__init__.py` files use `F401` ignore for re-exports.

### Data Stores

| Store | Resolution | Units | Notes |
|---|---|---|---|
| Daily LSTM | Daily | m³/s (converted from mm/day) | Nearest-neighbor → hourly in StreamflowReader |
| Hourly LSTM | Hourly | m³/s (converted from mm/day) | Direct hourly indexing |
| dHBV2.0 | Daily | m³/s (pre-converted) | No conversion needed |
| USGS observations | Daily | m³/s (ft³/s converted) | Mean of midnight-to-midnight LST |

All icechunk stores use `Qr` variable with dims `(divide_id, time)`.
