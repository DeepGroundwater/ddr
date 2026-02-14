# DDR Routing Refactor: Effective Java + BMI Lifecycle

## Context

The routing core (`MuskingumCunge` + `dmc`) works correctly but has architectural debt that will block future goals — particularly exposing routing via gRPC with init/update/finalize endpoints. The current design has a god class (`MuskingumCunge`, ~731 lines), duplicated state in the wrapper (`dmc` mirrors 12+ attributes), scripts that repeat identical orchestration logic, and no formal interface contract. This refactor applies Effective Java principles (translated to Python) to produce a modular, BMI-style lifecycle that maps cleanly to gRPC service endpoints.

---

## Effective Java Principles Applied

| Bloch Item | Python Translation | Current Problem | Proposed Fix |
|---|---|---|---|
| **Item 1**: Static factory methods | `@classmethod` factories | Single `__init__` constructor for all use modes | `MuskingumCungeRouter.for_training()`, `.for_inference()`, `.from_config()` |
| **Item 5**: Prefer DI to hardwiring | Constructor injection | Constructors open files, create internal objects, branch on booleans | Inject datasets, routing engines, and strategies as optional params |
| **Item 13**: Minimize accessibility | Private attrs + properties | All state is public (`self.n`, `self.q_prime`, mixed `_` prefix) | Prefix internal state with `_`, expose read-only `@property` |
| **Items 15-17**: Minimize mutability | Frozen dataclasses | Network topology, bounds stored as mutable attrs on god class | `NetworkTopology(frozen=True)`, `PhysicalBounds(frozen=True)` — immutable value objects |
| **Item 18**: Composition over inheritance | Compose small objects | `dmc` wraps `MuskingumCunge` but leaks its guts (12 duplicated attrs) | `dmc` composes `MuskingumCungeRouter` + `PhysicalBounds` — no state duplication |
| **Item 20**: Prefer interfaces | `typing.Protocol` | No formal contract — callers depend on concrete class | `RoutingModel` Protocol defines BMI lifecycle |
| **Item 51**: Design signatures carefully | Typed params, no `**kwargs` | `dmc.forward(**kwargs)` is untyped dict | Typed `forward(topology, lateral_inflow, spatial_params, ...)` |
| **Item 64**: Refer to objects by interfaces | Program to Protocol | Scripts import concrete `dmc`, `MuskingumCunge` | Scripts and gRPC use `RoutingModel` Protocol |

---

## New Type Hierarchy

### `src/ddr/routing/protocols.py` (NEW)

```
RoutingModel (Protocol)         # BMI lifecycle contract
├── initialize(topology, bounds) -> None
├── coldstart(q_prime_t0) -> None
├── update(timestep_input) -> Tensor           # single timestep
├── update_until(lateral_inflow, params, ...) -> Tensor  # full window
├── finalize() -> RoutingState
├── get_state() -> RoutingState
└── set_state(state) -> None

NetworkTopology (frozen dataclass)   # immutable network structure
├── adjacency_matrix, length, slope, x_storage, top_width, side_slope
├── outflow_idx, gage_catchment, flow_scale, divide_ids
├── num_segments (derived)
└── @classmethod from_routing_dataclass(rd, device, slope_min)

PhysicalBounds (frozen dataclass)    # immutable config-derived bounds
├── parameter_ranges, log_space_parameters, attribute_minimums
├── p_spatial, use_leakance, dt
└── @classmethod from_config(cfg)

RoutingState (mutable dataclass)     # THE ONLY mutable piece
├── discharge_t: Tensor
├── zeta_sum, q_prime_sum (leakance diagnostics)
├── K_D_t, d_gw_t, leakance_factor_t (time-varying leakance)
└── snapshot() -> RoutingState (deep copy for checkpoint/serialize)

DenormalizedParams (frozen dataclass)  # physics parameters after denorm
└── n, q_spatial, top_width, side_slope: Tensor
```

### `src/ddr/routing/physics.py` (NEW)

Pure functions extracted from `MuskingumCunge` — no state, no side effects, fully differentiable:

| Function | Source | Lines |
|---|---|---|
| `compute_trapezoid_velocity()` | `_get_trapezoid_velocity()` in mmc.py:74-143 | ~70 |
| `compute_muskingum_coefficients()` | `calculate_muskingum_coefficients()` in mmc.py | ~25 |
| `compute_zeta()` | `_compute_zeta()` in mmc.py:146-197 | ~50 |
| `compute_hotstart_discharge()` | `compute_hotstart_discharge()` in mmc.py:25-66 | ~40 |
| `denormalize_params()` | `_denormalize_spatial_parameters()` + `denormalize()` from utils.py | ~30 |

---

## Input Specifications & Contracts

### Formal Tensor Contracts

Each interface boundary has an exact shape/type contract. These replace the current implicit kwargs contracts.

#### `dmc.forward()` — new typed signature

```python
def forward(
    self,
    topology: NetworkTopology,              # from_routing_dataclass()
    lateral_inflow: torch.Tensor,           # (T_hourly, N) float32 — from StreamflowReader
    spatial_params: dict[str, torch.Tensor], # KAN output, each value (N,) in [0,1]
    carry_state: bool = False,
    leakance_params: dict[str, torch.Tensor] | None = None,  # LSTM output
) -> dict[str, torch.Tensor]:
    # Returns {"runoff": (num_outputs, T_hourly), ["zeta_sum": (N,), "q_prime_sum": (N,)]}
```

#### `RoutingModel.initialize()` — topology contract

```python
def initialize(self, topology: NetworkTopology, bounds: PhysicalBounds) -> None:
```

`NetworkTopology` **required fields** (must be non-None):
- `adjacency_matrix`: sparse CSR `(N, N)` — topologically ordered, lower-triangular
- `length`: `(N,)` float32, meters, > 0
- `slope`: `(N,)` float32, clamped to >= `bounds.attribute_minimums["slope"]`
- `x_storage`: `(N,)` float32, Muskingum x in [0, 0.5]

`NetworkTopology` **optional fields** (can be empty tensor or None):
- `top_width`: `(N,)` or `torch.empty(0)` — if empty, KAN must provide via `spatial_params`
- `side_slope`: `(N,)` or `torch.empty(0)` — if empty, KAN must provide via `spatial_params`
- `outflow_idx`: `list[np.ndarray]` or None — ragged indices for gage-mode output
- `gage_catchment`: `list[str]` or None — gage IDs corresponding to outflow_idx
- `flow_scale`: `(N,)` or None — area-mismatch scaling
- `divide_ids`: `np.ndarray` or None — segment IDs (COMIDs or "cat-{id}")

#### `RoutingModel.update()` — single timestep contract

```python
def update(self, timestep_input: TimestepInput) -> torch.Tensor:
    # Returns: (N,) discharge at t+1
```

`TimestepInput`:
- `lateral_inflow`: `(N,)` float32, clamped to >= `discharge_lb`
- `spatial_params`: `DenormalizedParams` — already in physical units

#### `RoutingModel.update_until()` — full window contract

```python
def update_until(
    self,
    lateral_inflow: torch.Tensor,           # (T_hourly, N) float32
    spatial_params: DenormalizedParams,
    leakance_params: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    # Returns: (num_outputs, T_hourly) or (N, T_hourly) depending on outflow_idx
```

#### `DenormalizedParams` — the physics parameter contract

```python
@dataclass(frozen=True)
class DenormalizedParams:
    n: torch.Tensor             # (N,) Manning's n, typically [0.015, 0.25]
    q_spatial: torch.Tensor     # (N,) channel shape, [0, 1]
    top_width: torch.Tensor     # (N,) meters — from KAN OR from NetworkTopology
    side_slope: torch.Tensor    # (N,) H:V ratio — from KAN OR from NetworkTopology
```

**Critical: MERIT vs Lynker parameter sourcing**

The `denormalize_params()` function handles both geodataset patterns:

```python
def denormalize_params(
    raw_params: dict[str, torch.Tensor],   # KAN output, each (N,) in [0, 1]
    bounds: PhysicalBounds,
    topology: NetworkTopology,
) -> DenormalizedParams:
    n = denormalize(raw_params["n"], bounds.parameter_ranges["n"])
    q_spatial = denormalize(raw_params["q_spatial"], bounds.parameter_ranges["q_spatial"])

    # top_width: use KAN if available, else use hydrofabric value
    if "top_width" in raw_params:
        top_width = denormalize(raw_params["top_width"], bounds.parameter_ranges["top_width"], log=True)
    else:
        top_width = topology.top_width  # Lynker provides this directly

    # side_slope: same logic
    if "side_slope" in raw_params:
        side_slope = denormalize(raw_params["side_slope"], bounds.parameter_ranges["side_slope"], log=True)
    else:
        side_slope = topology.side_slope

    return DenormalizedParams(n=n, q_spatial=q_spatial, top_width=top_width, side_slope=side_slope)
```

This preserves today's behavior (mmc.py `_denormalize_spatial_parameters()` already does this check) but makes it explicit.

#### `StreamflowReader.forward()` contract

```python
def forward(
    self,
    routing_dataclass: RoutingDataclass,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    use_hourly: bool = False,
) -> torch.Tensor:
    # Returns: (T_hourly, N) float32 — lateral inflow per reach
    # Daily data interpolated to hourly (nearest) unless use_hourly=True
```

#### `KAN.forward()` contract

```python
def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    # inputs: (N, num_attrs) float32 — normalized_spatial_attributes
    # Returns: {param_name: (N,) in [0, 1]} for each learnable_parameter
```

#### `LeakanceLSTM.forward()` contract

```python
def forward(
    self,
    q_prime: torch.Tensor,       # (T_daily, N) float32
    attributes: torch.Tensor,    # (N, num_attrs) float32
) -> dict[str, torch.Tensor]:
    # Returns: {"K_D": (T_daily, N), "d_gw": (T_daily, N), "leakance_factor": (T_daily, N)}
    # All in [0, 1] — denormalization happens in router.set_leakance()
```

---

## Geodataset Handling

### Current differences between MERIT and Lynker

| Aspect | MERIT | Lynker Hydrofabric |
|---|---|---|
| **ID type** | `int` (COMID) | `str` ("cat-{id}") |
| **Attribute source** | `xr.open_mfdataset()` (local NetCDF) | `read_ic()` (Icechunk cloud) |
| **Flowpath source** | Shapefile layer by COMID | GeoPackage layer "flowpath-attributes-ml" |
| **Channel top_width** | Not in hydrofabric — **KAN learns it** | In hydrofabric (`TopWdth`) — passed through |
| **Channel side_slope** | Not in hydrofabric — **KAN learns it** | In hydrofabric (`ChSlp`) — passed through |
| **Muskingum x** | Fixed at 0.3 (`torch.full`) | From hydrofabric (`MusX` per reach) |
| **Length units** | `lengthkm * 1000` — meters | `Length_m` already in meters |

### How the refactor handles this

The refactor does **not** change the geodataset layer. `RoutingDataclass` continues to be the bridge:

```
BaseGeoDataset.collate_fn()
  -> RoutingDataclass (contains whatever each geodataset provides)
    -> NetworkTopology.from_routing_dataclass()
      -> Carries top_width, side_slope (may be empty tensors for MERIT)
        -> denormalize_params() checks: "is this in KAN output? else use topology value"
```

The key design decision: **`NetworkTopology` carries optional physical properties, and `denormalize_params()` resolves the source.** This matches the current `_denormalize_spatial_parameters()` logic but makes it explicit in a pure function rather than buried in the god class.

### Adding a new geodataset

To add a third geodataset (e.g., NHDPlus, GloFAS), a developer would:
1. Subclass `BaseGeoDataset`, implement abstract methods
2. Add enum value to `GeoDataset` in `enums.py`
3. Add mapping in `GeoDataset.get_dataset_class()`
4. Ensure `_build_common_tensors()` populates RoutingDataclass fields
5. **No changes to routing, dmc, physics, or protocols needed** — those are geodataset-agnostic

---

## Benchmarking Architecture

### Current problem

`benchmarks/src/ddr_benchmarks/benchmark.py` (866 lines) does everything: DDR routing, DiffRoute routing, summed Q' loading, metrics, filtering, mass balance checks, 7 plot types, and zarr output. This is another god-function.

### Proposed decomposition

```
benchmarks/src/ddr_benchmarks/
|-- benchmark.py          # BenchmarkRunner: orchestrates all models + comparison
|-- adapters/
|   |-- protocols.py      # ModelAdapter Protocol
|   |-- ddr_adapter.py    # DDR routing via RoutingOrchestrator
|   |-- diffroute_adapter.py  # DiffRoute per-gage routing (mostly existing code)
|   +-- baseline_adapter.py   # Summed Q' from pre-computed zarr
|-- results.py            # BenchmarkResults dataclass
|-- comparison.py         # Multi-model metrics comparison + filtering
+-- plots.py              # Benchmark-specific plot generation (extract from benchmark.py:311-587)
```

#### `ModelAdapter` Protocol

```python
class ModelAdapter(Protocol):
    """Adapter for any routing model that produces discharge predictions."""
    name: str

    def run(
        self,
        dataset: BaseGeoDataset,
        dataloader: DataLoader,
        device: str | torch.device,
    ) -> ModelOutput: ...

@dataclass
class ModelOutput:
    predictions: np.ndarray          # (n_gages, n_days) daily discharge [m3/s]
    gage_ids: list[str]              # which gages have valid predictions
    metadata: dict[str, Any]         # model-specific info (checkpoint, params, etc.)
```

Each adapter implements `run()` returning predictions in the same shape:

- **DDRAdapter**: Wraps `RoutingOrchestrator.route_batch()` in a loop, accumulates hourly -> daily, returns predictions for all gages
- **DiffRouteAdapter**: Wraps existing per-gage DiffRoute logic (`run_diffroute_benchmark()`), returns predictions for routed gages (headwaters -> NaN)
- **BaselineAdapter**: Loads summed Q' from zarr, aligns gage IDs, returns predictions for matched gages

#### `BenchmarkRunner`

```python
class BenchmarkRunner:
    def __init__(self, adapters: list[ModelAdapter], dataset: BaseGeoDataset): ...

    def run_all(self) -> BenchmarkResults:
        outputs = {adapter.name: adapter.run(self.dataset, ...) for adapter in self.adapters}
        common_gages = self._intersect_valid_gages(outputs)
        return BenchmarkResults(outputs=outputs, common_gages=common_gages, observations=obs)

    def compute_metrics(self, results: BenchmarkResults) -> dict[str, Metrics]:
        # Per-model Metrics on common gage subset
        ...

    def generate_plots(self, results: BenchmarkResults, metrics: dict[str, Metrics]) -> None:
        # Delegates to plot functions
        ...
```

#### `BenchmarkResults`

```python
@dataclass
class BenchmarkResults:
    outputs: dict[str, ModelOutput]       # model_name -> output
    observations: np.ndarray              # (n_gages, n_days)
    common_gages: list[str]               # gages present in ALL models
    gage_info: pd.DataFrame               # lat/lon/drainage_area for maps
    time_range: pd.DatetimeIndex
```

### How the orchestrator interacts with benchmarking

The `RoutingOrchestrator` handles the DDR loop (flow -> KAN -> dmc). The `DDRAdapter` wraps it:

```python
class DDRAdapter:
    name = "DDR"

    def __init__(self, orchestrator: RoutingOrchestrator): ...

    def run(self, dataset, dataloader, device) -> ModelOutput:
        predictions = np.full((n_gages, n_hours), np.nan)
        for i, routing_dataclass in enumerate(dataloader):
            output = self.orchestrator.route_batch(routing_dataclass, carry_state=i > 0)
            predictions[:, dataset.dates.hourly_indices] = output["runoff"].cpu().numpy()
        daily = downsample_hourly_to_daily(predictions)
        return ModelOutput(predictions=daily, gage_ids=gage_ids, metadata={...})
```

DiffRoute and Summed Q' adapters don't use the orchestrator at all — they have their own logic. The `ModelAdapter` protocol is the common contract.

---

## Training Batch Management (DataLoader/Dataset — Unchanged)

### Current Architecture

Training uses standard PyTorch `Dataset` + `DataLoader`. **This layer is untouched by the refactor.**

```
Merit(BaseGeoDataset(torch.utils.data.Dataset))
  ├── __getitem__(idx) → gage_id: str       # Returns a single gage ID
  ├── collate_fn(gage_ids) → RoutingDataclass  # Builds network + loads data
  └── __len__() → num_training_gages
```

```python
# scripts/train.py — setup (unchanged by refactor)
dataset = cfg.geodataset.get_dataset_class(cfg=cfg)  # Merit instance
sampler = RandomSampler(dataset, generator=data_generator)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=cfg.experiment.batch_size,  # 64 gages (SPATIAL batching)
    num_workers=0,
    sampler=sampler,
    collate_fn=dataset.collate_fn,
    drop_last=True,
)
```

### Batching is Spatial, Not Temporal

- **batch_size=64** means 64 gages (spatial locations), NOT 64 timesteps
- Each batch contains the **full upstream network** for those 64 gages (~100-500 reaches)
- All gages in a batch share the **same random 365-day time window** (rho=365)

### Random Time Window Selection

Each batch gets a fresh random window via `collate_fn` → `Dates.calculate_time_period()`:

```python
# In collate_fn (base_geodataset.py:39-50):
def collate_fn(self, *args, **kwargs) -> RoutingDataclass:
    if self.cfg.mode == Mode.TRAINING:
        self.dates.calculate_time_period()  # Random 365-day window
        return self._collate_gages(np.array(args[0]))
```

```python
# In dataclasses.py — Dates.calculate_time_period():
random_start = randint(0, len(daily_time_range) - rho)
batch_daily_time_range = daily_time_range[random_start : random_start + rho]
# → 365 days → 8760 hourly timesteps
```

### What Each Batch Contains

| Component | Shape | Source |
|---|---|---|
| Gage IDs | `[64]` | `RandomSampler` → `__getitem__` |
| Reaches | `[N]` (~100-500) | Upstream network of those 64 gages |
| Time window | 365 consecutive days | Random start, new each batch |
| q_prime (lateral inflow) | `(8760, N)` | `StreamflowReader` from Icechunk |
| Attributes | `(N, 10)` | MERIT z-scored attributes |
| Observations | `(64, ~362)` | USGS daily obs (trimmed by warmup) |
| Adjacency matrix | sparse CSR `(N, N)` | `construct_network_matrix()` |

### State Management During Training

- **`carry_state=False`** for every batch — each batch cold-starts independently
- **Reason**: Random time windows break temporal continuity; carrying state would be invalid
- **Cold start**: `compute_hotstart_discharge()` solves `(I - N)^{-1} @ q_prime[0]` for topological accumulation
- **Warmup**: First 3 days excluded from MSE loss (not from routing), allows cold-started discharge to stabilize

### How the Refactor Preserves This

The `RoutingOrchestrator` wraps the **inner batch logic** (flow → KAN → dmc), but the **outer** DataLoader/Dataset/collate_fn/Sampler pattern is untouched:

```python
# BEFORE (train.py ~100 lines of inner logic):
for epoch in range(epochs):
    for i, routing_dataclass in enumerate(dataloader):
        q_prime = flow(routing_dataclass=routing_dataclass, ...)
        spatial_params = nn(inputs=routing_dataclass.normalized_spatial_attributes.to(device))
        # ... leakance logic ...
        output = routing_model(routing_dataclass=routing_dataclass, spatial_parameters=spatial_params,
                               streamflow=q_prime, carry_state=False)
        # ... loss computation, backward, optimizer step ...

# AFTER (train.py ~30 lines):
orchestrator = RoutingOrchestrator(cfg, routing_model, nn, flow, leakance_nn)
for epoch in range(epochs):
    for i, routing_dataclass in enumerate(dataloader):
        output = orchestrator.route_batch(routing_dataclass, carry_state=False)
        # ... loss computation, backward, optimizer step ...
```

**What stays the same**: Dataset, DataLoader, RandomSampler, collate_fn, batch_size, rho, random time windows, coldstart per batch, warmup exclusion from loss.

**What changes**: The 5 lines of inner logic (flow → KAN → leakance → dmc) collapse into `orchestrator.route_batch()`.

### Training vs Inference Batching

| Aspect | Training | Inference |
|---|---|---|
| **Sampler** | `RandomSampler` (shuffled gages) | `SequentialSampler` (ordered batches) |
| **Time window** | Random 365-day slice per batch | Full contiguous date range, split into sequential batches |
| **carry_state** | `False` (each batch independent) | `True` for batch i > 0 (state carries over) |
| **Batch composition** | 64 gages + upstream reaches | All reaches in network, batched by subnetwork |
| **Loss computation** | MSE after warmup | None (no gradients) |
| **collate_fn** | `_collate_gages()` (per-gage upstream networks) | `_collate_inference()` (full network batches) |

---

## Script Orchestration

### `src/ddr/routing/orchestrator.py` (NEW)

Captures the loop logic duplicated across train.py, test.py, router.py, benchmark.py:

```python
class RoutingOrchestrator:
    def __init__(self, cfg, routing_model, nn, flow, leakance_nn=None): ...

    def route_batch(self, routing_dataclass, carry_state=False) -> dict[str, Tensor]:
        topology = NetworkTopology.from_routing_dataclass(routing_dataclass, ...)
        q_prime = self._flow(routing_dataclass=routing_dataclass, ...)
        spatial_params = self._nn(inputs=routing_dataclass.normalized_spatial_attributes)
        leakance_params = self._run_leakance(q_prime, ...) if self._leakance else None
        return self._model(topology=topology, lateral_inflow=q_prime,
                          spatial_params=spatial_params, carry_state=carry_state,
                          leakance_params=leakance_params)
```

Scripts become thin (~30 lines each) — just config loading, DataLoader setup, and `orchestrator.route_batch()` in a loop.

---

## MuskingumCunge Decomposition

### Before: `MuskingumCunge` (~731 lines, god class)

```
__init__()              -> stores config, empty state containers
setup_inputs()          -> 4-step init (network + denorm + discharge + scatter)
setup_leakance_params() -> denorm LSTM outputs
forward()               -> full timestep loop + progress bar + output assembly
route_timestep()        -> single timestep physics + sparse solve
set_progress_info()     -> UI concern
+ ~5 module-level physics functions
```

### After: `MuskingumCungeRouter` (~250-300 lines)

```
__init__(device)
initialize(topology, bounds)   -> creates PatternMapper, scatter indices
coldstart(q_prime_t0)          -> hotstart discharge via topo accumulation
update(timestep_input)         -> single timestep (maps to gRPC Update)
update_until(inflow, params)   -> full window loop (maps to gRPC UpdateUntil)
finalize()                     -> return state snapshot
get_state() / set_state()      -> explicit state management
set_leakance(params, bounds)   -> denorm leakance into RoutingState
```

Progress bar **removed** from router — caller provides optional `progress_callback`.

### `dmc` wrapper refactored (~100 lines, down from ~368)

```python
class dmc(torch.nn.Module):
    def __init__(self, cfg, device="cpu", router=None):
        self._bounds = PhysicalBounds.from_config(cfg)
        self._router = router or MuskingumCungeRouter(device=device)

    def forward(self, topology, lateral_inflow, spatial_params,
                carry_state=False, leakance_params=None):
        self._router.initialize(topology, self._bounds)
        denorm = denormalize_params(spatial_params, self._bounds, topology)
        if not carry_state:
            self._router.coldstart(lateral_inflow[0])
        if leakance_params:
            self._router.set_leakance(leakance_params, self._bounds)
        output = self._router.update_until(lateral_inflow, denorm, leakance_params)
        result = {"runoff": output}
        state = self._router.get_state()
        if state.zeta_sum is not None:
            result["zeta_sum"] = state.zeta_sum
            result["q_prime_sum"] = state.q_prime_sum
        return result
```

No more: 12 duplicated attrs, `fill_op()`, `_sparse_eye()`, `_sparse_diag()`, `route_timestep()` compat methods, `set_progress_info()`.

---

## gRPC Service Mapping

```
gRPC Endpoint            -> Python Method
Initialize(request)      -> router.initialize(topology, bounds) + router.coldstart(q_t0)
Update(request)          -> router.update(timestep_input)
UpdateUntil(request)     -> router.update_until(inflow, params)
Finalize(request)        -> router.finalize()
GetState(request)        -> router.get_state()
SetState(request)        -> router.set_state(state)
```

The BMI lifecycle maps 1:1 to gRPC endpoints — no impedance mismatch. The server holds a `MuskingumCungeRouter` instance per session. `RoutingState.snapshot()` enables checkpoint/restore over the wire.

gRPC is **inference-only** (no autograd over the wire). Training stays in-process. The gRPC layer will be **stubs with `NotImplementedError`** in the initial implementation — full gRPC will be a future PR.

---

## Verification Strategy

### Current Testing Gap

The existing test suite (37 files, ~7500 lines) has **no numerical regression tests**. Key finding:

- `triangular_sparse_solve()` is **always mocked** in unit tests — the actual solver is never exercised
- Tests check structural properties (shape, dtype, NaN/Inf) and behavioral properties (downstream accumulation, mass balance at 5%)
- **No test compares old output vs new output at tight tolerance**
- The Sandbox network (5 reaches, 238 timesteps) runs end-to-end in `run_ddr_routing()` but only checks mass balance and monotonicity — not actual discharge values

This means: if the refactor introduces a subtle numerical bug (e.g., off-by-one in timestep indexing, wrong coefficient ordering), **no existing test would catch it**.

### Step 0: Capture Reference Outputs (BEFORE any refactoring)

Create `tests/routing/reference_data/` with golden outputs from the current (correct) code. This runs once, checked into git, never changes.

```python
# tests/routing/conftest.py — new fixture
@pytest.fixture(scope="session")
def reference_outputs(sandbox_zarr_path, sandbox_hourly_qprime):
    """Capture ground-truth outputs from current code for regression testing."""
    cfg = create_ddr_config()
    rd, ts_order = create_routing_dataclass(sandbox_zarr_path)
    mock_flow = MockStreamflow(sandbox_hourly_qprime)
    mock_kan = MockKAN(num_reaches=5, learnable_params=["n", "q_spatial", "top_width", "side_slope"])
    routing_model = dmc(cfg=cfg, device="cpu")

    q_prime = mock_flow(routing_dataclass=rd, device="cpu", dtype=torch.float32)
    spatial_params = mock_kan(inputs=rd.normalized_spatial_attributes)
    output = routing_model(
        routing_dataclass=rd, spatial_parameters=spatial_params,
        streamflow=q_prime, carry_state=False,
    )

    return {
        "runoff": output["runoff"].detach().clone(),
        "discharge_t": routing_model._discharge_t.detach().clone(),
        "q_prime": q_prime.detach().clone(),
        "spatial_params": {k: v.detach().clone() for k, v in spatial_params.items()},
    }
```

**What gets captured (checked into `tests/routing/reference_data/`):**

| Tensor | Shape | What it proves |
|---|---|---|
| `runoff` | `(num_gauges, 238)` | Full routed output at every timestep for every gauge |
| `discharge_t` | `(5,)` | Final discharge state after all 238 timesteps |
| `discharge_per_timestep` | `(238, 5)` | Discharge at every single timestep (catches off-by-one) |
| `hotstart_discharge` | `(5,)` | Cold-start initialization values |
| `muskingum_coefficients` | `(5, 4)` | c1, c2, c3, c4 for each reach |
| `velocity` | `(5,)` | Trapezoid velocity per reach |
| `q_prime` | `(238, 5)` | Lateral inflow (input, should be unchanged) |

### Step 1: Numerical Regression Test (runs after EVERY phase)

```python
# tests/routing/test_regression.py
class TestNumericalRegression:
    """Proves refactored code produces bit-identical output to original code."""

    def test_forward_pass_matches_reference(self, reference_outputs, ...):
        output = routing_model(...)  # Uses refactored code path
        torch.testing.assert_close(output["runoff"], reference_outputs["runoff"],
                                   atol=1e-6, rtol=1e-5)

    def test_discharge_state_matches_reference(self, reference_outputs, ...):
        torch.testing.assert_close(routing_model._discharge_t, reference_outputs["discharge_t"],
                                   atol=1e-6, rtol=1e-5)

    def test_per_timestep_discharge_matches(self, reference_outputs, ...):
        for t in range(238):
            torch.testing.assert_close(new_discharge_at_t[t],
                                       reference_outputs["discharge_per_timestep"][t],
                                       atol=1e-6, rtol=1e-5,
                                       msg=f"Divergence at timestep {t}")

    def test_hotstart_matches_reference(self, reference_outputs, ...):
        torch.testing.assert_close(new_hotstart, reference_outputs["hotstart_discharge"],
                                   atol=1e-6, rtol=1e-5)

    def test_muskingum_coefficients_match(self, reference_outputs, ...):
        torch.testing.assert_close(new_coefficients, reference_outputs["muskingum_coefficients"],
                                   atol=1e-7, rtol=1e-6)  # Tighter — pure math
```

**Tolerance rationale:**
- `atol=1e-6, rtol=1e-5` for accumulated outputs (238 timesteps of float32 accumulation)
- `atol=1e-7, rtol=1e-6` for single-step pure math (no accumulation)

### Step 2: Gradient Regression Test

```python
# tests/routing/test_gradient_regression.py
class TestGradientRegression:
    def test_kan_gradients_match_reference(self, reference_gradients, ...):
        output = routing_model(...)
        loss = output["runoff"].sum()
        loss.backward()
        for name, param in nn.named_parameters():
            torch.testing.assert_close(param.grad, reference_gradients[name],
                                       atol=1e-5, rtol=1e-4)

    def test_spatial_param_gradients_match(self, ...):
        output = routing_model(..., retain_grads=True)
        loss = output["runoff"].sum()
        loss.backward()
        torch.testing.assert_close(routing_model.n.grad, reference_gradients["n"],
                                   atol=1e-5, rtol=1e-4)
```

### Step 3: Unmocked Sparse Solve Test (NEW — fills existing gap)

```python
# tests/routing/test_sparse_solve_real.py
class TestRealSparseSolve:
    def test_sandbox_real_solve(self, sandbox_zarr_path, sandbox_hourly_qprime):
        output = run_ddr_routing(sandbox_zarr_path, sandbox_hourly_qprime)
        torch.testing.assert_close(output, reference_output, atol=1e-6, rtol=1e-5)

    def test_linear_chain_analytical(self):
        """3-reach linear chain with known analytical solution."""
        # Hand-calculated Muskingum discharge for constant inflow
        assert_close(actual, expected, atol=1e-6)
```

### Step 4: Carry-State Continuity Test (NEW — fills existing gap)

```python
# tests/routing/test_carry_state.py
class TestCarryState:
    def test_two_batches_equals_one_run(self, ...):
        """Split 238 timesteps into [0:120] + [120:238].
        Must be bit-identical to single run."""
        output_single = run_ddr_routing(full_qprime)
        output_batch1 = run_ddr_routing(qprime[:120])
        output_batch2 = run_ddr_routing(qprime[120:], carry_state=True)
        output_concat = torch.cat([output_batch1, output_batch2], dim=1)
        torch.testing.assert_close(output_single, output_concat, atol=1e-6, rtol=1e-5)
```

### Step 5: Leakance Regression Test (NEW)

```python
# tests/routing/test_leakance_regression.py
class TestLeakanceRegression:
    def test_leakance_forward_matches_reference(self, ...):
        output = routing_model(..., leakance_params=mock_leakance_params)
        torch.testing.assert_close(output["runoff"], reference_leakance_output["runoff"], ...)
        torch.testing.assert_close(output["zeta_sum"], reference_leakance_output["zeta_sum"], ...)

    def test_leakance_vs_no_leakance_difference(self, ...):
        """With leakance_factor=0, output must be identical to no-leakance path."""
        torch.testing.assert_close(output_no_leakance["runoff"], output_zero_leakance["runoff"])
```

### Step 6: Physics Function Equivalence (Phase 1 specific)

```python
# tests/routing/test_physics.py
class TestPhysicsExtraction:
    @pytest.mark.parametrize("scenario", create_test_scenarios())
    def test_trapezoid_velocity_matches_original(self, scenario):
        old_result = mc._get_trapezoid_velocity(...)
        new_result = physics.compute_trapezoid_velocity(...)
        torch.testing.assert_close(old_result, new_result, atol=0, rtol=0)  # EXACT match

    def test_muskingum_coefficients_matches_original(self, ...):
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_hotstart_matches_original(self, ...):
        torch.testing.assert_close(old, new, atol=0, rtol=0)

    def test_denormalize_matches_original(self, ...):
        torch.testing.assert_close(old_n, new.n, atol=0, rtol=0)
```

**Note: `atol=0, rtol=0`** — extracted functions must produce **exact** same output since they're the same math, just moved to a different file.

### Step 7: Autograd Correctness (Phase 1-2)

```python
# tests/routing/test_autograd.py
class TestAutogradCorrectness:
    def test_trapezoid_velocity_gradcheck(self):
        inputs = (q_t.double(), n.double(), q_spatial.double(), s0.double(), ...)
        for inp in inputs:
            inp.requires_grad_(True)
        assert torch.autograd.gradcheck(physics.compute_trapezoid_velocity, inputs,
                                        eps=1e-6, atol=1e-4, rtol=1e-3)

    def test_full_chain_backward(self):
        output = routing_model(...)
        loss = output["runoff"].sum()
        loss.backward()
        for name, param in nn.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert param.grad.abs().max() > 0, f"Zero gradient for {name}"
```

### Test Execution Order Per Phase

| Phase | Run These Tests | Pass Criteria |
|---|---|---|
| **Before Phase 1** | Capture reference outputs → commit to `tests/routing/reference_data/` | Reference captured successfully |
| **Phase 1** (Value Objects) | `test_physics.py` (exact match), `test_autograd.py` (gradcheck), all existing tests | Zero failures, exact physics match |
| **Phase 2** (BMI Lifecycle) | `test_regression.py` (tight tolerance), `test_gradient_regression.py`, `test_carry_state.py`, `test_sparse_solve_real.py` | All within tolerance, carry-state bit-identical |
| **Phase 3** (dmc Refactor) | `test_regression.py`, `test_gradient_regression.py`, deprecated bridge tests | Typed forward == kwargs forward |
| **Phase 4** (Orchestrator) | `test_orchestrator.py` matches manual script logic, all regression tests | Orchestrator output == direct dmc output |
| **Phase 5** (Benchmarks) | `test_benchmark_e2e.py`, adapter tests, all regression tests | Same metrics, same plots |
| **Every phase** | `uv run pytest tests/` (full suite) + `uv run pre-commit run --all-files` | Zero failures, zero lint errors |

### What This Catches That Current Tests Don't

| Bug Type | Current Tests | New Regression Tests |
|---|---|---|
| Off-by-one in timestep loop | NOT caught (mocked solver) | Caught by per-timestep comparison |
| Wrong Muskingum coefficient order | NOT caught (mocked solver) | Caught by coefficient comparison |
| Broken hotstart after extraction | Partially caught (checks cumulative sum) | Caught by exact match |
| Gradient graph broken by `.detach()` | Partially caught (checks grad exists) | Caught by gradient magnitude comparison |
| Carry-state bug across batches | NOT caught (no multi-batch test) | Caught by 2-batch vs 1-run comparison |
| Leakance zeta sign flip | Partially caught (checks losing/gaining) | Caught by exact output comparison |
| Sparse solve result changed | NOT caught (always mocked) | Caught by real solver test |
| Denormalization range shift | NOT caught (mock returns 0.5) | Caught by exact denorm comparison |

---

## Plotting & Examples

### Current plotting architecture

Plotting functions in `src/ddr/validation/plots.py` are **already well-separated** — they consume numpy arrays and metric dicts, not routing objects. They don't change:

| Function | Inputs | Output |
|---|---|---|
| `plot_time_series()` | `prediction: np.ndarray`, `observation: np.ndarray`, `time_range`, `gage_id`, `metrics: dict`, `additional_predictions: list[tuple]` | Single-gage hydrograph PNG |
| `plot_cdf()` | `list[np.ndarray]` (metric arrays per model), `labels`, `colors` | CDF comparison PNG |
| `plot_box_fig()` | `list[list[np.ndarray]]` (metrics x models), `labels` | 6-panel box plot PNG |
| `plot_drainage_area_boxplots()` | `metrics: dict`, `gage_info: pd.DataFrame` | Area-binned box plot PNG |
| `plot_gauge_map()` | `gages: pd.DataFrame` with metric column + lat/lon | Scatter map PNG |

### What changes for benchmarks

Currently, benchmark-specific plotting logic (~270 lines) lives inline in `benchmark.py:311-587`. This moves to `benchmarks/src/ddr_benchmarks/plots.py`:

```python
def generate_comparison_plots(
    results: BenchmarkResults,
    metrics: dict[str, Metrics],    # model_name -> Metrics
    save_path: Path,
) -> None:
    """Generate all benchmark comparison plots."""
    # CDF plots (NSE, KGE) — one per metric, all models overlaid
    # Box plot — 6 panels (bias, rmse, fhv, flv, nse, kge), all models
    # Gauge maps — one per model per metric (NSE), plus difference maps
    # Hydrographs — one per gage, all models overlaid
```

The input is `BenchmarkResults` (the new structured container), not raw arrays. The plotting functions from `validation/plots.py` are called inside — they don't change.

### Examples

Add `examples/bmi_lifecycle.py` — a standalone script showing the BMI lifecycle directly:

```python
"""Example: Using the BMI lifecycle API directly (no scripts/orchestrator)."""
from ddr.routing.protocols import NetworkTopology, PhysicalBounds
from ddr.routing.mmc import MuskingumCungeRouter
from ddr.routing.physics import denormalize_params, compute_hotstart_discharge

# 1. Build topology from your data
topology = NetworkTopology(
    adjacency_matrix=my_sparse_csr,
    length=my_lengths,
    slope=my_slopes,
    x_storage=my_x,
    top_width=torch.empty(0),   # Will be learned by KAN
    side_slope=torch.empty(0),
    ...
)

# 2. Create bounds from config
bounds = PhysicalBounds.from_config(cfg)

# 3. Initialize router
router = MuskingumCungeRouter(device="cpu")
router.initialize(topology, bounds)
router.coldstart(q_prime[0])

# 4. Route all timesteps
output = router.update_until(q_prime, denorm_params)

# 5. Get final state for next batch
state = router.finalize()

# 6. Continue from saved state
router.set_state(state)
output2 = router.update_until(q_prime_batch2, denorm_params)
```

Add `examples/benchmark_custom_model.py` — how to plug a new model into the benchmark:

```python
"""Example: Adding a custom routing model to the benchmark comparison."""
from ddr_benchmarks.adapters.protocols import ModelAdapter, ModelOutput

class MyCustomAdapter:
    name = "MyModel"

    def run(self, dataset, dataloader, device) -> ModelOutput:
        # Your custom routing logic here
        predictions = ...
        return ModelOutput(predictions=predictions, gage_ids=gage_ids, metadata={})

# Register in benchmark config or pass to BenchmarkRunner
runner = BenchmarkRunner(
    adapters=[DDRAdapter(orchestrator), MyCustomAdapter(), DiffRouteAdapter(...)],
    dataset=dataset,
)
results = runner.run_all()
runner.generate_plots(results, runner.compute_metrics(results))
```

---

## Documentation Changes

Bare minimum, "explain it to dummies" docs. No verbose prose — just enough that someone new can find their way.

### New Docs (Per Phase)

| Phase | File | What it says |
|---|---|---|
| Phase 1 | `src/ddr/routing/protocols.py` — module docstring | "Value objects and Protocol for routing. NetworkTopology = network structure, PhysicalBounds = config-derived limits, RoutingState = mutable discharge state, RoutingModel = BMI lifecycle interface." |
| Phase 1 | `src/ddr/routing/physics.py` — module docstring | "Pure physics functions for Muskingum-Cunge routing. No state, no side effects. Extracted from MuskingumCunge class for testability and reuse." |
| Phase 2 | `MuskingumCungeRouter` — class docstring | "BMI-style router: initialize() → coldstart() → update_until() → finalize(). Single-responsibility: routes flow through a network given topology and parameters." |
| Phase 3 | `dmc.forward()` — updated docstring | Replace `**kwargs` docs with typed param descriptions. One line per param: name, shape, units. |
| Phase 4 | `src/ddr/routing/orchestrator.py` — module docstring | "Shared batch logic for all scripts. Wraps flow → KAN → dmc. Scripts just loop over DataLoader and call route_batch()." |
| Phase 5 | `benchmarks/src/ddr_benchmarks/adapters/protocols.py` — module docstring | "ModelAdapter Protocol: implement run() to add a new routing model to benchmarks." |

### Existing Docs to Update

| File | What changes |
|---|---|
| `docs/index.md` | Add 2-sentence mention of BMI lifecycle under Architecture section |
| `docs/startup.md` | Add note: "See config/example_*.yaml for config templates. Copy and edit with your paths." |
| `MuskingumCunge` in `mmc.py` | Add missing class docstring (currently only has module docstring) |
| `kan.py` `forward()` | Add minimal docstring: inputs shape, outputs shape |

### Docs NOT Changed

Everything else — `docs/datasets.md`, `docs/gpu.md`, `docs/references.md`, `README.md`, `benchmarks/README.md` — stays as-is. The refactor doesn't change data formats, GPU setup, or project identity.

### Docstring Convention

NumPy-style, enforced by Ruff pydocstyle. All new public classes and functions get:
```python
"""One-liner summary.

Parameters
----------
param : type
    Shape (X, Y), units if applicable.

Returns
-------
type
    Shape (X, Y), what it contains.
"""
```

Private methods (`_foo`) get a one-liner only. No docstrings on obvious properties.

---

## Config File Strategy

### Current Problem

Production configs contain hardcoded HPC paths (`/projects/mhpi/...`) and are committed to git. This means:
- Cloning the repo gives you broken configs (paths don't exist on your machine)
- Modifying a config for your run risks accidentally committing your paths

### New Pattern: Example Configs + Gitignored Working Configs

**Committed to git (examples):**
```
config/
├── example_training.yaml       # Template with placeholder paths + inline comments
├── example_testing.yaml        # Template
├── example_routing.yaml        # Template
├── example_merit_training.yaml # MERIT-specific template
└── hydra/settings.yaml         # Framework settings (no paths)

benchmarks/config/
├── example_benchmark.yaml      # Template with placeholder paths
└── hydra/settings.yaml
```

**Gitignored (user creates by copying example):**
```
config/
├── training.yaml               # Your paths, your batch_size, your checkpoint
├── testing.yaml
├── routing.yaml
├── merit_training.yaml
└── ...any other *.yaml (except example_*)

benchmarks/config/
├── benchmark.yaml              # Your benchmark paths
└── ...
```

### .gitignore Changes

```gitignore
# Config files (gitignore working copies, keep examples)
config/*.yaml
!config/example_*.yaml
!config/hydra/
benchmarks/config/*.yaml
!benchmarks/config/example_*.yaml
!benchmarks/config/hydra/
```

### Example Config Format

Each example config uses **placeholder paths** and **inline comments**:

```yaml
# example_training.yaml — Copy to training.yaml and fill in your paths
# Usage: cp config/example_training.yaml config/training.yaml
# Then: uv run python scripts/train.py --config-name training

geodataset: merit  # or lynker_hydrofabric

data_sources:
  attributes: /path/to/your/merit_global_attributes_v2.nc  # NetCDF with reach attributes
  geospatial_fabric_gpkg: /path/to/your/merit_shapefile    # Flowpath geometries
  conus_adjacency: /path/to/your/adjacency.zarr            # Network connectivity
  streamflow: /path/to/your/streamflow_data                # Lateral inflow source
  observations: /path/to/your/usgs_observations            # USGS daily discharge
  gages: /path/to/your/gage_info.csv                       # Gage metadata

experiment:
  batch_size: 64        # Number of gages per batch (spatial, not temporal)
  epochs: 50            # Training epochs
  warmup: 3             # Days excluded from loss (cold-start stabilization)
  rho: 365              # Days per random time window
  learning_rate:
    1: 0.001            # LR for epochs 1+
    30: 0.0001          # LR for epochs 30+
  checkpoint: null      # Path to resume from, or null for fresh start

params:
  use_leakance: false   # Enable GW-SW exchange (requires LSTM)
  tau: 3                # Time shift parameter
```

### Migration

- **Phase 1**: Rename current committed configs to `example_*` prefix. Update `.gitignore`.
- **Phase 4**: When scripts use orchestrator, update example configs if any keys change.
- **No functional change** — Hydra `--config-name` flag picks which YAML to load.

---

## Dependency Injection Opportunities (Bloch Item 5)

**Item 5: Prefer dependency injection to hardwiring resources.** The codebase has numerous hard-wired dependencies — constructors that open files, create internal objects, and branch on boolean flags.

### Tier 1: High-Impact (unblocks most unit tests)

| Component | Hard-Wired Dependency | Inject Instead | Testing Gain |
|---|---|---|---|
| `StreamflowReader.__init__()` | `read_ic()` opens Icechunk store | `xr.Dataset` | No I/O in unit tests |
| `IcechunkUSGSReader.__init__()` | `read_ic()` + `read_gage_info()` | `xr.Dataset` + `dict` | No file reads |
| `AttributesReader.__init__()` | `read_ic()` or `open_mfdataset()` | `xr.Dataset` | Single code path tested |
| `dmc.__init__()` | `MuskingumCunge(cfg, device)` created internally | `MuskingumCungeRouter` instance | Inject spy/mock router |
| `Merit.__init__()` | `xr.open_mfdataset()`, `gpd.read_file()`, `read_zarr()` | `xr.Dataset`, `gpd.GeoDataFrame`, adjacency | Tests run without 46 GB HPC data |

**Pattern**: Optional constructor param with `None` default; `None` triggers real I/O.

### Tier 2: Strategy Pattern (eliminates boolean branching)

| Component | Current Flag | Strategy Injection | Testing Gain |
|---|---|---|---|
| Leakance (`mmc.py`) | `use_leakance: bool` -> 6+ if/else branches | `LeakanceStrategy` Protocol | Test leakance in isolation |
| Sparse solver (`utils.py`) | `if device == "cpu": scipy else: cupy` | `SparseSolver` Protocol | Test GPU logic on CPU |

### Tier 3: Configuration as Value Objects

| Component | Hard-Wired | Inject Instead | Testing Gain |
|---|---|---|---|
| `MuskingumCunge.__init__()` | Reads `cfg.params.*` -> 6+ tensors | `PhysicalBounds` frozen dataclass | Custom bounds per test |
| `Merit._init_training()` | Creates `IcechunkUSGSReader` internally | Accept reader instance | Swap mock reader |

### Where DI Fits in Migration

- **Phase 1** creates `PhysicalBounds` and `NetworkTopology` -> injectable types
- **Phase 2** introduces `MuskingumCungeRouter.initialize()` -> natural injection point
- **Phase 3** refactors `dmc` -> accepts router as constructor param
- Tier 2 (strategies) can be done at any phase

---

## Migration Phases

### Phase 1: Extract Value Objects (LOW RISK)
- Create `protocols.py` with `NetworkTopology`, `PhysicalBounds`, `RoutingState`, `DenormalizedParams`, `RoutingModel`
- Create `physics.py` — move 5 pure functions from mmc.py (no math changes)
- Move `denormalize()` from utils.py to physics.py (keep re-export)
- Add `from_routing_dataclass()` and `from_config()` factory methods
- **Capture reference outputs before starting** (Level 2 regression baseline)
- **No behavioral change. All existing tests pass.**

### Phase 2: BMI Lifecycle on MuskingumCunge (MEDIUM RISK)
- Rename to `MuskingumCungeRouter`, add `initialize/coldstart/update/update_until/finalize/get_state/set_state`
- Keep `setup_inputs()` and `forward()` as deprecated bridges calling new methods
- Remove tqdm from router — add `progress_callback` param
- Replace instance attrs with `RoutingState` container
- **Existing tests pass via deprecated bridges. New BMI + regression tests added.**

### Phase 3: Refactor dmc (MEDIUM RISK)
- Typed `forward()` signature (no `**kwargs`)
- Remove all duplicated attributes
- Accept `router` as optional constructor param (DI)
- **Backward-compat bridge**: detect old kwargs, convert, emit `DeprecationWarning`
- Simplify `state_dict()`/`load_state_dict()`

### Phase 4: Orchestrator + Script Simplification (LOW RISK)
- Create `orchestrator.py` with `RoutingOrchestrator`
- Refactor 4 scripts (train, test, router, benchmark) to use it
- ~30 lines of duplicated logic per script eliminated

### Phase 5: Benchmark Decomposition (LOW RISK)
- Extract `ModelAdapter` Protocol, `BenchmarkResults` dataclass, `BenchmarkRunner`
- Create `DDRAdapter`, `DiffRouteAdapter`, `BaselineAdapter`
- Extract benchmark plotting to `benchmarks/src/ddr_benchmarks/plots.py`
- Add example script: `examples/benchmark_custom_model.py`

### Phase 6: gRPC Service Layer (ADDITIVE, LOW RISK)
- `src/ddr/serving/` package with proto, server, client
- Maps 1:1 to BMI lifecycle from Phase 2
- **Initial implementation: stubs with `NotImplementedError`** — full gRPC in a future PR
- Add `grpcio` extras to pyproject.toml
- Add example script: `examples/bmi_lifecycle.py`

```
Phase 1 --> Phase 2 --> Phase 3 --+--> Phase 4 --> Phase 5
                                  +--> Phase 6
```

Phases 4-5 and Phase 6 are independent tracks after Phase 3.

---

## Files Changed

| File | Phase | Action |
|---|---|---|
| `src/ddr/routing/protocols.py` | 1 | NEW — Protocol, value objects |
| `src/ddr/routing/physics.py` | 1 | NEW — pure physics functions |
| `src/ddr/routing/mmc.py` | 1-2 | MODIFY — extract functions, then BMI lifecycle |
| `src/ddr/routing/utils.py` | 1 | MODIFY — denormalize() moves, re-export kept |
| `src/ddr/routing/torch_mc.py` | 3 | MODIFY — typed forward, no duplicated state, DI for router |
| `src/ddr/routing/orchestrator.py` | 4 | NEW — shared script loop logic |
| `scripts/train.py` | 4 | MODIFY — use orchestrator |
| `scripts/test.py` | 4 | MODIFY — use orchestrator |
| `scripts/router.py` | 4 | MODIFY — use orchestrator |
| `benchmarks/scripts/benchmark.py` | 4-5 | MODIFY — use orchestrator, then adapter pattern |
| `benchmarks/src/ddr_benchmarks/benchmark.py` | 5 | MODIFY — decompose into BenchmarkRunner |
| `benchmarks/src/ddr_benchmarks/adapters/` | 5 | NEW — ModelAdapter protocol + adapters |
| `benchmarks/src/ddr_benchmarks/results.py` | 5 | NEW — BenchmarkResults dataclass |
| `benchmarks/src/ddr_benchmarks/plots.py` | 5 | NEW — extracted benchmark plotting |
| `src/ddr/serving/` | 6 | NEW — gRPC stubs |
| `examples/bmi_lifecycle.py` | 6 | NEW — BMI lifecycle example |
| `examples/benchmark_custom_model.py` | 5 | NEW — custom model adapter example |

**Untouched**: `kan.py`, `leakance_lstm.py`, `dataclasses.py`, `base_geodataset.py`, `merit.py`, `lynker_hydrofabric.py`, `readers.py`, `configs.py`, `validation/plots.py`, `validation/metrics.py`, all data/config files.

---

## Entry Point Impact Analysis

The refactor touches 6 distinct entry points. This section traces each one end-to-end.

### Entry Point 1: Training (`scripts/train.py`)

**Current flow:**
```
main() [Hydra]
  → validate_config() → Config
  → Instantiate: kan, leakance_lstm?, dmc, StreamflowReader
  → train():
    → Dataset + DataLoader + RandomSampler
    → Optional: load_checkpoint() → restore optimizer, epoch, nn weights
    → For epoch in range(epochs):
      → LR scheduling (cfg.experiment.learning_rate: {epoch: lr})
      → For batch in dataloader:
        → flow() → q_prime (T_hourly, N)
        → nn() → spatial_params {name: (N,)}
        → [leakance_nn() → leakance_params]
        → routing_model() → dmc_output {"runoff": (N, T_hourly)}
        → downsample() → daily_runoff
        → Filter NaN observations
        → mse_loss(predictions[warmup:], observations[warmup:])
        → loss.backward() → optimizer.step()
        → Metrics(pred, target) → per-batch NSE/KGE/RMSE
        → plot_time_series() → training progress plots
        → save_state() → checkpoint to disk
```

**What changes (Phase 4):**
- Lines between `flow()` and `dmc_output` collapse into `orchestrator.route_batch()`
- Everything else stays: Dataset, DataLoader, RandomSampler, collate_fn, loss computation, backward, optimizer, LR scheduling, checkpoint save/load, Metrics, training plots
- `carry_state=False` always (random time windows)

**What does NOT change:**
- `load_checkpoint()` / `save_state()` — orchestrator doesn't own checkpointing
- Optimizer setup (Adam, params from nn + leakance_nn)
- LR scheduling logic (dict of {epoch: lr})
- Loss function (MSE with warmup skip)
- Training-time plotting (plot_time_series called per-batch)
- Config loading (Hydra pattern)

**Output:** `cfg.params.save_path/saved_models/*.pt`, `cfg.params.save_path/plots/*.png`

---

### Entry Point 2: Evaluation (`scripts/test.py`)

**Current flow:**
```
main() [Hydra]
  → validate_config() → Config (mode=INFERENCE)
  → test():
    → Dataset + DataLoader + SequentialSampler (deterministic order)
    → load_checkpoint() → restore nn weights
    → nn.eval(), leakance_nn.eval()
    → with torch.no_grad():
      → For batch i in dataloader:
        → flow() → q_prime
        → nn() → spatial_params
        → routing_model(..., carry_state=i > 0) → dmc_output
        → predictions[:, hourly_indices] = dmc_output["runoff"]
      → compute_daily_runoff() → daily predictions
      → Save xr.Dataset → model_test.zarr (predictions + observations)
      → Compute Metrics → print summary
      → Print: "Run examples/eval/evaluate.ipynb for plots"
```

**What changes (Phase 4):**
- Inner loop collapses to `orchestrator.route_batch(routing_dataclass, carry_state=i > 0)`
- `carry_state=i > 0` is critical — orchestrator must accept this

**What does NOT change:**
- SequentialSampler, zarr output, daily downsampling, Metrics summary
- Checkpoint loading, eval mode, no_grad context

**Output:** `model_test.zarr` with dims `[gage_ids, time]`

---

### Entry Point 3: Forward Routing (`scripts/router.py`)

**Current flow:**
```
main() [Hydra]
  → validate_config() → Config (mode=INFERENCE)
  → route_trained_model():
    → Dataset + DataLoader + SequentialSampler
    → load_checkpoint()
    → with torch.no_grad():
      → For batch i in dataloader:
        → flow() → q_prime
        → nn() → spatial_params
        → routing_model(..., carry_state=i > 0)
        → predictions[:, hourly_indices] = output["runoff"]
        → [If leakance: accumulate zeta_sum, q_prime_sum]
      → compute_daily_runoff()
      → Save xr.Dataset → chrout.zarr
        → coords: {divide_ids or gage_ids, time}
        → variables: {predictions, [zeta_sum, q_prime_sum]}
```

**Difference from test.py:**
- Outputs `chrout.zarr` (not `model_test.zarr`)
- Supports 3 output modes: target catchments, gage mode, full domain
- Includes leakance accumulation output (zeta_sum, q_prime_sum)
- Does NOT compute metrics (no observations available in all modes)

**What changes (Phase 4):**
- Same inner loop collapse via orchestrator
- Orchestrator `route_batch()` return dict includes `zeta_sum`/`q_prime_sum` when leakance enabled — this must be preserved

**Output:** `chrout.zarr` with variable set depending on leakance config

---

### Entry Point 4: Benchmarking (`benchmarks/scripts/benchmark.py`)

**Current flow (866-line god function):**
```
main() [Hydra]
  → validate_benchmark_config() → BenchmarkConfig {ddr, diffroute, summed_q_prime}
  → benchmark():
    Phase 1: DDR Routing (same as test.py loop)
    Phase 2: DiffRoute Per-Gage Routing
      → For each gage: build NetworkX DiGraph, RivTree, LTIRouter
      → Route individually, collect in array
    Phase 3: Filter to common gages (DiffRoute → NaN for headwaters)
    Phase 4: Downsample + Metrics (DDR, DiffRoute, SummedQ')
    Phase 5: Mass balance check
    Phase 6: Plots (calls validation/plots.py functions)
      → CDF, box plots, gauge maps, hydrographs
    Phase 7: Save → benchmark_results.zarr
```

**What changes (Phase 4-5):**
- Phase 1 (DDR) uses DDRAdapter wrapping orchestrator
- Phase 2 (DiffRoute) becomes DiffRouteAdapter
- Phase 3 (SummedQ') becomes BaselineAdapter
- BenchmarkRunner orchestrates adapters + common filtering
- Plotting extracted to `benchmarks/src/ddr_benchmarks/plots.py`

**What does NOT change:**
- DiffRoute per-gage logic (just wrapped in adapter)
- Metric computation (Metrics class)
- Plot functions (validation/plots.py)
- Zarr output format

**Output:** `plots/{cdf,boxplot,gauge_map,hydrographs}/*.png`, `benchmark_results.zarr`

---

### Entry Point 5: Plots & Examples for Papers

**Existing plotting infrastructure:**

| Entry Point | Location | Purpose | Affected? |
|---|---|---|---|
| Training plots | `train.py` calls `plot_time_series()` | Monitor convergence per-batch | No |
| Evaluation notebook | `examples/eval/evaluate.ipynb` | Post-training metrics & figures | No |
| Parameter maps | `examples/parameter_maps/plot_parameter_map.ipynb` | Visualize learned n, width, etc. | No |
| Leakance impact | `examples/leakance/plot_leakance_impact.ipynb` | Show GW-SW exchange effect | No |
| Benchmark plots | `benchmark.py:311-587` | DDR vs DiffRoute comparison | Yes → moves to plots.py |
| BMI lifecycle example | (new) `examples/bmi_lifecycle.py` | Demonstrate BMI API directly | New in Phase 6 |
| Custom model example | (new) `examples/benchmark_custom_model.py` | Show how to plug in a model | New in Phase 5 |

**Key insight:** Plotting functions in `validation/plots.py` are already well-separated. They consume numpy arrays + metrics dicts. The refactor doesn't change them. Only the benchmark's inline plotting code (~270 lines) gets extracted to a dedicated module.

**Existing notebooks are untouched** — they read from zarr output files (model_test.zarr, chrout.zarr) which keep the same format.

---

### Entry Point 6: Pytest Testing (`tests/`)

**Current test infrastructure (37 test files, ~7500 lines):**

```
tests/
├── conftest.py                     # Root: 603 lines of fixtures (25-reach networks, mock gages)
├── benchmarks/conftest.py          # Session: Sandbox network, MockStreamflow, MockKAN
├── routing/test_mmc.py             # MuskingumCunge unit tests
├── routing/test_torch_mc.py        # dmc wrapper tests
├── routing/test_utils.py           # PatternMapper, TriangularSparseSolver
├── routing/test_leakance.py        # Leakance module tests
├── routing/gradient_utils.py       # Gradient test helpers
├── nn/test_kan.py                  # KAN forward/backward
├── nn/test_leakance_lstm.py        # LSTM forward/backward
├── engine/merit/                   # Merit geodataset tests
├── engine/lynker_hydrofabric/      # Lynker geodataset tests
├── validation/test_configs.py      # Config schema tests
├── validation/test_metrics.py      # NSE, KGE, RMSE tests
├── io/test_readers.py              # StreamflowReader tests
└── ... (20+ more files)
```

**Existing mocks/fixtures:**
- `MockStreamflow(nn.Module)` — returns fixed qprime array
- `MockKAN(nn.Module)` — returns all params = 0.5
- `create_ddr_config()` → minimal Config
- `create_routing_dataclass()` → from Sandbox adjacency zarr
- `run_ddr_routing()` → full DDR pipeline on Sandbox
- `simple_merit_flowpaths` → 25-reach test GeoDataFrame
- `sandbox_network` → 5-reach NetworkX DiGraph (k=0.1042, x=0.3)
- `sandbox_hourly_qprime` → (238, 5) tensor

**Impact per refactoring phase:**

| Phase | Tests Affected | How |
|---|---|---|
| Phase 1 (Value Objects) | None break | New tests added: `test_physics.py`, `test_protocols.py` |
| Phase 2 (BMI Lifecycle) | `test_mmc.py` | Deprecated bridges keep existing calls working. New `test_mmc_bmi.py` added |
| Phase 3 (dmc Refactor) | `test_torch_mc.py` | Backward-compat bridge: old `**kwargs` still works with `DeprecationWarning`. New typed-signature tests added |
| Phase 4 (Orchestrator) | None break | New `test_orchestrator.py`. Scripts tests may need update if they test script internals |
| Phase 5 (Benchmarks) | `tests/benchmarks/test_ddr.py` | Tests refactored to use `DDRAdapter` + `BenchmarkRunner`. `MockStreamflow`/`MockKAN` continue to work |

**New mocks needed by refactor:**
- `MockRouter(RoutingModel)` — implements BMI Protocol for unit testing dmc in isolation
- `MockTopology` → `NetworkTopology` fixture from Sandbox data
- `MockBounds` → `PhysicalBounds` with test ranges

**Key: No existing test should break** — deprecated bridges in Phase 2-3 keep backward compatibility. Tests are updated incrementally, not all at once.

---

### Entry Point 7: gRPC (Phase 6 — Stubs Only)

**Proposed structure:**
```
src/ddr/serving/
├── __init__.py
├── proto/
│   └── routing.proto       # Message definitions
├── server.py               # DDRRoutingServicer (stubs → NotImplementedError)
└── client.py               # Example client
```

**BMI → gRPC mapping (1:1):**
```
gRPC Endpoint            → Python Method
────────────────────────────────────────
Initialize(request)      → router.initialize(topology, bounds) + router.coldstart(q_t0)
Update(request)          → router.update(timestep_input)
UpdateUntil(request)     → router.update_until(inflow, params)
Finalize(request)        → router.finalize()
GetState(request)        → router.get_state()
SetState(request)        → router.set_state(state)
```

**What gets implemented now:** Stubs with `raise NotImplementedError("Full gRPC support coming in future PR")`.
**What gets implemented later:** Proto message serialization, tensor encoding, connection pooling, model lifecycle management.
**Why stubs are valuable:** Forces the BMI interface design to be gRPC-compatible from day 1. If the Protocol can't cleanly serialize, we discover that now, not when wiring gRPC.

---

## Status

**This is an analysis/assessment only.** No implementation now — this plan documents what would be necessary for the refactor when the time comes.

---

## Verification Summary

| Level | What | When |
|---|---|---|
| **1. Unit** | Physics functions, value objects, BMI lifecycle ordering | Each phase |
| **2. Numerical regression** | `torch.allclose()` predictions + gradients against captured reference | Each phase |
| **3. Gradient flow** | `torch.autograd.gradcheck()` on physics functions; full-chain backward pass | Phase 1-2 |
| **4. Benchmark regression** | Full Sandbox benchmark: DDR median NSE/KGE match to 4 decimals | Phase 2-3 |
| **5. Mass balance** | `sum(routed) ~ sum(inflow)` within numerical precision | Phase 2-3 |
| **6. Integration** | Train 1 epoch, sequential inference continuity, leakance toggle, full BMI lifecycle | Phase 2-4 |
| **7. Pre-commit** | `uv run pre-commit run --all-files` — ruff, mypy, formatting | Every phase |
