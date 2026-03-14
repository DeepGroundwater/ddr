# DDR — Distributed Differentiable Routing

AI-agent context file. Committed to version control so every coding assistant
(Claude, Copilot, Cursor, etc.) gets the same codebase orientation.

---

## Architecture

DDR couples a **Kolmogorov-Arnold Network (KAN)** with **differentiable
Muskingum-Cunge (MC) routing** to learn spatially varying river-routing
parameters end-to-end via PyTorch autograd.

1. **KAN** ingests catchment attributes and predicts three spatial parameters
   per reach: Manning's *n*, *q_spatial*, and *p_spatial*.
2. **Leopold & Maddock power law** converts depth to channel geometry:
   `top_width = p_spatial * depth^(q_spatial + 1e-6)`.
3. **MC routing** solves the linearized Saint-Venant equations on a trapezoidal
   channel cross-section using a sparse-matrix solve (CSR format).
4. Gradients flow from the loss (KGE, NSE, etc.) back through the routing
   physics into the KAN weights.

---

## Module Map

| Path | Role |
|---|---|
| `src/ddr/routing/mmc.py` | Core `MuskingumCunge` engine — sparse matrix solve, trapezoid velocity, parameter denormalization |
| `src/ddr/routing/torch_mc.py` | PyTorch `nn.Module` wrapper (`dmc` class) |
| `src/ddr/nn/kan.py` | KAN neural network for spatial parameter prediction |
| `src/ddr/io/readers.py` | `StreamflowReader` for loading lateral inflows |
| `src/ddr/io/functions.py` | Utility functions (downsampling, etc.) |
| `src/ddr/validation/configs.py` | Pydantic config models (`Config`, `DataSources`, `Params`, `Kan`, `ExperimentConfig`) |
| `src/ddr/validation/enums.py` | `GeoDataset` and `Mode` enums |
| `src/ddr/validation/metrics.py` | Evaluation metrics |
| `src/ddr/validation/plots.py` | Plotting utilities |
| `src/ddr/geodatazoo/` | Dataset abstraction layer — `BaseGeoDataset`, `Merit`, `LynkerHydrofabric` |
| `src/ddr/scripts_utils.py` | Shared helpers used by the scripts below |

## Public API (`src/ddr/__init__.py`)

```python
from .routing.torch_mc import dmc        # Differentiable routing model
from .nn import kan                       # KAN neural network
from .io.readers import StreamflowReader as streamflow  # Data reader
from .io import functions as ddr_functions              # Utilities
from . import validation                  # Config, Metrics, plotting
```

---

## Config Flow

```
Hydra YAML (config/) → OmegaConf DictConfig → validate_config() → Pydantic Config
```

- Hydra parses YAML from the `config/` directory.
- `validate_config()` in `src/ddr/validation/configs.py` converts the
  `DictConfig` to a typed Pydantic model.
- Config supports `${oc.env:VAR_NAME,default}` interpolation for portable
  paths across machines.

---

## Scripts

| Script | Purpose |
|---|---|
| `scripts/train.py` | Training loop (`python scripts/train.py --config-name=<config>`) |
| `scripts/test.py` | Evaluation |
| `scripts/train_and_test.py` | Combined train + test |
| `scripts/router.py` | Forward routing with a trained model |
| `scripts/summed_q_prime.py` | Baseline — unrouted sum of lateral inflows |

---

## Downstream Call-Site Checklist

When modifying `src/ddr/` interfaces (constructor signatures, `forward()`
return types, config fields), **always check and update these downstream
consumers**:

1. **`examples/`** — Example notebooks that instantiate `kan()`, `dmc()`, and
   load configs.
2. **`benchmarks/scripts/benchmark.py`** and
   **`benchmarks/src/ddr_benchmarks/benchmark.py`** — Own `kan()`/`dmc()`
   instantiation and evaluation loops that must stay in sync with the core
   scripts.
3. **`scripts/`** — All training/testing scripts.
4. **`config/`** — YAML files may reference field names that changed.

Quick grep to find all `kan()` constructor call sites:

```bash
grep -r "kan(" examples/ benchmarks/ scripts/
```

---

## Testing

```bash
uv run pytest                    # Unit tests (no data dependencies)
uv run pytest -m integration     # Integration tests (requires HPC data)
```

- Unit tests live in `tests/`.
- Integration tests are marked with `@pytest.mark.integration` and deselected
  by default (`addopts = "-m 'not integration'"`).

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

All config lives in `pyproject.toml`. Pre-commit hooks are defined in
`.pre-commit-config.yaml`.

---

## Datasets (GeoDataset enum)

Two supported geodatasets (see `src/ddr/validation/enums.py`):

| Enum value | Dataset | Attributes | Geometry |
|---|---|---|---|
| `merit` | MERIT-Hydro global river network | `.nc` | `.shp` |
| `lynker_hydrofabric` | Lynker Hydrofabric v2.2 (CONUS) | icechunk store | `.gpkg` |

Key difference: MERIT uses `log10_uparea`; Lynker uses `log_uparea` for
upstream area.

---

## Workspace (monorepo)

Three packages managed by `uv`:

| Package | Directory | Description |
|---|---|---|
| `ddr` | `.` (root) | Core routing library |
| `ddr-engine` | `engine/` | Geospatial data preparation |
| `ddr-benchmarks` | `benchmarks/` | Comparison framework (vs DiffRoute, etc.) |

Install everything:

```bash
uv sync --all-packages
```

---

## Key Conventions

- Python **>=3.11, <3.14**.
- PyTorch with CUDA 13.0 index by default (configurable in `pyproject.toml`).
- Sparse CSR tensors are used for the routing matrix solve — expect beta
  warnings from PyTorch (already suppressed in pytest config).
- `__init__.py` files use `F401` ignore so re-exports don't trigger unused-import
  lint errors.
