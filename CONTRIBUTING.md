# Contributing to DDR

## Development Setup

```bash
git clone https://github.com/DeepGroundwater/ddr.git
cd ddr
uv sync --all-packages
pre-commit install
```

## Code Style

- **Formatter/Linter:** ruff (line length 110)
- **Type checking:** mypy (strict mode)
- **Docstrings:** NumPy convention
- Pre-commit hooks enforce all of the above on every commit.

## Running Tests

```bash
uv run pytest                     # Unit tests
uv run pytest -m integration      # Integration tests (requires local data)
uv run pytest tests/routing/ -v   # Run specific test directory
```

## Interface Change Checklist

When modifying `src/ddr/` interfaces (constructor signatures, forward() return types, config fields), check these downstream consumers:

1. **`scripts/`** — `train.py`, `test.py`, `train_and_test.py`, `router.py` instantiate `kan()` and `dmc()`
2. **`examples/`** — Notebooks load configs and instantiate models
3. **`benchmarks/`** — `benchmarks/scripts/benchmark.py` and `benchmarks/src/ddr_benchmarks/benchmark.py` have their own model instantiation
4. **`config/`** — YAML files may reference renamed or removed fields

Quick check: `grep -r "kan(" examples/ benchmarks/ scripts/` to find all call sites.

## Pull Request Process

1. Create a feature branch from `master`
2. Make your changes with tests
3. Ensure CI passes: `uv run pytest && uv run ruff check . && uv run mypy src/`
4. Open a PR with a clear description of what and why

## Adding a New GeoDataset

1. Add enum value to `src/ddr/validation/enums.py` (`GeoDataset`)
2. Create dataset class in `src/ddr/geodatazoo/` extending `BaseGeoDataset`
3. Register in `GeoDataset.get_dataset_class()`
4. Add config template in `config/templates/`
5. Add tests in `tests/geodatazoo/`

## Project Structure

```
ddr/
├── src/ddr/          # Core library
├── engine/           # Data preparation (ddr-engine package)
├── benchmarks/       # Comparison framework (ddr-benchmarks package)
├── scripts/          # Training/testing entry points
├── config/           # Hydra YAML configs
│   └── templates/    # Portable config templates (version controlled)
├── examples/         # Example notebooks + trained weights
├── tests/            # Test suite
└── docs/             # Zensical documentation
```
