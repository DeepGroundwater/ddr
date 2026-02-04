"""Pytest configuration for benchmarks tests."""

import pytest

pytest.importorskip("ddr_benchmarks")
pytest.importorskip("ddr_engine")

# Load fixtures from MERIT engine tests (sandbox fixtures)
pytest_plugins = ["tests.engine.merit.conftest"]
