"""Benchmark configuration validation."""

from .benchmark import BenchmarkConfig, validate_benchmark_config
from .diffroute import DiffRouteConfig

__all__ = [
    "BenchmarkConfig",
    "DiffRouteConfig",
    "validate_benchmark_config",
]
