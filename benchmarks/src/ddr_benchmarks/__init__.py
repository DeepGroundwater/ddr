"""DDR Benchmarks - Tools for comparing DDR against other routing models."""

from ddr_benchmarks.benchmark import (
    benchmark,
    load_summed_q_prime,
    reorder_to_diffroute,
    reorder_to_topo,
    run_ddr,
    run_diffroute_benchmark,
)
from ddr_benchmarks.configs import (
    BenchmarkConfig,
    DiffRouteConfig,
    validate_benchmark_config,
)
from ddr_benchmarks.diffroute_adapter import (
    build_diffroute_inputs,
    create_param_df,
    load_rapid_params,
    zarr_group_to_networkx,
    zarr_to_networkx,
)

__version__ = "0.1.0"

__all__ = [
    # Adapter functions
    "build_diffroute_inputs",
    "create_param_df",
    "load_rapid_params",
    "zarr_group_to_networkx",
    "zarr_to_networkx",
    # Benchmark functions
    "benchmark",
    "load_summed_q_prime",
    "reorder_to_diffroute",
    "reorder_to_topo",
    "run_ddr",
    "run_diffroute_benchmark",
    # Config classes
    "BenchmarkConfig",
    "DiffRouteConfig",
    "validate_benchmark_config",
]
