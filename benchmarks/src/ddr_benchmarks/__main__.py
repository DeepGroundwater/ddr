"""Entry point for `python -m ddr_benchmarks`."""

import os

import ddr_benchmarks
from ddr._version import __version__
from ddr_benchmarks.benchmark import main

os.environ["BENCHMARKS_VERSION"] = ddr_benchmarks.__version__
print(f"Running DDR Benchmark v{ddr_benchmarks.__version__} (DDR {__version__})")
main()
