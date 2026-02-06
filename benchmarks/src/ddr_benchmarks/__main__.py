"""Entry point for `python -m ddr_benchmarks`."""

import os

from ddr._version import __version__ as ddr_version
from ddr_benchmarks import __version__
from ddr_benchmarks.benchmark import main

os.environ["BENCHMARKS_VERSION"] = __version__
print(f"Running DDR Benchmark v{__version__} (DDR {ddr_version})")
main()
