"""Entry point for running DDR benchmarks. Same pattern as scripts/test.py."""

import logging
import os
import time
from pathlib import Path

import ddr_benchmarks
import hydra
from ddr_benchmarks.benchmark import benchmark
from ddr_benchmarks.validation import validate_benchmark_config
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ddr import dmc, kan, streamflow
from ddr._version import __version__

log = logging.getLogger(__name__)

os.environ["BENCHMARKS_VERSION"] = ddr_benchmarks.__version__
print(f"Running DDR Benchmark v{ddr_benchmarks.__version__} (DDR {__version__})")


@hydra.main(version_base="1.3", config_path="../config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main function - adapted from scripts/test.py:main()."""
    cfg.params.save_path = Path(HydraConfig.get().run.dir)
    (cfg.params.save_path / "plots").mkdir(exist_ok=True)

    # Validate benchmark config (DDR + model-specific)
    benchmark_cfg = validate_benchmark_config(cfg)
    config = benchmark_cfg.ddr
    diffroute_cfg = benchmark_cfg.diffroute

    start_time = time.perf_counter()

    try:
        # Initialize DDR models (same as test.py)
        nn = kan(
            input_var_names=config.kan.input_var_names,
            learnable_parameters=config.kan.learnable_parameters,
            hidden_size=config.kan.hidden_size,
            num_hidden_layers=config.kan.num_hidden_layers,
            grid=config.kan.grid,
            k=config.kan.k,
            seed=config.seed,
            device=config.device,
        )
        routing_model = dmc(cfg=config, device=config.device)
        flow = streamflow(config)

        benchmark(
            cfg=config,
            flow=flow,
            routing_model=routing_model,
            nn=nn,
            diffroute_cfg=diffroute_cfg,
            summed_q_prime_path=benchmark_cfg.summed_q_prime,
        )

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")

    finally:
        log.info("Cleaning up...")
        total_time = time.perf_counter() - start_time
        log.info(f"Time Elapsed: {(total_time / 60):.6f} minutes")


if __name__ == "__main__":
    main()
