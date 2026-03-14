"""DDR command-line interface.

Provides a single ``ddr`` entry point that dispatches to the individual
Hydra-based scripts (train, test, route, etc.).

Usage::

    ddr train --config-name=merit_training
    ddr test  --config-name=merit_testing
    ddr route --config-name=merit_routing
    ddr train-and-test --config-name=lynker_train_and_test
"""

import importlib
import os
import sys


def main() -> None:
    """Dispatch subcommands to their respective script modules."""
    subcommands: dict[str, str] = {
        "train": "scripts.train",
        "test": "scripts.test",
        "route": "scripts.router",
        "train-and-test": "scripts.train_and_test",
        "summed-q-prime": "scripts.summed_q_prime",
    }

    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        _print_usage(subcommands)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in subcommands:
        print(f"Unknown command: {cmd}")
        _print_usage(subcommands)
        sys.exit(1)

    # Remove the subcommand from argv so Hydra sees only its own flags
    sys.argv.pop(1)

    # Set version env var (same as the if __name__ == "__main__" blocks)
    from ddr._version import __version__

    os.environ.setdefault("DDR_VERSION", __version__)

    module = importlib.import_module(subcommands[cmd])
    module.main()


def _print_usage(subcommands: dict[str, str]) -> None:
    """Print usage information."""
    cmds = " | ".join(subcommands)
    print(f"Usage: ddr <{cmds}> [hydra overrides]")
    print()
    print("Commands:")
    print("  train           Train the KAN + MC routing model")
    print("  test            Evaluate a trained model")
    print("  route           Forward routing with a trained model")
    print("  train-and-test  Train then immediately evaluate")
    print("  summed-q-prime  Compute unrouted baseline (sum of lateral inflows)")
