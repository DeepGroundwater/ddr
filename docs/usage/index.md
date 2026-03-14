
# Usage

DDR supports three operating modes, each with a corresponding CLI command and script:

| Mode | CLI Command | Script | Description |
|------|-------------|--------|-------------|
| **Training** | `ddr train` | `scripts/train.py` | Learn routing parameters from observed streamflow |
| **Testing** | `ddr test` | `scripts/test.py` | Evaluate a trained model on a held-out period |
| **Routing** | `ddr route` | `scripts/router.py` | Forward inference on new domains or time periods |

Additionally, two utility commands are available:

| Command | Script | Description |
|---------|--------|-------------|
| `ddr train-and-test` | `scripts/train_and_test.py` | Train then immediately evaluate |
| `ddr summed-q-prime` | `scripts/summed_q_prime.py` | Compute the unrouted baseline |

## Workflow

A typical DDR workflow follows these steps:

1. **Prepare data** — Build adjacency matrices with the [Engine](../engine/index.md)
2. **Copy a config template** — Start from `config/templates/` and customize paths
3. **[Train](train.md)** — Learn routing parameters from observed streamflow
4. **[Test](test.md)** — Evaluate on a held-out time period
5. **[Route](routing.md)** — Run inference on new domains or time periods

## Config Templates

Pre-built config templates are available in `config/templates/`:

| Template | Dataset | Mode |
|----------|---------|------|
| `merit_training.yaml` | MERIT Hydro | Training |
| `merit_routing.yaml` | MERIT Hydro | Routing |
| `lynker_training.yaml` | Lynker Hydrofabric v2.2 | Training |
| `lynker_routing.yaml` | Lynker Hydrofabric v2.2 | Routing |

All templates use `${oc.env:DDR_DATA_DIR,./data}` for portable paths. Set the `DDR_DATA_DIR` environment variable or edit paths directly.

## Additional Resources

- [Hot-Start Initialization](hot_start.md) — How DDR computes initial discharge states
- [Summed Lateral Flow](summed_q_prime.md) — The unrouted baseline for comparison
- [Examples](examples.md) — Pre-trained notebooks for visualizing results
