# Juniata single-catchment sample

DDR on one basin: the Juniata River at Newport, PA (USGS 01567000,
8,657 km², 213 MERIT reaches). Everything needed is in `data/` — no HPC,
S3, or external stores.

## Quickstart

    git clone <repo> && cd ddr
    uv sync --all-packages
    uv run python -m examples.juniata.train_and_test --bundle examples/juniata/data
    # then open examples/juniata/juniata_routing.ipynb

Training is ~30 optimizer steps (one random 90-day window per epoch on a
single gauge): it demonstrates learning and physically plausible parameter
fields, not a converged CONUS-grade model. Reference run (CPU, 30 epochs):
routed NSE 0.784 KGE 0.877, summed-q' baseline NSE 0.695 KGE 0.820.

## What's in the bundle

| File | Contents |
|---|---|
| `juniata_qprime.ic` | icechunk, `Qr(divide_id, time)` daily m³/s, 213 divides, 1980–2010 (dHBV2 UH retrospective) |
| `juniata_obs.ic` | icechunk, `streamflow(gage_id, time)` daily m³/s, USGS 01567000, 1980–2010 |
| `juniata_attributes.nc` | 10 KAN input attributes per COMID |
| `juniata_conus_adjacency.zarr` | binsparse COO subgraph + `length_m`, `slope`, `order` (compact 0..212 indexing) |
| `juniata_gages_adjacency.zarr` | single-gage COO group, same schema as the CONUS store |
| `juniata_gage.csv` | one-row gage metadata (gages_3000 schema) |

Normalization statistics are computed from these 213 catchments on first
run (basin-local z-scores) into `data/statistics/` (gitignored).

The bundle was written with the repo-locked icechunk (2.0.3) — install via
`uv sync`, not a standalone icechunk.

Regenerate (maintainer, needs the CONUS stores):
`uv run python examples/juniata/extract_bundle.py --out examples/juniata/data`

## Toward end-to-end gradients

`train_and_test.py` marks the seam: `q_prime = flow(routing_dataclass=batch)`.
A differentiable runoff model replacing `flow` must return an hourly
`(num_timesteps, num_divides)` float32 m³/s tensor with gradients attached —
nothing else changes.
