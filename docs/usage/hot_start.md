---
icon: lucide/flame
---

# Hot-Start Initialization

When DDR begins routing, it needs an initial discharge value at every node in the river network. This page explains the hot start mechanism that computes a physically reasonable initial state.

## The Problem

Each routing batch requires an initial discharge $Q_0(i)$ at every segment $i$ before the first timestep can be routed. A naive approach sets $Q_0(i) = Q'_0(i)$ — the local lateral inflow at time $t{=}0$. This is a poor estimate because downstream segments should carry the accumulated flow from all upstream tributaries, not just their own local contribution.

Consider a simple linear network with 5 reaches, each contributing 2 m$^3$/s of lateral inflow:

| Node | Naive $Q_0$ | Correct $Q_0$ |
|------|-------------|---------------|
| 0 (headwater) | 2 | 2 |
| 1 | 2 | 4 |
| 2 | 2 | 6 |
| 3 | 2 | 8 |
| 4 (outlet) | 2 | 10 |

The naive approach underestimates discharge at every non-headwater node, creating an artificial "ramp-up" period at the start of each training window.

## The Solution: Topological Accumulation

The hot-start computes accumulated discharge by solving:

$$
(\mathbf{I} - \mathbf{N}) \cdot \mathbf{Q}_0 = \mathbf{Q}'_0
$$

Where:

- $\mathbf{N}$ is the adjacency matrix (lower triangular, $N_{ij} = 1$ if flow goes from $j$ to $i$)
- $\mathbf{Q}'_0$ is the lateral inflow at $t{=}0$
- $\mathbf{Q}_0$ is the initial discharge we want

Expanding for a single node $i$:

$$
Q_0(i) - \sum_{j \in \text{upstream}(i)} Q_0(j) = Q'_0(i)
$$

$$
Q_0(i) = Q'_0(i) + \sum_{j \in \text{upstream}(i)} Q_0(j)
$$

Because nodes are indexed in topological order (headwaters first, outlets last) and $\mathbf{I} - \mathbf{N}$ is lower triangular, this system is solved efficiently via forward substitution using `triangular_sparse_solve` — the same sparse solver used for each routing timestep.

## When It Applies

| Scenario | Initialization |
|----------|---------------|
| **Training** (every batch) | Hot-start via topological accumulation |
| **Inference** (first batch) | Hot-start via topological accumulation |
| **Inference** (subsequent batches) | State carried from previous batch (`carry_state=True`) |

### Training

Each training batch samples a random time window. There is no state to carry between batches, so every batch uses the hot-start. Combined with the warmup period (`cfg.experiment.warmup`), this gives the model a realistic starting point to route from.

### Inference

The first batch uses the hot-start. All subsequent batches pass `carry_state=True`, which preserves the discharge state from the end of the previous batch — maintaining physical continuity across the full simulation period.

## Implementation

The hot-start is implemented in `compute_hotstart_discharge()` in `src/ddr/routing/mmc.py`:

```python
from ddr.routing.mmc import compute_hotstart_discharge

discharge_t0 = compute_hotstart_discharge(
    q_prime_t0=q_prime[0],   # lateral inflow at t=0
    mapper=mapper,           # PatternMapper from the network
    discharge_lb=discharge_lb,
    device=device,
)
```

The function:

1. Constructs the $(\mathbf{I} - \mathbf{N})$ matrix using the same `PatternMapper` and `fill_op` infrastructure as `route_timestep`
2. Solves the lower-triangular system via `triangular_sparse_solve`
3. Clamps the result to `discharge_lb` (physical lower bound)


### Differentiability

`triangular_sparse_solve` has a custom backward pass, so gradients flow through the hot-start initialization during training.

## Relationship to Summed Q'

The hot-start solves the same mathematical problem as [Summed Lateral Flow](summed_q_prime.md), but at runtime rather than as a preprocessing step. The sparse solve approach avoids needing to precompute and store accumulated flows in the dataset, and naturally handles arbitrary subnetworks extracted for gauge-based training.
