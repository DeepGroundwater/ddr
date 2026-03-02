# Physics-Grounded GNN Node Processor for Muskingum-Cunge Routing

This document describes the encoder-processor-decoder architecture that augments
DDR's Muskingum-Cunge (MC) routing with a Graph Neural Network (GNN) node
processor.  The physics solve remains exact — the GNN evolves latent per-reach
embeddings **alongside** the MC timestep loop, using the four MC coefficient
terms as physics-derived message channels.  A decoder then maps the evolving
embeddings back to physical routing parameters (Manning's $n$, channel geometry)
so that parameters can **adapt dynamically over time** rather than remaining
static across the hydrograph.

## Motivation

In standard DDR, the KAN produces a single set of physical parameters per reach
that remain constant across all timesteps within a batch.  Manning's $n$ = 0.04
at timestep 1 is the same $n$ = 0.04 at timestep 2160.  But physical roughness
is not truly static — it varies with flow regime, vegetation state, ice cover,
and floodplain activation.  The node processor allows parameters to evolve in
response to the physics of the routing itself: high flows can dynamically reduce
roughness (floodplain smoothing), low flows can increase it (exposed bed
roughness), and upstream conditions can propagate parameter adjustments
downstream through the network.

## Architecture Overview

```
                          ┌─────────────────────────────────────────────────┐
                          │                ENCODER                         │
                          │                                                │
  Catchment Attributes    │   ┌───────────┐     ┌──────────┐              │
  [N, D_attr]  ──────────►│   │  KAN      │────►│  h^0     │              │
                          │   │  Encoder   │     │  [N,D_h] │              │
                          │   └───────────┘     └────┬─────┘              │
                          └──────────────────────────┼──────────────────────┘
                                                     │
                          ┌──────────────────────────┼──────────────────────┐
                          │         PROCESSOR        │    (T timesteps)     │
                          │                          ▼                      │
                          │   ┌──────────────────────────────────────────┐  │
                          │   │         for t = 1, 2, ..., T:           │  │
                          │   │                                         │  │
                          │   │  ┌──────────────┐                       │  │
                          │   │  │ ParamDecoder  │◄── h^{t-1}           │  │
                          │   │  │ (Linear+σ)   │                       │  │
                          │   │  └──────┬───────┘                       │  │
                          │   │         │                                │  │
                          │   │         ▼                                │  │
                          │   │  params^{t-1} = {n, q_s, w_t, z}       │  │
                          │   │         │                                │  │
                          │   │         ▼                                │  │
                          │   │  ┌──────────────────────────────────┐   │  │
                          │   │  │  MC PHYSICS SOLVE (unchanged)    │   │  │
                          │   │  │                                  │   │  │
                          │   │  │  velocity ← Manning(Q_t, n, ..) │   │  │
                          │   │  │  C1..C4  ← Muskingum(v, dx, x)  │   │  │
                          │   │  │  b ← C2·(N@Q_t)+C3·Q_t+C4·q'   │   │  │
                          │   │  │  Q_{t+1} ← solve (I-C1·N)·Q=b  │   │  │
                          │   │  └──────────────┬───────────────────┘   │  │
                          │   │                 │                        │  │
                          │   │                 ▼                        │  │
                          │   │  ┌──────────────────────────────────┐   │  │
                          │   │  │  GNN NODE UPDATE                 │   │  │
                          │   │  │                                  │   │  │
                          │   │  │  physics channels:               │   │  │
                          │   │  │    φ1 = C1·(N@Q_{t+1})          │   │  │
                          │   │  │    φ2 = C2·(N@Q_t)              │   │  │
                          │   │  │    φ3 = C3·Q_t                  │   │  │
                          │   │  │    φ4 = C4·q'_t                 │   │  │
                          │   │  │    φ5 = Q_{t+1}                 │   │  │
                          │   │  │                                  │   │  │
                          │   │  │  upstream_h = N @ h^{t-1}       │   │  │
                          │   │  │                                  │   │  │
                          │   │  │  input = [h; upstream_h; φ1..φ5]│   │  │
                          │   │  │  h^t = LN(h^{t-1} + MLP(input))│   │  │
                          │   │  └──────────────┬───────────────────┘   │  │
                          │   │                 │                        │  │
                          │   │                 ▼                        │  │
                          │   │              h^t  [N, D_h]              │  │
                          │   └─────────────────────────────────────────┘  │
                          └────────────────────────────────────────────────┘
                                                     │
                          ┌──────────────────────────┼──────────────────────┐
                          │         DECODER          │                      │
                          │                          ▼                      │
                          │   Output = Q_{1..T}  (physical discharge,      │
                          │                       already produced by       │
                          │                       the MC solve above)       │
                          └─────────────────────────────────────────────────┘
```

## Comparison with Standard DDR

```
  STANDARD DDR (static parameters)       GNN-ENHANCED DDR (dynamic parameters)
  ════════════════════════════════        ═══════════════════════════════════════

  KAN(attrs) ──► {n, q_s, w_t, z}       KAN(attrs) ──► h^0 [N, D_h]
       │          (fixed for all t)            │
       │                                       ▼
       ▼                                 ┌─── ParamDecoder(h^{t-1}) ──► params^{t-1}
  for t = 1..T:                          │     │
    Q_{t+1} = MC_solve(params, Q_t)      │     ▼
                                         │  for t = 1..T:
  (same n at t=1 and t=2160)             │    Q_{t+1} = MC_solve(params^{t-1}, Q_t)
                                         │    h^t = NodeProcessor(h^{t-1}, MC terms)
                                         │    params^t = ParamDecoder(h^t)
                                         └─── (n can differ at t=1 vs t=2160)
```

## The GNN Update Step in Detail

At each routing timestep, after the MC sparse solve produces $Q_{t+1}$,
the node processor updates embeddings using five physics channels derived
directly from the MC row equation:

$$Q_{i,t+1} = \underbrace{C_{1,i}\,(\mathbf{N}\mathbf{Q}_{t+1})}_{\varphi_1}
            + \underbrace{C_{2,i}\,(\mathbf{N}\mathbf{Q}_t)}_{\varphi_2}
            + \underbrace{C_{3,i}\, Q_{i,t}}_{\varphi_3}
            + \underbrace{C_{4,i}\, q'_{i,t}}_{\varphi_4}$$

Each term captures a different physical process:

| Channel | MC Term | Physical Meaning |
|---------|---------|-----------------|
| $\varphi_1$ | $C_1 \cdot (\mathbf{N}\mathbf{Q}_{t+1})$ | Implicit upstream — how much future inflow propagates |
| $\varphi_2$ | $C_2 \cdot (\mathbf{N}\mathbf{Q}_t)$ | Explicit upstream — current upstream contribution |
| $\varphi_3$ | $C_3 \cdot Q_{i,t}$ | Memory/attenuation — how much current discharge persists |
| $\varphi_4$ | $C_4 \cdot q'_{i,t}$ | Lateral forcing — local runoff entering the reach |
| $\varphi_5$ | $Q_{i,t+1}$ | Total discharge — the solved physical state |

The five channels are sign-preserving log-transformed for scale invariance
(discharge spans 6+ orders of magnitude):

$$\tilde{\varphi} = \text{sign}(\varphi) \cdot \log(|\varphi| + \epsilon)$$

### Node embedding aggregation

Upstream embeddings are aggregated via the same sparse adjacency matrix
used by the MC routing itself:

$$\bar{\mathbf{h}}_i^t = \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^t = (\mathbf{N}\, \mathbf{h}^t)_i$$

This mirrors the upstream flow aggregation in the MC equation, but operates
on latent embeddings rather than discharge.

### Residual MLP update

The node state is updated via a residual MLP with LayerNorm:

$$\mathbf{h}_i^{t+1} = \text{LayerNorm}\!\left(\mathbf{h}_i^t + \text{MLP}\!\left([\mathbf{h}_i^t \;\|\; \bar{\mathbf{h}}_i^t \;\|\; \tilde{\varphi}_{1..5}]\right)\right)$$

The MLP is a two-layer network: $[2 D_h + 5 \to D_h \to D_h]$ with SiLU
activation.  The residual connection ensures gradients flow directly through
the embedding chain even when the MLP contribution is small.

## Parameter Decoder

The ParamDecoder is the extracted output layer of the KAN — a single
Linear + sigmoid mapping from the evolving embedding to physical parameters
in $[0, 1]$:

$$\hat{\theta}_i^t = \sigma\!\left(\mathbf{W}_{\text{dec}}\, \mathbf{h}_i^t + \mathbf{b}_{\text{dec}}\right) \in [0, 1]^{|\mathcal{P}|}$$

Where $\mathcal{P} = \{n, q_s, w_t, z\}$ is the set of learnable parameters.
The $[0, 1]$ outputs are then denormalized to physical bounds (same path as
classic DDR — see Section 2 of equations.md).

### Bias initialization

| Mode | Bias | $\sigma(\text{bias})$ | Purpose |
|------|------|-----------------------|---------|
| Default | 0.0 | 0.50 | Centered in parameter range |
| Gate | +1.0 | 0.73 | Start ON (parameter active) |
| Off | −2.0 | 0.12 | Start near minimum (parameter mostly inactive) |

## Why Physics Channels, Not Physics in the Loss

A key design choice distinguishes this approach from prior work (e.g.,
HydroGraphNet, Taghizadeh et al. 2025, which places physics only in the
loss function).  In DDR's node processor:

1. **The MC physics solve is exact and unchanged** — the sparse triangular
   solve produces physically consistent discharge at every timestep.

2. **Physics enters the GNN as input channels**, not as a soft constraint.
   The four MC coefficient terms are computed from the exact solve and fed
   directly to the node MLP.

3. **The GNN modifies parameters, not discharge** — the processor updates
   the embedding, the decoder produces new parameters, and those parameters
   feed back into the *next* timestep's exact MC solve.

This gives the best of both worlds: exact physics at every timestep (mass
conservation guaranteed by the sparse solve), plus adaptive parameters that
respond to flow conditions via learned dynamics.

```
  Physics-in-loss (HydroGraphNet):     Physics-in-forward (DDR + NodeProcessor):
  ─────────────────────────────────    ──────────────────────────────────────────

  GNN predicts Q directly              GNN predicts parameters (n, geometry)
  Physics only in loss function         Physics in EVERY forward timestep
  Mass conservation: soft penalty       Mass conservation: exact (sparse solve)
  Stability: pushforward trick          Stability: MC physics + clamps
  Domain: 2D SWE on meshes             Domain: 1D MC on dendritic networks
```

## Gradient Flow

The full gradient path through the encoder-processor-decoder:

```
loss
  │
  ▼
Q_{t+1}  ◄── sparse_solve(A, b)
  │              │           │
  │          A = I-C1·N    b = C2·I_t + C3·Q_t + C4·q'
  │              │                     │
  │          C1..C4 ◄── k = dx/c ◄── v ◄── Manning(n^{t-1}, geometry^{t-1})
  │                                              │
  │                                    ParamDecoder(h^{t-1})
  │                                              │
  │                                         h^{t-1}
  │                                              │
  ├──────────────────────────────────────► NodeProcessor.step(h^{t-2}, φ1..φ5)
  │                                              │
  │   (unrolled through T timesteps)        h^{t-2}
  │                                              │
  :                                              :
  │                                              │
  └──────────────────────────────────────► h^0 ◄── KAN(attributes)
                                                        │
                                                   KAN weights
```

Key gradient paths:
- **Through MC coefficients**: loss → $Q_{t+1}$ → $C_i$ → $k$ → $v$ → $n$ → ParamDecoder → $h$ → KAN
- **Through residual embedding chain**: loss → $Q_{t+1}$ → $h^{t-1}$ → $h^{t-2}$ → ... → $h^0$ → KAN
- **Through physics channels**: loss → NodeProcessor MLP → $\varphi_{1..5}$ (conditioning signal from MC solve)

## State Management

| State | Training (`carry_state=False`) | Inference (`carry_state=True`) |
|-------|-------------------------------|-------------------------------|
| $\mathbf{Q}_t$ | Reinit from hotstart each batch | Carry (detached) across batches |
| $\mathbf{H}_t$ (pool) | Reinit from equilibrium each batch | Carry (detached) across batches |
| $\mathbf{h}^t$ (embedding) | Reinit from KAN each batch | Carry (detached) across batches |

Detaching on carry prevents computation graphs from accumulating across batches
while preserving the tensor values for sequential inference continuity.

## Implementation Map

| Component | Class / Function | Location |
|-----------|-----------------|----------|
| KAN encoder (`output_embedding=True`) | `kan` | `nn/kan.py` |
| Node processor (GNN step) | `MCNodeProcessor` | `nn/node_processor.py` |
| Parameter decoder | `ParamDecoder` | `nn/node_processor.py` |
| GNN submodule ownership | `dmc` | `routing/torch_mc.py` |
| Physics channel computation | `route_timestep()` | `routing/mmc.py:821-839` |
| Embedding init / carry | `setup_inputs()` | `routing/mmc.py:412-426` |
| Parameter decode from embedding | `_update_params_from_embedding()` | `routing/mmc.py:512-523` |
| State detach on clear | `clear_batch_state()` | `routing/mmc.py:356-376` |
| Config (`use_node_processor`) | `Kan` | `validation/configs.py` |

## Configuration

Enable GNN-enhanced routing in the YAML config:

```yaml
kan:
  hidden_size: 128          # D_h — embedding dimension
  use_node_processor: true  # Enable encoder-processor-decoder mode
  use_graph_context: false  # Mutually exclusive with node_processor
  learnable_parameters:
    - n
    - q_spatial
    - top_width
    - side_slope

experiment:
  rho: 90                   # 90-day batches (shorter gradient chains)
  epochs: 20                # More passes to compensate for shorter rho
```

When `use_node_processor: true`:
- KAN operates in `output_embedding` mode: returns $\mathbf{h}^0 \in \mathbb{R}^{N \times D_h}$ instead of a parameter dict
- `MCNodeProcessor` and `ParamDecoder` are registered as `nn.Module` submodules of `dmc`, included in the optimizer
- Parameters are decoded from the evolving embedding at each timestep
- Checkpoints save and restore all three components (KAN + NodeProcessor + ParamDecoder)
