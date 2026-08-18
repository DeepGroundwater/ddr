"""Generate juniata_routing.ipynb. Run: uv run --with nbformat python examples/juniata/make_notebook.py"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells: list = []


def md(s: str) -> None:
    """Append a markdown cell."""
    cells.append(nbf.v4.new_markdown_cell(s))


def code(s: str) -> None:
    """Append a code cell."""
    cells.append(nbf.v4.new_code_cell(s))


md("""# From dMC-Juniata to DDR: differentiable routing on one catchment

The Juniata River at Newport, PA (USGS 01567000; 8,657 km²; 213 MERIT reaches)
— the basin where differentiable Muskingum-Cunge routing started. This notebook
walks the full physics and training chain on a laptop-sized bundle.

**Contents** — 1. The basin · 2. Muskingum-Cunge physics · 3. The network solve
· 4. Why differentiable · 5. Train & evaluate · 6. The road to end-to-end.""")

# --- 1. The basin ---
code("""import sys
from pathlib import Path

# Notebook CWD is examples/juniata; add repo root so `examples` package is importable
_repo_root = Path.cwd().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.juniata.train_and_test import make_config
from ddr.validation.enums import GeoDataset  # imported for reference; cfg.geodataset is an instance

BUNDLE = Path("data")
cfg = make_config(bundle_dir=BUNDLE)
# cfg.geodataset is a GeoDataset enum instance; call .get_dataset_class on it
dataset = cfg.geodataset.get_dataset_class(cfg=cfg)
batch = dataset.collate_fn(list(dataset.gage_ids))
N = batch.adjacency_matrix.shape[0]
print(f"{N} reaches, gauge reach index {batch.outflow_idx[0][0]}")""")

code("""import rustworkx as rx
from rustworkx.visualization import mpl_draw

adj = batch.adjacency_matrix.to_dense().numpy()
g = rx.PyDiGraph()
g.add_nodes_from(range(N))
rows, cols = np.nonzero(adj)
g.add_edges_from_no_data([(int(c), int(r)) for r, c in zip(rows, cols)])
fig, ax = plt.subplots(figsize=(10, 7))
mpl_draw(g, ax=ax, node_size=12, arrow_size=4)
ax.set_title("Juniata reach network (edges point downstream)")""")

code("""obs = batch.observations.streamflow.values[0]
t = dataset.dates.daily_time_range
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(t[: len(obs)], obs, lw=0.5)
ax.set_ylabel("Q (m³/s)"); ax.set_title("Observed discharge at Newport")""")

md("""At 8,657 km², the Juniata sits squarely in the 5–10 k km² drainage-area band
where DDR's area-stratified skill analysis shows the largest gain over the
unrouted lateral-inflow baseline: +0.135 median NSE relative to summed Q′
(ddrs Table 2). Routing acts here because basin travel times at this scale
span one to several days — long enough to shift and attenuate the hydrograph
at daily resolution. Smaller basins respond within hours and are effectively
point sources; larger basins damp all signals and are less sensitive to routing
parameter choice. The Juniata is therefore a nearly ideal calibration target:
it is big enough for travel time to matter yet small enough (213 reaches) to
run end-to-end on a laptop.""")

# --- 2. Physics ---
md(r"""## 2. Muskingum-Cunge physics

Storage routing from continuity $\frac{dS}{dt} = I - Q$ with
$S = K[XI + (1-X)Q]$ gives the update
$Q_{t+1} = c_1 I_{t+1} + c_2 I_t + c_3 Q_t + c_4 q'$ with

$$c_1 = \frac{\Delta t - 2KX}{D},\; c_2 = \frac{\Delta t + 2KX}{D},\;
c_3 = \frac{2K(1-X) - \Delta t}{D},\; c_4 = \frac{2\Delta t}{D},\;
D = 2K(1-X) + \Delta t$$

$c_1 + c_2 + c_3 = 1$ **exactly** — mass conservation holds for any $(K, X)$.
$K = L/c$ is the reach travel time; everything hinges on celerity $c$ and $X$.""")

code("""# Verify the mass identity on the actual implementation
from ddr.routing.mmc import MuskingumCunge
mc = MuskingumCunge(cfg, device="cpu")
c1, c2, c3, c4 = mc.calculate_muskingum_coefficients(
    length=torch.tensor([5000.0]), celerity=torch.tensor([1.2]), x=torch.tensor([0.4]))
print(c1 + c2 + c3)  # tensor([1.])""")

md(r"""### Trapezoid-exact celerity

Kinematic celerity is $c = dQ/dA$. For the trapezoid DDR builds
(Leopold & Maddock: $T = p\,d^{\,q}$), $c = v\,\beta$ with
$\beta = \frac{5}{3} - \frac{4}{3}\frac{A\sqrt{1+z^2}}{T\,P}$ — the classic
$5/3$ is the wide-rectangular limit and runs 22–27% high on real channels.""")

code("""def beta(b, y, z):
    T = b + 2 * z * y
    A = (b + T) * y / 2
    P = b + 2 * y * np.sqrt(1 + z**2)
    return 5 / 3 - (4 / 3) * A * np.sqrt(1 + z**2) / (T * P)

by = np.logspace(-2, 3, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for z in [0.0, 1.0, 2.0]:
    ax.semilogx(by, beta(by, 1.0, z), label=f"z={z}")
ax.axhline(5 / 3, ls="--", c="k", lw=0.7); ax.axhline(4 / 3, ls=":", c="gray", lw=0.7)
ax.set_xlabel("b / y"); ax.set_ylabel("β"); ax.legend()
ax.set_title("β is non-monotone in b/y and NOT bounded below by 4/3")""")

md(r"""### Cunge X: matching numerical to physical diffusion

The scheme's numerical diffusion is $D_{num} = cL(0.5 - X)$; the channel's
physical diffusivity is $D_{phys} = Q/(2TS)$. Setting them equal:
$X = \mathrm{clamp}\!\left(0.5\left(1 - \frac{Q}{T\,S\,c\,L}\right), 0, 0.5\right)$.
The legacy constant $X = 0.3$ traded diffusion accuracy for a wide stability
window $2X \le C_r \le 2(1-X)$ — DDR now computes $X$ per reach per timestep.""")

# --- 3. Network solve ---
md(r"""## 3. The network solve

Per timestep DDR solves $(I - c_1 N)\,Q_{t+1} = c_2 N Q_t + c_3 Q_t + c_4 q'$.
$N$ is the downstream adjacency; topological ordering makes $(I - c_1 N)$
**lower triangular**, so the solve is a single forward substitution.""")

code("""fig, ax = plt.subplots(figsize=(5, 5))
ax.spy(np.eye(N) + adj, markersize=1)
ax.set_title("(I − c₁N) sparsity — lower triangular")""")

# --- 4. Why differentiable ---
md("""## 4. Why differentiable

The KAN maps catchment attributes → {n, q_spatial, p_spatial} ∈ [0,1] →
physical bounds. The loss differentiates through the solve, the coefficients,
Cunge X, and β back into the KAN weights — one autograd chain.""")

code("""from ddr import dmc, kan, streamflow

nn = kan(input_var_names=cfg.kan.input_var_names,
         learnable_parameters=cfg.kan.learnable_parameters,
         hidden_size=cfg.kan.hidden_size, num_hidden_layers=cfg.kan.num_hidden_layers,
         grid=cfg.kan.grid, k=cfg.kan.k, seed=cfg.seed, device="cpu")
routing = dmc(cfg=cfg, device="cpu")
flow = streamflow(cfg)

dataset.dates.calculate_time_period()
b = dataset.collate_fn(list(dataset.gage_ids))
q_prime = flow(routing_dataclass=b)          # <-- future runoff model plugs in HERE
params = nn(inputs=b.normalized_spatial_attributes)
out = routing(routing_dataclass=b, spatial_parameters=params, streamflow=q_prime)
loss = out["runoff"].mean()
loss.backward()
g = [p.grad.abs().mean().item() for p in nn.parameters() if p.grad is not None]
print(f"{len(g)} KAN tensors received gradients; mean |grad| {np.mean(g):.2e}")""")

# --- 5. Train & evaluate ---
code("""from examples.juniata.train_and_test import summed_qprime_baseline, test, train

cfg = make_config(bundle_dir=BUNDLE, epochs=1)   # fast mode; raise to 30 for the README numbers
ckpt = train(cfg)
result = test(cfg, ckpt)
baseline = summed_qprime_baseline(cfg)
print(f"routed  NSE {result.attrs['nse']:.3f}  KGE {result.attrs['kge']:.3f}")
print(f"summed  NSE {baseline.attrs['nse']:.3f}  KGE {baseline.attrs['kge']:.3f}")""")

code("""fig, ax = plt.subplots(figsize=(12, 4))
sl = slice(0, 730)
ax.plot(result.time[sl], result.observations[0, sl], "k", lw=0.8, label="observed")
ax.plot(result.time[sl], result.predictions[0, sl], "C0", lw=0.8, label="DDR routed")
ax.plot(baseline.time[sl], baseline.predictions[0, sl], "C1", lw=0.6, ls="--", label="summed q'")
ax.legend(); ax.set_ylabel("Q (m³/s)")""")

# --- 6. Road to end-to-end ---
md("""## 6. The road to end-to-end

Q' entered this notebook through `flow(routing_dataclass=...)` — a reader of
precomputed runoff. The contract for replacing it with a **differentiable
runoff model** is exactly: return an hourly `(num_timesteps, num_divides)`
float32 tensor in m³/s that carries `requires_grad`. Then `loss.backward()`
reaches the runoff model's parameters through the routing physics — the full
end-to-end gradient chain. That toy model is the next project.""")

nb["cells"] = cells
nbf.write(nb, "examples/juniata/juniata_routing.ipynb")
print("wrote examples/juniata/juniata_routing.ipynb")
