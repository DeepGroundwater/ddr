"""Tests for MCNodeProcessor and ParamDecoder (GNN-like MC mode).

Covers output shapes, value ranges, gradient flow, and integration with
the MuskingumCunge routing engine.
"""

import pytest
import torch

from ddr.nn.node_processor import MCNodeProcessor, ParamDecoder

# ─── fixtures ────────────────────────────────────────────────────────────────

N = 12  # number of reaches
D_H = 32  # embedding dimension (smaller than production for fast tests)
PARAMS = ["n", "q_spatial", "top_width", "side_slope"]


def _adjacency(n: int) -> torch.Tensor:
    """Return a small lower-triangular sparse CSR adjacency (chain topology)."""
    # Simple chain: reach i+1 receives from reach i
    indices = torch.arange(n - 1, dtype=torch.long)
    row_idx = indices + 1  # downstream
    col_idx = indices  # upstream
    # Build dense then convert so we don't need crow_indices by hand
    dense = torch.zeros(n, n)
    dense[row_idx, col_idx] = 1.0
    return dense.to_sparse_csr()


@pytest.fixture
def processor() -> MCNodeProcessor:
    return MCNodeProcessor(d_hidden=D_H)


@pytest.fixture
def decoder() -> ParamDecoder:
    return ParamDecoder(d_hidden=D_H, learnable_parameters=PARAMS)


@pytest.fixture
def adjacency() -> torch.Tensor:
    return _adjacency(N)


@pytest.fixture
def h(request) -> torch.Tensor:
    return torch.randn(N, D_H, requires_grad=True)


# ─── ParamDecoder tests ───────────────────────────────────────────────────────


def test_param_decoder_output_shape(decoder: ParamDecoder) -> None:
    h = torch.randn(N, D_H)
    out = decoder(h)
    assert isinstance(out, dict)
    assert set(out.keys()) == set(PARAMS)
    for key, val in out.items():
        assert val.shape == (N,), f"Parameter '{key}' has wrong shape: {val.shape}"


def test_param_decoder_range(decoder: ParamDecoder) -> None:
    h = torch.randn(N, D_H)
    out = decoder(h)
    for key, val in out.items():
        assert (val >= 0.0).all() and (val <= 1.0).all(), (
            f"Parameter '{key}' out of [0, 1]: min={val.min():.4f}, max={val.max():.4f}"
        )


def test_param_decoder_gradients_flow(decoder: ParamDecoder) -> None:
    h = torch.randn(N, D_H, requires_grad=True)
    out = decoder(h)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_param_decoder_gate_bias_init() -> None:
    """Gate parameters should start with bias +1.0 (sigmoid ~ 0.73)."""
    dec = ParamDecoder(d_hidden=D_H, learnable_parameters=PARAMS, gate_parameters=["n"])
    idx = PARAMS.index("n")
    assert abs(dec.linear.bias[idx].item() - 1.0) < 1e-6


def test_param_decoder_off_bias_init() -> None:
    """Off parameters should start with bias -2.0 (sigmoid ~ 0.12)."""
    dec = ParamDecoder(d_hidden=D_H, learnable_parameters=PARAMS, off_parameters=["q_spatial"])
    idx = PARAMS.index("q_spatial")
    assert abs(dec.linear.bias[idx].item() - (-2.0)) < 1e-6


# ─── MCNodeProcessor tests ────────────────────────────────────────────────────


def test_mc_node_processor_step_shape(processor: MCNodeProcessor, adjacency: torch.Tensor) -> None:
    h = torch.randn(N, D_H)
    q = torch.rand(N).clamp(min=1e-4)
    h_new = processor.step(
        h=h,
        c1_next_upstream=q,
        c2_prev_upstream=q,
        c3_self=q,
        c4_lateral=q,
        q_new=q,
        adjacency=adjacency,
    )
    assert h_new.shape == (N, D_H)


def test_mc_node_processor_output_finite(processor: MCNodeProcessor, adjacency: torch.Tensor) -> None:
    h = torch.randn(N, D_H)
    q = torch.rand(N).clamp(min=1e-4)
    h_new = processor.step(
        h=h, c1_next_upstream=q, c2_prev_upstream=q, c3_self=q, c4_lateral=q, q_new=q, adjacency=adjacency
    )
    assert torch.isfinite(h_new).all(), "MCNodeProcessor produced non-finite embeddings"


def test_mc_node_processor_all_four_channels_used(
    processor: MCNodeProcessor, adjacency: torch.Tensor
) -> None:
    """Gradients must be non-zero w.r.t. each of the four MC coefficient terms."""
    h = torch.randn(N, D_H)
    q = torch.rand(N).clamp(min=1e-4)

    c1 = q.detach().clone().requires_grad_(True)
    c2 = q.detach().clone().requires_grad_(True)
    c3 = q.detach().clone().requires_grad_(True)
    c4 = q.detach().clone().requires_grad_(True)

    h_new = processor.step(
        h=h, c1_next_upstream=c1, c2_prev_upstream=c2, c3_self=c3, c4_lateral=c4, q_new=q, adjacency=adjacency
    )
    h_new.sum().backward()

    assert c1.grad is not None and c1.grad.abs().sum() > 0, "No gradient through c1_next_upstream"
    assert c2.grad is not None and c2.grad.abs().sum() > 0, "No gradient through c2_prev_upstream"
    assert c3.grad is not None and c3.grad.abs().sum() > 0, "No gradient through c3_self"
    assert c4.grad is not None and c4.grad.abs().sum() > 0, "No gradient through c4_lateral"


def test_mc_node_processor_gradients_through_h(processor: MCNodeProcessor, adjacency: torch.Tensor) -> None:
    """Gradients must flow back through the embedding h."""
    h = torch.randn(N, D_H, requires_grad=True)
    q = torch.rand(N).clamp(min=1e-4)
    h_new = processor.step(
        h=h, c1_next_upstream=q, c2_prev_upstream=q, c3_self=q, c4_lateral=q, q_new=q, adjacency=adjacency
    )
    h_new.sum().backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    assert h.grad.abs().sum() > 0


def test_mc_node_processor_negative_inputs(processor: MCNodeProcessor, adjacency: torch.Tensor) -> None:
    """C3 can be negative — signed_log must handle this without NaN."""
    h = torch.randn(N, D_H)
    q_pos = torch.rand(N).clamp(min=1e-4)
    c3_neg = -torch.rand(N).clamp(min=1e-4)  # C3 negative (large k)
    h_new = processor.step(
        h=h,
        c1_next_upstream=q_pos,
        c2_prev_upstream=q_pos,
        c3_self=c3_neg,
        c4_lateral=q_pos,
        q_new=q_pos,
        adjacency=adjacency,
    )
    assert torch.isfinite(h_new).all()


# ─── Integration tests ────────────────────────────────────────────────────────


def test_dynamic_params_change_across_timesteps(adjacency: torch.Tensor) -> None:
    """Manning's n decoded from h^t should differ across timesteps (params evolve)."""
    proc = MCNodeProcessor(d_hidden=D_H)
    dec = ParamDecoder(d_hidden=D_H, learnable_parameters=PARAMS)

    h = torch.randn(N, D_H)
    q = torch.rand(N).clamp(min=1e-4)

    params_t0 = dec(h)
    # Run several processor steps
    for _ in range(5):
        h = proc.step(
            h=h, c1_next_upstream=q, c2_prev_upstream=q, c3_self=q, c4_lateral=q, q_new=q, adjacency=adjacency
        )
    params_t5 = dec(h)

    # Parameters should differ (embedding has evolved)
    assert not torch.allclose(params_t0["n"], params_t5["n"]), (
        "Manning's n did not change across timesteps — embedding is not evolving"
    )


def test_node_processor_with_routing() -> None:
    """Full forward pass with MCNodeProcessor produces finite discharge."""
    from unittest.mock import MagicMock

    from ddr.routing.mmc import MuskingumCunge

    # Minimal routing with node processor — use sandbox-like data
    proc = MCNodeProcessor(d_hidden=D_H)
    dec = ParamDecoder(d_hidden=D_H, learnable_parameters=["n", "q_spatial", "top_width", "side_slope"])

    # Build a tiny (3-reach) routing problem
    n_reaches = 3
    dense_adj = torch.zeros(n_reaches, n_reaches)
    dense_adj[1, 0] = 1.0  # reach 0 → reach 1
    dense_adj[2, 1] = 1.0  # reach 1 → reach 2
    adj = dense_adj.to_sparse_csr()

    # Mock routing_dataclass
    rd = MagicMock()
    rd.adjacency_matrix = adj
    rd.outflow_idx = None
    rd.gage_catchment = None
    rd.observations = None
    rd.attribute_names = ["n", "q_spatial", "top_width", "side_slope"]
    rd.length = torch.full((n_reaches,), 10000.0)
    rd.slope = torch.full((n_reaches,), 0.001)
    rd.x = torch.full((n_reaches,), 0.3)
    rd.top_width = torch.empty(0)  # empty → use decoded
    rd.side_slope = torch.empty(0)  # empty → use decoded
    rd.flow_scale = None
    rd.reservoir_mask = None

    # Mock config (no spec so attribute assignment works freely)
    cfg = MagicMock()
    cfg.params.parameter_ranges = {
        "n": [0.015, 0.25],
        "q_spatial": [0.0, 1.0],
        "top_width": [1.0, 5000.0],
        "side_slope": [0.5, 50.0],
    }
    cfg.params.log_space_parameters = ["top_width", "side_slope"]
    cfg.params.defaults = {"p_spatial": 21}
    cfg.params.attribute_minimums = {
        "velocity": 0.01,
        "depth": 0.01,
        "discharge": 1e-4,
        "bottom_width": 0.01,
        "slope": 1e-4,
    }
    cfg.params.use_reservoir = False

    engine = MuskingumCunge(cfg, device="cpu", node_processor=proc, param_decoder=dec)

    T = 10
    q_prime = torch.rand(T, n_reaches) * 10.0
    h0 = torch.randn(n_reaches, D_H)

    engine.setup_inputs(
        routing_dataclass=rd,
        streamflow=q_prime,
        spatial_parameters=None,
        carry_state=False,
        node_embeddings=h0,
    )
    output = engine.forward()

    assert output.shape == (n_reaches, T)
    assert torch.isfinite(output).all(), "Routing with MCNodeProcessor produced non-finite discharge"
    assert (output > 0).all(), "Discharge must be positive"
