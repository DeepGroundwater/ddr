"""Tests for level pool reservoir routing physics.

Tests the standalone _level_pool_outflow() and _compute_equilibrium_pool_elevation()
functions, plus the single-solve RHS override integration with MuskingumCunge routing.
"""

from pathlib import Path

import pytest
import torch

from ddr.routing.mmc import (
    _compute_equilibrium_pool_elevation,
    _level_pool_outflow,
)

# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #

DISCHARGE_LB = torch.tensor(1e-4)

# Standard reservoir: elevation=100m, depth=10m
# orifice_elevation=90, weir_elevation=97.5 (75% of full pool)
ELEVATION = 100.0
DEPTH = 10.0


@pytest.fixture()
def reservoir_params() -> dict[str, torch.Tensor]:
    """Return a standard set of reservoir parameters for testing."""
    return {
        "weir_elevation": torch.tensor([97.5]),
        "orifice_elevation": torch.tensor([90.0]),
        "weir_coeff": torch.tensor([0.4]),
        "weir_length": torch.tensor([10.0]),
        "orifice_coeff": torch.tensor([0.6]),
        "orifice_area": torch.tensor([5.0]),
        "lake_area_m2": torch.tensor([1e6]),
        "initial_pool_elevation": torch.tensor([95.0]),
    }


# --------------------------------------------------------------------------- #
# Outflow function tests                                                       #
# --------------------------------------------------------------------------- #


def test_outflow_zero_below_orifice(reservoir_params: dict[str, torch.Tensor]) -> None:
    """When pool is below orifice elevation, outflow should be at discharge_lb."""
    pool_elev = torch.tensor([89.0])  # below orifice at 90
    outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k not in ("lake_area_m2", "initial_pool_elevation")},
    )
    assert torch.allclose(outflow, DISCHARGE_LB, atol=1e-6)


def test_outflow_orifice_only_between_elevations(reservoir_params: dict[str, torch.Tensor]) -> None:
    """When pool is between orifice and weir, only orifice flow should occur."""
    pool_elev = torch.tensor([95.0])  # above orifice (90), below weir (97.5)
    outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k not in ("lake_area_m2", "initial_pool_elevation")},
    )
    # Manual orifice calc: Q = 0.6 * 5.0 * sqrt(2 * 9.81 * 5.0) = 3.0 * sqrt(98.1)
    h_orifice = 5.0  # 95 - 90
    expected_orifice = 0.6 * 5.0 * (2.0 * 9.81 * h_orifice) ** 0.5
    # Weir head is zero (95 < 97.5) so no weir contribution
    assert outflow.item() > 0
    assert abs(outflow.item() - expected_orifice) < 0.1  # small eps from sqrt(x + 1e-8)


def test_outflow_weir_plus_orifice_above_weir(reservoir_params: dict[str, torch.Tensor]) -> None:
    """When pool is above weir, both weir and orifice flow should occur."""
    pool_elev = torch.tensor([99.0])  # above weir at 97.5
    outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k not in ("lake_area_m2", "initial_pool_elevation")},
    )
    # Weir: 0.4 * 10.0 * (1.5)^1.5 ~ 7.35
    # Orifice: 0.6 * 5.0 * sqrt(2*9.81*9.0) ~ 39.8
    # Total should be significant
    assert outflow.item() > 40.0  # both components active


def test_outflow_gradient_flows(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Pool elevation gradient should flow through outflow computation."""
    pool_elev = torch.tensor([95.0], requires_grad=True)
    outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k not in ("lake_area_m2", "initial_pool_elevation")},
    )
    outflow.sum().backward()
    assert pool_elev.grad is not None
    assert pool_elev.grad.item() != 0.0


# --------------------------------------------------------------------------- #
# Forward Euler pool update tests (outflow + dH inline)                        #
# --------------------------------------------------------------------------- #


def _outflow_params(reservoir_params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract outflow-only params (exclude lake_area_m2 and initial_pool_elevation)."""
    return {k: v for k, v in reservoir_params.items() if k not in ("lake_area_m2", "initial_pool_elevation")}


def test_step_mass_balance(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Verify A_s * (H_new - H_old) = dt * (inflow - outflow)."""
    dt = 3600.0
    inflow = torch.tensor([50.0])
    pool_elev = torch.tensor([95.0])
    lake_area = reservoir_params["lake_area_m2"]

    outflow = _level_pool_outflow(
        pool_elevation=pool_elev, discharge_lb=DISCHARGE_LB, **_outflow_params(reservoir_params)
    )
    dh = dt * (inflow - outflow) / (lake_area + 1e-8)
    new_elev = pool_elev + dh

    # Mass balance: A_s * dH = dt * (I - O)
    lhs = lake_area * (new_elev - pool_elev)
    rhs = dt * (inflow - outflow)
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_step_pool_rises_when_inflow_exceeds_outflow(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Pool should rise when inflow exceeds outflow."""
    pool_elev = torch.tensor([95.0])
    inflow = torch.tensor([1000.0])
    lake_area = reservoir_params["lake_area_m2"]

    outflow = _level_pool_outflow(
        pool_elevation=pool_elev, discharge_lb=DISCHARGE_LB, **_outflow_params(reservoir_params)
    )
    dh = 3600.0 * (inflow - outflow) / (lake_area + 1e-8)
    new_elev = pool_elev + dh
    assert new_elev.item() > pool_elev.item()


def test_step_pool_drops_when_outflow_exceeds_inflow(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Pool should drop when outflow exceeds inflow."""
    pool_elev = torch.tensor([99.0])  # high pool -> large outflow
    inflow = torch.tensor([0.001])  # tiny inflow
    lake_area = reservoir_params["lake_area_m2"]

    outflow = _level_pool_outflow(
        pool_elevation=pool_elev, discharge_lb=DISCHARGE_LB, **_outflow_params(reservoir_params)
    )
    dh = 3600.0 * (inflow - outflow) / (lake_area + 1e-8)
    new_elev = pool_elev + dh
    assert new_elev.item() < pool_elev.item()


# --------------------------------------------------------------------------- #
# Vectorized / multi-reach tests                                              #
# --------------------------------------------------------------------------- #


def test_reservoir_mask_selectivity() -> None:
    """Non-reservoir reaches should be unaffected by level pool outflow override."""
    inflow = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    reservoir_mask = torch.tensor([False, True, False, True, False])

    pool_elev = torch.tensor([95.0, 95.0])
    outflow_res = _level_pool_outflow(
        pool_elevation=pool_elev,
        weir_elevation=torch.tensor([97.5, 97.5]),
        orifice_elevation=torch.tensor([90.0, 90.0]),
        weir_coeff=torch.tensor([0.4, 0.4]),
        weir_length=torch.tensor([10.0, 10.0]),
        orifice_coeff=torch.tensor([0.6, 0.6]),
        orifice_area=torch.tensor([5.0, 5.0]),
        discharge_lb=DISCHARGE_LB,
    )

    # Apply to full vector using clone + masked assignment (same as route_timestep)
    q_t1 = inflow.clone()
    q_t1[reservoir_mask] = outflow_res

    # Non-reservoir reaches unchanged
    assert q_t1[0].item() == 10.0
    assert q_t1[2].item() == 30.0
    assert q_t1[4].item() == 50.0

    # Reservoir reaches were modified
    assert q_t1[1].item() != 20.0
    assert q_t1[3].item() != 40.0


def test_reservoir_attenuates_peak() -> None:
    """Peak discharge through reservoir should be attenuated vs pass-through."""
    # Simulate a flood pulse: rising then falling inflow
    timesteps = 100
    inflow_series = torch.cat(
        [
            torch.linspace(1.0, 100.0, timesteps // 2),
            torch.linspace(100.0, 1.0, timesteps // 2),
        ]
    )

    h = torch.tensor([95.0])
    lake_area = torch.tensor([1e6])
    outflow_kw = {
        "weir_elevation": torch.tensor([97.5]),
        "orifice_elevation": torch.tensor([90.0]),
        "weir_coeff": torch.tensor([0.4]),
        "weir_length": torch.tensor([10.0]),
        "orifice_coeff": torch.tensor([0.6]),
        "orifice_area": torch.tensor([5.0]),
        "discharge_lb": DISCHARGE_LB,
    }

    outflows = []
    for t in range(timesteps):
        outflow = _level_pool_outflow(pool_elevation=h, **outflow_kw)
        dh = 3600.0 * (inflow_series[t : t + 1] - outflow) / (lake_area + 1e-8)
        h = h + dh
        outflows.append(outflow.item())

    peak_inflow = inflow_series.max().item()
    peak_outflow = max(outflows)
    # Reservoir should attenuate the peak
    assert peak_outflow < peak_inflow


def test_pool_elevation_carry_state() -> None:
    """Pool elevation should change after a forward Euler step with large inflow."""
    pool_elev_init = torch.tensor([95.0])
    lake_area = torch.tensor([1e6])
    inflow = torch.tensor([500.0])

    outflow = _level_pool_outflow(
        pool_elevation=pool_elev_init,
        weir_elevation=torch.tensor([97.5]),
        orifice_elevation=torch.tensor([90.0]),
        weir_coeff=torch.tensor([0.4]),
        weir_length=torch.tensor([10.0]),
        orifice_coeff=torch.tensor([0.6]),
        orifice_area=torch.tensor([5.0]),
        discharge_lb=DISCHARGE_LB,
    )
    dh = 3600.0 * (inflow - outflow) / (lake_area + 1e-8)
    modified_elev = pool_elev_init + dh

    # Modified elevation should differ from initial
    assert not torch.allclose(modified_elev, pool_elev_init, atol=1e-6)

    # Pool should rise (inflow=500 >> outflow at h=5m above orifice)
    # carry_state semantics (preserve vs reset) are tested via
    # MuskingumCunge._init_pool_elevation_state, not here.
    assert modified_elev.item() > pool_elev_init.item()


def test_config_validation_reservoir() -> None:
    """use_reservoir=True without reservoir_params should raise ValueError."""
    from unittest.mock import patch

    from ddr.validation.configs import Config

    base_cfg = {
        "name": "test",
        "geodataset": "merit",
        "mode": "training",
        "data_sources": {
            "geospatial_fabric_gpkg": "/tmp/test.gpkg",
            "conus_adjacency": "/tmp/test.zarr",
            "forcings": "/tmp/forcings",
        },
        "params": {
            "use_reservoir": True,
        },
        "kan": {
            "input_var_names": ["aridity"],
            "learnable_parameters": ["q_spatial"],
        },
        "cuda_lstm": {
            "input_var_names": ["aridity"],
            "learnable_parameters": ["n"],
        },
    }

    with pytest.raises(ValueError, match="use_reservoir=True requires data_sources.reservoir_params"):
        # Patch check_path to avoid filesystem checks
        with patch("ddr.validation.configs.check_path", side_effect=lambda v: Path(v)):
            Config(**base_cfg)


# --------------------------------------------------------------------------- #
# Single-solve reservoir integration tests                                     #
# --------------------------------------------------------------------------- #


def test_single_solve_reservoir_outflow() -> None:
    """Verify RHS override: sparse solve produces outflow at reservoir row.

    3-reach linear network: 0 -> 1(reservoir) -> 2.
    When c_1_[1] = 0, the matrix A has identity at row 1, so the forward
    substitution yields q_t1[1] = b[1] = outflow.  Downstream reach 2 then
    uses this correct outflow (not MC-solved value) in the same solve.
    """
    from ddr.routing.utils import PatternMapper, triangular_sparse_solve

    # 3-reach linear: 0 -> 1 -> 2 (topologically ordered)
    network = torch.zeros(3, 3)
    network[1, 0] = 1.0  # reach 1 receives from reach 0
    network[2, 1] = 1.0  # reach 2 receives from reach 1

    # Build pattern mapper via fill_op (I + diag(c_1_) @ N)
    def fill_op(data_vector: torch.Tensor) -> torch.Tensor:
        n = network.shape[0]
        eye = torch.eye(n)
        vec_diag = torch.diag(data_vector)
        return (eye + vec_diag @ network).to_sparse_csr()

    mapper = PatternMapper(fill_op, 3)

    # MC coefficients (arbitrary but physically plausible)
    c_1 = torch.tensor([0.3, 0.25, 0.35])
    c_2 = torch.tensor([0.4, 0.45, 0.3])
    c_3 = torch.tensor([0.3, 0.3, 0.35])
    c_4 = torch.tensor([0.6, 0.6, 0.6])

    discharge_t = torch.tensor([10.0, 20.0, 30.0])
    q_prime = torch.tensor([5.0, 3.0, 4.0])

    # Upstream inflow: N @ Q_t
    i_t = network @ discharge_t  # [0, 10, 20]
    b = c_2 * i_t + c_3 * discharge_t + c_4 * q_prime

    # Reservoir at reach 1: compute outflow from pool state
    pool_elev = torch.tensor([95.0])
    res_outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        weir_elevation=torch.tensor([97.5]),
        orifice_elevation=torch.tensor([90.0]),
        weir_coeff=torch.tensor([0.4]),
        weir_length=torch.tensor([10.0]),
        orifice_coeff=torch.tensor([0.6]),
        orifice_area=torch.tensor([5.0]),
        discharge_lb=DISCHARGE_LB,
    )

    # Override b[1] with reservoir outflow
    b_override = b.clone()
    b_override[1] = res_outflow.item()

    # Zero c_1_[1] -> identity row at reservoir
    c_1_ = c_1 * -1
    c_1_[0] = 1.0
    c_1_[1] = 0.0  # reservoir row -> identity
    A_values = mapper.map(c_1_)

    solution = triangular_sparse_solve(
        A_values,
        mapper.crow_indices,
        mapper.col_indices,
        b_override,
        True,
        False,
        "cpu",
    )
    q_t1 = torch.clamp(solution, min=DISCHARGE_LB)

    # q_t1[1] should equal the level-pool outflow (identity row)
    assert torch.allclose(q_t1[1], res_outflow.squeeze(), atol=1e-4), (
        f"q_t1[1]={q_t1[1].item():.4f} != outflow={res_outflow.item():.4f}"
    )

    # Downstream reach 2 should use the reservoir outflow, not MC-solved value.
    # For reach 2: A[2,2]*q_t1[2] + A[2,1]*q_t1[1] = b[2]
    # => q_t1[2] = b[2] - c_1_[2]*q_t1[1]  (forward sub with unit diagonal)
    # c_1_[2] = -c_1[2] = -0.35, so A[2,1] = c_1_[2]*N[2,1] = -0.35*1 = -0.35
    expected_q2 = b_override[2] - (c_1_[2] * 1.0 * q_t1[1])  # forward sub
    assert torch.allclose(q_t1[2], torch.clamp(expected_q2, min=DISCHARGE_LB), atol=1e-4), (
        f"q_t1[2]={q_t1[2].item():.4f} != expected={expected_q2.item():.4f}"
    )


def test_equilibrium_pool_initialization() -> None:
    """Equilibrium pool elevation: orifice outflow should match inflow.

    Max orifice outflow at weir (h=7.5m): 0.6*5*sqrt(2*9.81*7.5) ≈ 36.4 m³/s.
    So inflows below ~36 m³/s produce sub-weir equilibria; above → capped.
    """
    inflow = torch.tensor([10.0, 30.0, 200.0])
    orifice_elev = torch.tensor([90.0, 90.0, 90.0])
    orifice_coeff = torch.tensor([0.6, 0.6, 0.6])
    orifice_area = torch.tensor([5.0, 5.0, 5.0])
    weir_elev = torch.tensor([97.5, 97.5, 97.5])

    pool_eq = _compute_equilibrium_pool_elevation(
        inflow=inflow,
        orifice_elevation=orifice_elev,
        orifice_coeff=orifice_coeff,
        orifice_area=orifice_area,
        weir_elevation=weir_elev,
    )

    # Verify: outflow at equilibrium pool ≈ inflow (for sub-weir cases)
    outflow = _level_pool_outflow(
        pool_elevation=pool_eq,
        weir_elevation=weir_elev,
        orifice_elevation=orifice_elev,
        weir_coeff=torch.tensor([0.4, 0.4, 0.4]),
        weir_length=torch.tensor([10.0, 10.0, 10.0]),
        orifice_coeff=orifice_coeff,
        orifice_area=orifice_area,
        discharge_lb=DISCHARGE_LB,
    )

    # First two have equilibrium below weir -> orifice outflow ≈ inflow
    assert torch.allclose(outflow[0], inflow[0], rtol=0.02), (
        f"outflow[0]={outflow[0].item():.2f} != inflow[0]={inflow[0].item():.2f}"
    )
    assert torch.allclose(outflow[1], inflow[1], rtol=0.02), (
        f"outflow[1]={outflow[1].item():.2f} != inflow[1]={inflow[1].item():.2f}"
    )

    # Third case: inflow=200 exceeds orifice capacity at weir → capped at weir
    assert pool_eq[2] == weir_elev[2], "Pool should be capped at weir elevation"

    # All equilibrium elevations should be >= orifice elevation
    assert (pool_eq >= orifice_elev).all()


def test_pool_elevation_update_gradient() -> None:
    """Pool elevation forward Euler update should propagate gradients."""
    pool_elev = torch.tensor([95.0], requires_grad=True)
    orifice_elev = torch.tensor([90.0])
    weir_elev = torch.tensor([97.5])
    lake_area = torch.tensor([1e6])

    # Compute outflow from current pool state
    outflow = _level_pool_outflow(
        pool_elevation=pool_elev,
        weir_elevation=weir_elev,
        orifice_elevation=orifice_elev,
        weir_coeff=torch.tensor([0.4]),
        weir_length=torch.tensor([10.0]),
        orifice_coeff=torch.tensor([0.6]),
        orifice_area=torch.tensor([5.0]),
        discharge_lb=DISCHARGE_LB,
    )

    # Forward Euler pool update (same as route_timestep)
    inflow = torch.tensor([50.0])
    dh = 3600.0 * (inflow - outflow) / (lake_area + 1e-8)
    new_pool_elev = pool_elev + dh

    # Compute outflow at new pool elevation
    outflow_t1 = _level_pool_outflow(
        pool_elevation=new_pool_elev,
        weir_elevation=weir_elev,
        orifice_elevation=orifice_elev,
        weir_coeff=torch.tensor([0.4]),
        weir_length=torch.tensor([10.0]),
        orifice_coeff=torch.tensor([0.6]),
        orifice_area=torch.tensor([5.0]),
        discharge_lb=DISCHARGE_LB,
    )

    # Gradient should flow: outflow_t1 -> new_pool_elev -> pool_elev
    outflow_t1.sum().backward()
    assert pool_elev.grad is not None, "Gradient did not flow through pool update"
    assert pool_elev.grad.item() != 0.0, "Gradient is zero"

    # Pool should have risen (inflow > outflow at h=5m above orifice)
    assert new_pool_elev.item() > pool_elev.item(), "Pool should rise when inflow > outflow"


# --------------------------------------------------------------------------- #
# NaN safety / stability tests                                                 #
# --------------------------------------------------------------------------- #


def test_size_filtering_excludes_small_reservoirs() -> None:
    """Reservoirs below min_reservoir_area_km2 should be excluded from the mask."""
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    # Create a mock Merit instance with just the fields _build_reservoir_tensors needs
    mock_self = MagicMock()
    mock_self.cfg.params.min_reservoir_area_km2 = 10.0  # 10 km² threshold

    # Reservoir CSV: 3 COMIDs, one small (5 km² = 5e6 m²), two large (15/20 km²)
    mock_self.reservoir_df = pd.DataFrame(
        {
            "lake_area_m2": [5e6, 15e6, 20e6],  # 5, 15, 20 km²
            "weir_elevation": [97.5, 97.5, 97.5],
            "orifice_elevation": [90.0, 90.0, 90.0],
            "weir_coeff": [0.4, 0.4, 0.4],
            "weir_length": [10.0, 10.0, 10.0],
            "orifice_coeff": [0.1, 0.2, 0.3],
            "orifice_area": [5.0, 10.0, 15.0],
            "initial_pool_elevation": [95.0, 95.0, 95.0],
        },
        index=pd.Index([100, 200, 300], name="COMID"),
    )

    # Batch with all 3 COMIDs + one non-reservoir COMID
    catchment_ids = np.array([100, 200, 300, 400])

    # Import and call the actual method, binding mock_self as self
    from ddr.geodatazoo.merit import Merit

    result = Merit._build_reservoir_tensors(mock_self, catchment_ids)

    # COMID 100 (5 km²) should be excluded, 200 (15 km²) and 300 (20 km²) included
    assert result["reservoir_mask"][0].item() is False, "COMID 100 (5 km²) should be filtered"
    assert result["reservoir_mask"][1].item() is True, "COMID 200 (15 km²) should be retained"
    assert result["reservoir_mask"][2].item() is True, "COMID 300 (20 km²) should be retained"
    assert result["reservoir_mask"][3].item() is False, "COMID 400 (not in CSV) should be False"

    # Verify per-dam attributes flow through for retained reservoirs
    assert result["orifice_coeff"][1].item() == pytest.approx(0.2)
    assert result["orifice_coeff"][2].item() == pytest.approx(0.3)


def test_size_filtering_zero_threshold_keeps_all() -> None:
    """With min_reservoir_area_km2=0, all reservoirs should be retained."""
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    mock_self = MagicMock()
    mock_self.cfg.params.min_reservoir_area_km2 = 0.0

    mock_self.reservoir_df = pd.DataFrame(
        {
            "lake_area_m2": [1e4, 5e6],  # 0.01 km² and 5 km²
            "weir_elevation": [97.5, 97.5],
            "orifice_elevation": [90.0, 90.0],
            "weir_coeff": [0.4, 0.4],
            "weir_length": [10.0, 10.0],
            "orifice_coeff": [0.1, 0.1],
            "orifice_area": [5.0, 5.0],
            "initial_pool_elevation": [95.0, 95.0],
        },
        index=pd.Index([100, 200], name="COMID"),
    )

    catchment_ids = np.array([100, 200])

    from ddr.geodatazoo.merit import Merit

    result = Merit._build_reservoir_tensors(mock_self, catchment_ids)

    assert result["reservoir_mask"][0].item() is True
    assert result["reservoir_mask"][1].item() is True


def test_config_min_reservoir_area_default() -> None:
    """Verify min_reservoir_area_km2 default is 10.0."""
    from ddr.validation.configs import Params

    p = Params()
    assert p.min_reservoir_area_km2 == 10.0


def test_no_nan_small_reservoir_large_inflow() -> None:
    """Forward Euler must not produce NaN even for tiny reservoirs with huge inflow.

    Small lake_area_m2 violates the explicit Euler stability criterion
    (dt * dQ/dH / A > 2).  Without a pool elevation clamp, pool oscillates
    and blows up to inf, then inf - inf = NaN.  The clamp in route_timestep
    prevents this.  This test simulates the same loop standalone.
    """
    # Tiny reservoir: 1000 m² area, 1m weir length (worst-case stability)
    orifice_elev = torch.tensor([90.0])
    weir_elev = torch.tensor([97.5])
    pool = torch.tensor([95.0], requires_grad=True)
    lake_area = torch.tensor([1_000.0])  # very small
    outflow_kw = {
        "weir_elevation": weir_elev,
        "orifice_elevation": orifice_elev,
        "weir_coeff": torch.tensor([0.4]),
        "weir_length": torch.tensor([1.0]),  # minimum
        "orifice_coeff": torch.tensor([0.6]),
        "orifice_area": torch.tensor([5.0]),
        "discharge_lb": DISCHARGE_LB,
    }
    pool_min = orifice_elev
    pool_max = weir_elev + (weir_elev - orifice_elev)

    # Run 500 hourly timesteps with large inflow (forward Euler + clamp)
    h = pool
    for _ in range(500):
        outflow = _level_pool_outflow(pool_elevation=h, **outflow_kw)
        dh = 3600.0 * (torch.tensor([500.0]) - outflow) / (lake_area + 1e-8)
        h = h + dh
        h = torch.maximum(h, pool_min)
        h = torch.minimum(h, pool_max)

    # No NaN in forward pass
    assert not torch.isnan(h).any(), f"Pool elevation is NaN: {h}"
    assert not torch.isinf(h).any(), f"Pool elevation is inf: {h}"

    # No NaN in backward pass
    h.sum().backward()
    assert pool.grad is not None, "No gradient"
    assert not torch.isnan(pool.grad).any(), f"Gradient is NaN: {pool.grad}"
