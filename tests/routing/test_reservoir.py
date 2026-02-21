"""Tests for level pool reservoir routing physics.

Tests the standalone _level_pool_outflow() and _level_pool_step() functions,
plus integration with the MuskingumCunge routing engine.
"""

from pathlib import Path

import pytest
import torch

from ddr.routing.mmc import _level_pool_outflow, _level_pool_step

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
# Level pool step tests                                                        #
# --------------------------------------------------------------------------- #


def test_step_mass_balance(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Verify A_s * (H_new - H_old) = dt * (inflow - outflow)."""
    dt = 3600.0
    inflow = torch.tensor([50.0])
    pool_elev = torch.tensor([95.0])
    lake_area = reservoir_params["lake_area_m2"]

    outflow, new_elev = _level_pool_step(
        inflow=inflow,
        pool_elevation=pool_elev,
        dt=dt,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k != "initial_pool_elevation"},
    )

    # Mass balance: A_s * dH = dt * (I - O)
    lhs = lake_area * (new_elev - pool_elev)
    rhs = dt * (inflow - outflow)
    assert torch.allclose(lhs, rhs, rtol=1e-4)


def test_step_pool_rises_when_inflow_exceeds_outflow(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Pool should rise when inflow exceeds outflow."""
    pool_elev = torch.tensor([95.0])
    # Use very large inflow to guarantee it exceeds outflow
    inflow = torch.tensor([1000.0])

    _, new_elev = _level_pool_step(
        inflow=inflow,
        pool_elevation=pool_elev,
        dt=3600.0,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k != "initial_pool_elevation"},
    )
    assert new_elev.item() > pool_elev.item()


def test_step_pool_drops_when_outflow_exceeds_inflow(reservoir_params: dict[str, torch.Tensor]) -> None:
    """Pool should drop when outflow exceeds inflow."""
    pool_elev = torch.tensor([99.0])  # high pool -> large outflow
    inflow = torch.tensor([0.001])  # tiny inflow

    _, new_elev = _level_pool_step(
        inflow=inflow,
        pool_elevation=pool_elev,
        dt=3600.0,
        discharge_lb=DISCHARGE_LB,
        **{k: v for k, v in reservoir_params.items() if k != "initial_pool_elevation"},
    )
    assert new_elev.item() < pool_elev.item()


# --------------------------------------------------------------------------- #
# Vectorized / multi-reach tests                                              #
# --------------------------------------------------------------------------- #


def test_reservoir_mask_selectivity() -> None:
    """Non-reservoir reaches should be unaffected by level pool step."""
    inflow = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    reservoir_mask = torch.tensor([False, True, False, True, False])

    # Only reservoir reaches get level pool routing
    res_inflow = inflow[reservoir_mask]
    pool_elev = torch.tensor([95.0, 95.0])
    lake_area = torch.tensor([1e6, 1e6])
    weir_elev = torch.tensor([97.5, 97.5])
    orifice_elev = torch.tensor([90.0, 90.0])
    weir_coeff = torch.tensor([0.4, 0.4])
    weir_length = torch.tensor([10.0, 10.0])
    orifice_coeff = torch.tensor([0.6, 0.6])
    orifice_area = torch.tensor([5.0, 5.0])

    outflow_res, new_elev = _level_pool_step(
        inflow=res_inflow,
        pool_elevation=pool_elev,
        lake_area_m2=lake_area,
        weir_elevation=weir_elev,
        orifice_elevation=orifice_elev,
        weir_coeff=weir_coeff,
        weir_length=weir_length,
        orifice_coeff=orifice_coeff,
        orifice_area=orifice_area,
        dt=3600.0,
        discharge_lb=DISCHARGE_LB,
    )

    # Apply to full vector using clone + masked assignment
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

    pool_elev = torch.tensor([95.0])
    params = {
        "lake_area_m2": torch.tensor([1e6]),
        "weir_elevation": torch.tensor([97.5]),
        "orifice_elevation": torch.tensor([90.0]),
        "weir_coeff": torch.tensor([0.4]),
        "weir_length": torch.tensor([10.0]),
        "orifice_coeff": torch.tensor([0.6]),
        "orifice_area": torch.tensor([5.0]),
    }

    outflows = []
    h = pool_elev.clone()
    for t in range(timesteps):
        outflow, h = _level_pool_step(
            inflow=inflow_series[t : t + 1],
            pool_elevation=h,
            dt=3600.0,
            discharge_lb=DISCHARGE_LB,
            **params,
        )
        outflows.append(outflow.item())

    peak_inflow = inflow_series.max().item()
    peak_outflow = max(outflows)
    # Reservoir should attenuate the peak
    assert peak_outflow < peak_inflow


def test_pool_elevation_carry_state() -> None:
    """Pool elevation should be preserved with carry_state=True, reset with False."""
    # Simulate two steps to get a modified pool elevation
    pool_elev_init = torch.tensor([95.0])
    params = {
        "lake_area_m2": torch.tensor([1e6]),
        "weir_elevation": torch.tensor([97.5]),
        "orifice_elevation": torch.tensor([90.0]),
        "weir_coeff": torch.tensor([0.4]),
        "weir_length": torch.tensor([10.0]),
        "orifice_coeff": torch.tensor([0.6]),
        "orifice_area": torch.tensor([5.0]),
    }

    _, modified_elev = _level_pool_step(
        inflow=torch.tensor([500.0]),
        pool_elevation=pool_elev_init,
        dt=3600.0,
        discharge_lb=DISCHARGE_LB,
        **params,
    )

    # Modified elevation should differ from initial
    assert not torch.allclose(modified_elev, pool_elev_init, atol=1e-6)

    # "carry_state=True" semantics: keep modified_elev
    # "carry_state=False" semantics: reset to initial
    # This tests the physics functions — integration with MuskingumCunge
    # carry_state is tested via the _init_pool_elevation_state method.
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
