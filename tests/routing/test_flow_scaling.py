"""Acceptance and chaos tests for flow_scale integration in routing."""

import pytest
import torch

from ddr.io.readers import build_flow_scale_tensor
from ddr.routing.mmc import MuskingumCunge

from .test_utils import (
    create_mock_config,
    create_mock_routing_dataclass,
    create_mock_spatial_parameters,
    create_mock_streamflow,
)


class TestFlowScaleBackwardCompatible:
    """Acceptance: flow_scale=None leaves q_prime unmodified."""

    def test_flow_scale_none_backward_compatible(self):
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hf = create_mock_routing_dataclass(num_reaches=10)
        streamflow = create_mock_streamflow(24, 10)
        params = create_mock_spatial_parameters(10)

        assert hf.flow_scale is None
        mc.setup_inputs(hf, streamflow, params)
        # q_prime should equal the original streamflow (just moved to device)
        assert torch.allclose(mc.q_prime, streamflow)


class TestFlowScaleReducesQPrime:
    """Acceptance: flow_scale reduces q_prime at the scaled segment."""

    def test_flow_scale_reduces_q_prime_at_segment(self):
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hf = create_mock_routing_dataclass(num_reaches=10)
        streamflow = create_mock_streamflow(24, 10)
        params = create_mock_spatial_parameters(10)

        # Scale segment 2 to 50%
        flow_scale = torch.ones(10, dtype=torch.float32)
        flow_scale[2] = 0.5
        hf.flow_scale = flow_scale

        mc.setup_inputs(hf, streamflow, params)

        # Segment 2 should be halved
        assert mc.q_prime is not None
        expected_seg2 = streamflow[:, 2] * 0.5
        assert torch.allclose(mc.q_prime[:, 2], expected_seg2)

        # Other segments should be untouched
        for i in [0, 1, 3, 4, 5, 6, 7, 8, 9]:
            assert torch.allclose(mc.q_prime[:, i], streamflow[:, i])


class TestFlowScaleChaos:
    """Chaos tests: edge cases that shouldn't crash or produce NaN/Inf."""

    def test_near_zero_fraction(self):
        """Extreme scaling (fraction ~0.005) doesn't produce NaN/Inf."""
        cfg = create_mock_config()
        mc = MuskingumCunge(cfg, device="cpu")
        hf = create_mock_routing_dataclass(num_reaches=5)
        streamflow = create_mock_streamflow(12, 5)
        params = create_mock_spatial_parameters(5)

        flow_scale = torch.ones(5, dtype=torch.float32)
        flow_scale[1] = 0.005
        hf.flow_scale = flow_scale

        mc.setup_inputs(hf, streamflow, params)
        assert mc.q_prime is not None
        assert not torch.isnan(mc.q_prime).any()
        assert not torch.isinf(mc.q_prime).any()
        assert torch.allclose(mc.q_prime[:, 1], streamflow[:, 1] * 0.005)

    def test_gage_not_in_dict_skipped(self):
        """Unknown STAID in batch doesn't crash, returns all-ones."""

        def _make_gage_dict(staids, drains, comid_drains, unit_areas):
            return {
                "STAID": staids,
                "DRAIN_SQKM": drains,
                "COMID_DRAIN_SQKM": comid_drains,
                "COMID_UNITAREA_SQKM": unit_areas,
            }

        gage_dict = _make_gage_dict(
            staids=["01000000"],
            drains=[90.0],
            comid_drains=[100.0],
            unit_areas=[20.0],
        )
        result = build_flow_scale_tensor(
            batch=["99999999"],
            gage_dict=gage_dict,
            gage_compressed_indices=[0],
            num_segments=5,
        )
        assert torch.all(result == 1.0)

    def test_two_catchments_contributing_to_gage(self):
        """Gage at a confluence: outflow_idx has multiple segment indices but
        flow_scale only modifies the gage's own catchment segment, leaving
        upstream contributors at 1.0."""

        def _make_gage_dict(staids, drains, comid_drains, unit_areas):
            return {
                "STAID": staids,
                "DRAIN_SQKM": drains,
                "COMID_DRAIN_SQKM": comid_drains,
                "COMID_UNITAREA_SQKM": unit_areas,
            }

        gage_dict = _make_gage_dict(
            staids=["01000000"],
            drains=[90.0],
            comid_drains=[100.0],
            unit_areas=[20.0],
        )
        # gage_compressed_indices points to the gage's own segment (index 3)
        # outflow_idx might include [1, 2, 3] (upstream + gage), but flow_scale
        # only touches the gage's segment
        result = build_flow_scale_tensor(
            batch=["01000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[3],
            num_segments=5,
        )
        # Gage segment (3) should be scaled
        assert result[3].item() == pytest.approx(0.5)
        # Upstream segments untouched
        for i in [0, 1, 2, 4]:
            assert result[i].item() == 1.0

    def test_two_gages_same_segment(self):
        """Two gages map to the same compressed segment index. Second gage's
        factor overwrites the first. Test documents this behavior."""

        def _make_gage_dict(staids, drains, comid_drains, unit_areas):
            return {
                "STAID": staids,
                "DRAIN_SQKM": drains,
                "COMID_DRAIN_SQKM": comid_drains,
                "COMID_UNITAREA_SQKM": unit_areas,
            }

        gage_dict = _make_gage_dict(
            staids=["01000000", "02000000"],
            drains=[90.0, 95.0],
            comid_drains=[100.0, 100.0],
            unit_areas=[20.0, 20.0],
        )
        # Both gages map to segment 2
        result = build_flow_scale_tensor(
            batch=["01000000", "02000000"],
            gage_dict=gage_dict,
            gage_compressed_indices=[2, 2],
            num_segments=5,
        )
        # Second gage (02000000): diff=-5, fraction=(20-5)/20=0.75
        assert result[2].item() == pytest.approx(0.75)
