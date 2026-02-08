"""Tests for benchmark validation configs — DiffRouteConfig and BenchmarkConfig."""

import pytest

pytest.importorskip("ddr_benchmarks")

from ddr_benchmarks.validation import BenchmarkConfig, DiffRouteConfig
from pydantic import ValidationError


class TestDiffRouteConfig:
    """Tests for DiffRouteConfig defaults and validation."""

    def test_defaults(self) -> None:
        cfg = DiffRouteConfig()
        assert cfg.enabled is True
        assert cfg.irf_fn == "muskingum"
        assert cfg.max_delay == 100
        assert abs(cfg.dt - 0.0416667) < 1e-6
        assert cfg.k is None
        assert cfg.x == 0.3

    def test_custom_values(self) -> None:
        cfg = DiffRouteConfig(
            enabled=False,
            irf_fn="nash_cascade",
            max_delay=200,
            dt=0.5,
            k=0.2,
            x=0.1,
        )
        assert cfg.enabled is False
        assert cfg.irf_fn == "nash_cascade"
        assert cfg.max_delay == 200
        assert cfg.k == 0.2

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DiffRouteConfig(nonexistent_field=42)


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig construction."""

    @pytest.fixture
    def _minimal_ddr_config(self, tmp_path):
        """Minimal kwargs to construct a DDR Config."""
        from ddr.validation import Config

        return Config(
            name="test",
            device="cpu",
            seed=42,
            mode="testing",
            geodataset="merit",
            params={"tau": 3, "save_path": str(tmp_path)},
            experiment={
                "start_time": "2000/01/01",
                "end_time": "2000/02/01",
                "batch_size": 1,
                "epochs": 1,
                "warmup": 0,
                "learning_rate": {1: 0.001},
            },
            data_sources={
                "streamflow": "/fake",
                "observations": "/fake",
                "conus_adjacency": "/fake",
            },
            kan={
                "input_var_names": ["slope"],
                "learnable_parameters": ["n"],
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "grid": 5,
                "k": 3,
            },
        )

    def test_construction(self, _minimal_ddr_config) -> None:
        bc = BenchmarkConfig(ddr=_minimal_ddr_config)
        assert bc.diffroute.enabled is True
        assert bc.summed_q_prime is None

    def test_extra_field_rejected(self, _minimal_ddr_config) -> None:
        with pytest.raises(ValidationError):
            BenchmarkConfig(ddr=_minimal_ddr_config, unknown_field="oops")

    def test_summed_q_prime_optional(self, _minimal_ddr_config) -> None:
        bc = BenchmarkConfig(ddr=_minimal_ddr_config, summed_q_prime="/some/path")
        assert bc.summed_q_prime == "/some/path"
