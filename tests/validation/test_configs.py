"""Tests for ddr.validation.configs."""

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import ValidationError

from ddr.validation.configs import Config, Params, validate_config


def _minimal_config_dict(**overrides):
    """Build a minimal valid config dict."""
    base = {
        "name": "test_run",
        "data_sources": {
            "geospatial_fabric_gpkg": "/tmp/fake.gpkg",
            "conus_adjacency": "/tmp/fake_adj",
            "forcings": "mock://forcings",
        },
        "geodataset": "merit",
        "mode": "training",
        "params": {},
        "kan": {
            "input_var_names": ["slope", "length"],
            "learnable_parameters": ["q_spatial", "n"],
        },
        "cuda_lstm": {
            "input_var_names": ["slope", "length"],
            "learnable_parameters": [],
        },
    }
    base.update(overrides)
    return base


class TestConfigCreation:
    """Test Config construction."""

    def test_valid_config_creation(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict()
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)

        config = Config(**d)
        assert config.name == "test_run"

    def test_missing_name_raises(self, tmp_path) -> None:
        d = _minimal_config_dict()
        del d["name"]
        with pytest.raises(ValidationError):
            Config(**d)

    def test_extra_fields_rejected(self, tmp_path) -> None:
        d = _minimal_config_dict()
        d["nonexistent_field"] = "bad"
        with pytest.raises(ValidationError):
            Config(**d)


class TestConfigDevice:
    """Test device validation."""

    def test_device_cpu_string(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict(device="cpu")
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)

        config = Config(**d)
        assert config.device == "cpu"

    def test_device_negative_int_raises(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict(device=-1)
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)

        with pytest.raises(ValidationError, match="non-negative"):
            Config(**d)


class TestConfigCheckpoint:
    """Test checkpoint validation."""

    def test_checkpoint_none_valid(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict()
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)
        d["experiment"] = {"checkpoint": None}

        config = Config(**d)
        assert config.experiment.checkpoint is None

    def test_checkpoint_missing_path_raises(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict()
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)
        d["experiment"] = {"checkpoint": "/nonexistent/path/model.pt"}

        with pytest.raises(ValidationError, match="does not exist"):
            Config(**d)


class TestParamsDefaults:
    """Test Params default values."""

    def test_attribute_minimums_defaults(self) -> None:
        p = Params()
        expected = {
            "discharge": 0.0001,
            "slope": 0.0001,
            "velocity": 0.01,
            "depth": 0.01,
            "bottom_width": 0.01,
        }
        assert p.attribute_minimums == expected

    def test_parameter_ranges_defaults(self) -> None:
        p = Params()
        assert "n" in p.parameter_ranges
        assert "q_spatial" in p.parameter_ranges
        assert "top_width" in p.parameter_ranges
        assert "side_slope" in p.parameter_ranges

    def test_log_space_parameters_default(self) -> None:
        p = Params()
        assert p.log_space_parameters == ["top_width", "side_slope", "d_gw"]


class TestSetSeed:
    """Test seed determinism."""

    def test_set_seed_determinism(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict(seed=42, device="cpu")
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)

        # Use validate_config indirectly via _set_seed by constructing Config
        from ddr.validation.configs import _set_seed

        config = Config(**d)
        _set_seed(config)
        a = torch.rand(5)
        _set_seed(config)
        b = torch.rand(5)

        assert torch.equal(a, b)


class TestExperimentConfigAreaThreshold:
    """Test max_area_diff_sqkm config field."""

    def test_valid_threshold(self, tmp_path) -> None:
        """Positive float accepted."""
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict()
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)
        d["experiment"] = {"max_area_diff_sqkm": 50.0}

        config = Config(**d)
        assert config.experiment.max_area_diff_sqkm == 50.0

    def test_zero_threshold_valid(self, tmp_path) -> None:
        """Zero means exact match required."""
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict()
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)
        d["experiment"] = {"max_area_diff_sqkm": 0.0}

        config = Config(**d)
        assert config.experiment.max_area_diff_sqkm == 0.0


class TestValidateConfig:
    """Test validate_config from DictConfig."""

    def test_validate_config_from_dictconfig(self, tmp_path) -> None:
        gpkg = tmp_path / "test.gpkg"
        gpkg.touch()
        adj = tmp_path / "adj"
        adj.mkdir()

        d = _minimal_config_dict(device="cpu")
        d["data_sources"]["geospatial_fabric_gpkg"] = str(gpkg)
        d["data_sources"]["conus_adjacency"] = str(adj)

        dc = OmegaConf.create(d)
        config = validate_config(dc, save_config=False)
        assert isinstance(config, Config)
        assert config.name == "test_run"
