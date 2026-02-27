"""Session-scoped fixtures for integration tests.

These fixtures load real HPC data once and share it across all integration tests.
The Merit dataset init takes ~30s so session-scoping is critical for performance.
"""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from ddr import dmc, kan, streamflow
from ddr.geodatazoo.merit import Merit
from ddr.validation.configs import Config, validate_config

_GAGE_CSV = Path(__file__).parent / "integration_test_gages.csv"


def _build_integration_config(tmp_path: Path) -> DictConfig:
    """Build a DictConfig matching production structure but with small test params."""
    cfg_dict = {
        "mode": "training",
        "geodataset": "merit",
        "name": "integration-test",
        "device": "cpu",
        "seed": 42,
        "np_seed": 42,
        "s3_region": "us-east-2",
        "data_sources": {
            "attributes": "/projects/mhpi/tbindas/ddr/data/merit_global_attributes_v2.nc",
            "geospatial_fabric_gpkg": (
                "/projects/mhpi/data/MERIT/raw/continent/riv_pfaf_7_MERIT_Hydro_v07_Basins_v01_bugfix1.shp"
            ),
            "conus_adjacency": "/projects/mhpi/tbindas/ddr/data/merit_conus_adjacency.zarr",
            "gages_adjacency": "/projects/mhpi/tbindas/ddr/data/merit_gages_conus_adjacency.zarr",
            "statistics": "/projects/mhpi/tbindas/ddr/data/statistics",
            "streamflow": "/projects/mhpi/tbindas/ddr/data/merit_dhbv2_UH_retrospective",
            "observations": "/projects/mhpi/data/icechunk/usgs_daily_observations",
            "gages": str(_GAGE_CSV),
        },
        "experiment": {
            "batch_size": 10,
            "epochs": 2,
            "rho": 90,
            "start_time": "1985/10/01",
            "end_time": "1986/09/30",
            "warmup": 3,
            "shuffle": True,
            "learning_rate": {1: 0.005},
            "checkpoint": None,
            "loss": {
                "overall_weight": 0.01,
                "peak_weight": 1.0,
                "baseflow_weight": 1.0,
                "timing_weight": 0.5,
            },
        },
        "params": {
            "save_path": str(tmp_path),
        },
        "kan": {
            "hidden_size": 21,
            "input_var_names": [
                "FW",
                "aridity",
                "meanelevation",
                "meanP",
                "NDVI",
                "meanslope",
                "log10_uparea",
                "glaciers",
                "ETPOT_Hargr",
                "Porosity",
            ],
            "num_hidden_layers": 2,
            "learnable_parameters": [
                "q_spatial",
                "top_width",
                "side_slope",
                "n",
            ],
            "grid": 50,
            "k": 2,
        },
    }
    return OmegaConf.create(cfg_dict)


@pytest.fixture(scope="session")
def integration_config(tmp_path_factory: pytest.TempPathFactory) -> Config:
    """Validated Config for integration tests."""
    tmp_path = tmp_path_factory.mktemp("integration")
    (tmp_path / "plots").mkdir(exist_ok=True)
    (tmp_path / "saved_models").mkdir(exist_ok=True)
    raw_cfg = _build_integration_config(tmp_path)
    return validate_config(raw_cfg, save_config=False)


@pytest.fixture(scope="session")
def integration_dataset(integration_config: Config) -> Merit:
    """Merit dataset loaded from real HPC data."""
    return Merit(cfg=integration_config)


@pytest.fixture(scope="session")
def integration_models(
    integration_config: Config,
) -> tuple[kan, dmc, streamflow]:
    """All models needed for training: (kan, routing_model, flow_reader)."""
    cfg = integration_config
    nn = kan(
        input_var_names=cfg.kan.input_var_names,
        learnable_parameters=cfg.kan.learnable_parameters,
        hidden_size=cfg.kan.hidden_size,
        num_hidden_layers=cfg.kan.num_hidden_layers,
        grid=cfg.kan.grid,
        k=cfg.kan.k,
        seed=cfg.seed,
        device=cfg.device,
        gate_parameters=cfg.kan.gate_parameters,
        off_parameters=cfg.kan.off_parameters,
    )
    routing_model = dmc(cfg=cfg, device=cfg.device)
    flow = streamflow(cfg)
    return nn, routing_model, flow
