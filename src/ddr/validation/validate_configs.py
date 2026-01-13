import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

log = logging.getLogger(__name__)


def check_path(v: str) -> Path:
    """Check if the path exists"""
    path = Path(v)
    if not path.exists():
        log.exception(f"Path {v} does not exist")
        raise ValueError(f"Path {v} does not exist")
    return path


class AttributeMinimums(BaseModel):
    """Represents the minimum values for the attributes to maintain physical consistency"""

    model_config = ConfigDict(extra="forbid")

    discharge: float = Field(default=1e-4, description="Minimum discharge value in cubic meters per second")
    slope: float = Field(default=1e-4, description="Minimum channel slope as a dimensionless ratio")
    velocity: float = Field(default=0.01, description="Minimum flow velocity in meters per second")
    depth: float = Field(default=0.01, description="Minimum water depth in meters")
    bottom_width: float = Field(default=0.1, description="Minimum channel bottom width in meters")


class DataSources(BaseModel):
    """Represents the data path sources for the model"""

    model_config = ConfigDict(extra="forbid")

    attributes: str = Field(
        default="s3://mhpi-spatial/hydrofabric_v2.2_attributes/",  # MHPI extracted spatial attributes for HF v2.2
        description="Path to the icechunk store containing catchment attribute data",
    )
    hydrofabric_gpkg: str = Field(
        description="Path to the CONUS hydrofabric geopackage containing network topology"
    )
    conus_adjacency: str = Field(
        description="Path to the CONUS adjacency matrix created by engine/adjacency.py"
    )
    statistics: str = Field(
        default="./data/", description="Path to the folder where normalization statistics files are saved"
    )
    streamflow: str = Field(
        default="s3://mhpi-spatial/hydrofabric_v2.2_dhbv_retrospective",  # MHPI dhbv v2.2 streamflow retrospective
        description="Path to the icechunk store containing modeled streamflow data",
    )
    observations: str = Field(
        default="s3://mhpi-spatial/usgs_streamflow_observations/",  # MHPI versioned USGS data
        description="Path to the USGS streamflow observations for model validation",
    )
    gages: str | None = Field(
        default=None, description="Path to CSV file containing gauge metadata, or None to use all segments"
    )
    gages_adjacency: str | None = Field(
        default=None, description="Path to the gages adjacency matrix (required if gages is provided)"
    )
    target_catchments: list[str] | None = Field(
        default=None, description="Optional list of specific catchment IDs to route to (overrides gages)"
    )


class Params(BaseModel):
    """Parameters configuration"""

    model_config = ConfigDict(extra="forbid")

    attribute_minimums: dict[str, float] = Field(
        description="Minimum values for physical routing components to ensure numerical stability",
        default_factory=lambda: {
            "discharge": 0.0001,
            "slope": 0.0001,
            "velocity": 0.01,
            "depth": 0.01,
            "bottom_width": 0.01,
        },
    )
    parameter_ranges: dict[str, list[float]] = Field(
        default_factory=lambda: {
            "n": [0.01, 0.35],
            "q_spatial": [0.0, 3.0],
        },
        description="The parameter space bounds [min, max] to project learned physical values to",
    )
    defaults: dict[str, int | float] = Field(
        default_factory=lambda: {
            "p_spatial": 21,
        },
        description="Default parameter values for physical processes when not learned",
    )
    tau: int = Field(
        default=3,
        description="Routing time step adjustment parameter to handle double routing and timezone differences",
    )
    save_path: str | Path = Field(
        default="./", description="Directory path where model outputs and checkpoints will be saved"
    )


class Kan(BaseModel):
    """KAN (Kolmogorov-Arnold Network) configuration"""

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(
        default=11,
        description="Number of neurons in each hidden layer of the KAN. This should be 2n+1 where n is the number of input attributes",
    )
    input_var_names: list[str] = Field(description="Names of catchment attributes used as network inputs")
    num_hidden_layers: int = Field(default=1, description="Number of hidden layers in the KAN architecture")
    learnable_parameters: list[str] = Field(
        description="Names of physical parameters the network will learn to predict",
        default_factory=lambda: ["n", "q_spatial"],
    )
    grid: int = Field(default=3, description="Grid size for KAN spline basis functions")
    k: int = Field(default=3, description="Order of B-spline basis functions in KAN layers")


class ExperimentConfig(BaseModel):
    """Experiment configuration for training and testing"""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        default=1, description="Number of gauge catchments processed simultaneously in each batch"
    )
    start_time: str = Field(
        default="1981/10/01", description="Start date for time period selection in YYYY/MM/DD format"
    )
    end_time: str = Field(
        default="1995/09/30", description="End date for time period selection in YYYY/MM/DD format"
    )
    checkpoint: Path | None = Field(
        default=None, description="Path to checkpoint file (.pt) for resuming model from previous state"
    )
    epochs: int = Field(default=1, description="Number of complete passes through the training dataset")
    learning_rate: dict[int, float] = Field(
        default_factory=lambda: {1: 0.005, 3: 0.001},
        description="Learning rate schedule mapping epoch numbers to learning rate values",
    )
    rho: int | None = Field(
        default=None, description="Number of consecutive days selected in each training batch"
    )
    shuffle: bool = Field(
        default=True, description="Whether to randomize the order of samples in the dataloader"
    )
    warmup: int = Field(
        default=3,
        description="Number of days excluded from loss calculation as routing starts from dry conditions",
    )

    @field_validator("checkpoint", mode="before")
    @classmethod
    def validate_checkpoint(cls, v: str | Path | None) -> Path | None:
        """Validate the checkpoint path exists if provided"""
        if v is None:
            return None
        if isinstance(v, Path):
            return v
        return check_path(str(v))


class Config(BaseModel):
    """The base level configuration for the dMC (differentiable Muskingum-Cunge) model"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True, str_strip_whitespace=True)

    name: str = Field(description="Unique identifier name for this model run used in output file naming")
    data_sources: DataSources = Field(
        description="Configuration of all data source paths required by the model"
    )
    experiment: ExperimentConfig = Field(
        default_factory=ExperimentConfig,
        description="Experiment settings controlling training behavior and data selection",
    )
    params: Params = Field(description="Physical and numerical parameters for the routing model")
    kan: Kan = Field(description="Architecture and configuration settings for the Kolmogorov-Arnold Network")
    np_seed: int = Field(default=1, description="Random seed for NumPy operations to ensure reproducibility")
    seed: int = Field(default=0, description="Random seed for PyTorch operations to ensure reproducibility")
    device: int | str = Field(
        default=0, description="Compute device specification (GPU index number, 'cpu', or 'cuda', or 'mps')"
    )
    s3_region: str = Field(
        default="us-east-2", description="AWS S3 region for accessing cloud-stored datasets"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: int | str) -> int | str:
        """Validate device configuration"""
        if isinstance(v, str):
            if v not in ["cpu", "cuda", "mps"]:
                log.warning(f"Unknown device string '{v}', proceeding anyway")
        elif isinstance(v, int):
            if v < 0:
                raise ValueError("Device ID must be non-negative")
        return v

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "Config":
        """Validate configuration consistency"""
        # Set save_path if using default and Hydra is available
        if self.params.save_path == "./":
            try:
                hydra_run_dir = HydraConfig.get().run.dir
                # Create a new params object with updated save_path
                self.params = self.params.model_copy(update={"save_path": hydra_run_dir})
            except ValueError:
                log.info(
                    "HydraConfig is not set. Using default save_path './'. "
                    "If using a jupyter notebook, manually set save_path."
                )

        return self


def _set_seed(cfg: Config) -> None:
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.np_seed)
    random.seed(cfg.seed)


def _save_cfg(cfg: Config) -> None:
    import warnings

    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=r"^Pydantic serializer warnings:\n.*Expected `str` but got `PosixPath`.*",
    )
    save_path = Path() / "pydantic_config.yaml"
    json_cfg = cfg.model_dump_json(indent=4)
    log.info(
        "\n"
        + "======================================\n"
        + "Running DDR with the following config:\n"
        + "======================================\n"
        + f"{json_cfg}\n"
        + "======================================\n"
    )

    with save_path.open("w") as f:
        OmegaConf.save(config=OmegaConf.create(json_cfg), f=f)


def validate_config(cfg: DictConfig, save_config: bool = True) -> Config:
    """Creating the Pydantic config object from the DictConfig

    Parameters
    ----------
    cfg : DictConfig
        The Hydra DictConfig object
    save_config: bool, optional
        A check of whether to save the config outputs or not. Tests set this to false

    Returns
    -------
    Config
        The Pydantic Config object

    """
    try:
        # Convert the DictConfig to a dictionary and then to a Config object for validation
        config_dict: dict[str, Any] | Any = OmegaConf.to_container(cfg, resolve=True)
        config = Config(**config_dict)
        _set_seed(cfg=config)
        if save_config:
            _save_cfg(cfg=config)
        return config
    except ValidationError as e:
        log.exception(e)
        raise e
