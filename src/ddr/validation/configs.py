import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ddr.validation.enums import GeoDataset, Mode

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
    geospatial_fabric_gpkg: Path = Field(
        description="Path to the geospatial fabric geopackage containing network topology"
    )
    conus_adjacency: Path = Field(
        description="Path to the CONUS adjacency matrix created by engine/adjacency.py"
    )
    statistics: Path = Field(
        default=Path("./data/"),
        description="Path to the folder where normalization statistics files are saved",
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
    forcings: str | None = Field(
        default=None,
        description="Path to icechunk store with meteorological forcings (P, PET, Temp)",
    )
    reservoir_params: str | None = Field(
        default=None,
        description="Path to preprocessed HydroLAKES reservoir parameters CSV (from build_reservoir_params.py)",
    )


_DEFAULT_PARAMETER_RANGES: dict[str, list[float]] = {
    "n": [0.015, 0.25],  # Manning's roughness (s/m¹ᐟ³)
    "q_spatial": [0.0, 1.0],  # Channel shape: 0=rectangular, 1=triangular (-)
    "top_width": [1.0, 5000.0],  # Channel top width, log-space (m)
    "side_slope": [0.5, 50.0],  # H:V ratio, log-space (-)
    "K_D_delta": [-3.0, 1.0],  # Log-space delta from Cosby PTF prior (-)
    "d_gw": [0.01, 300.0],  # Depth to water table from ground surface (m)
    "leakance_gate": [0.0, 1.0],  # Binary STE gate for leakance (-)
}


class LossConfig(BaseModel):
    """Multi-component hydrograph loss configuration."""

    model_config = ConfigDict(extra="forbid")

    overall_weight: float = Field(
        default=0.01, description="Weight for overall MSE component (all timesteps, un-normalized)"
    )
    peak_weight: float = Field(
        default=1.0, description="Weight for peak amplitude (high-flow) loss component"
    )
    baseflow_weight: float = Field(default=1.0, description="Weight for baseflow (low-flow) loss component")
    timing_weight: float = Field(
        default=0.5, description="Weight for temporal gradient (timing) loss component"
    )
    peak_percentile: float = Field(
        default=0.98, description="Percentile threshold (0–1) for peak flow selection"
    )
    baseflow_percentile: float = Field(
        default=0.30, description="Percentile threshold (0–1) for baseflow selection"
    )
    eps: float = Field(default=0.1, description="Stabilization constant added to variance denominators")


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
        default_factory=lambda: dict(_DEFAULT_PARAMETER_RANGES),
        description="The parameter space bounds [min, max] to project learned physical values to. "
        "Partial overrides are merged with defaults — only specify the ranges you want to change.",
    )

    @field_validator("parameter_ranges", mode="before")
    @classmethod
    def merge_parameter_ranges_with_defaults(cls, v: dict[str, list[float]]) -> dict[str, list[float]]:
        """Merge user-provided parameter_ranges on top of defaults.

        This allows YAML configs to specify only the ranges they want to override
        (e.g. just n) without wiping out defaults for other parameters.
        """
        if isinstance(v, dict):
            merged = dict(_DEFAULT_PARAMETER_RANGES)
            merged.update(v)
            return merged
        return v

    log_space_parameters: list[str] = Field(
        default_factory=lambda: [
            "top_width",
            "side_slope",
            "d_gw",
        ],
        description="Parameters to denormalize in log-space for right-skewed distributions",
    )
    defaults: dict[str, int | float] = Field(
        default_factory=lambda: {
            "p_spatial": 21,
        },
        description="Default parameter values for physical processes when not learned",
    )
    use_leakance: bool = Field(
        default=False,
        description="Enable groundwater-surface water exchange (leakance) in routing. "
        "When True, K_D_delta and d_gw must be in params.parameter_ranges.",
    )
    use_reservoir: bool = Field(
        default=False,
        description="Enable level pool reservoir routing for reaches intersecting HydroLAKES waterbodies.",
    )
    ptf_sand_var: str = Field(
        default="SoilGrids1km_sand",
        description="Sand % variable name in the attribute dataset for Cosby PTF",
    )
    ptf_clay_var: str = Field(
        default="SoilGrids1km_clay",
        description="Clay % variable name in the attribute dataset for Cosby PTF",
    )
    tau: int = Field(
        default=3,
        description="Routing time step adjustment parameter to handle double routing and timezone differences",
    )
    save_path: Path = Field(
        default=Path("./"), description="Directory path where model outputs and checkpoints will be saved"
    )


class CudaLstm(BaseModel):
    """LSTM configuration for time-varying parameter prediction."""

    model_config = ConfigDict(extra="forbid")

    hidden_size: int = Field(default=64, description="LSTM hidden dimension")
    num_layers: int = Field(default=1, description="Number of LSTM layers")
    dropout: float = Field(
        default=0.5,
        description="Dropout rate (applied between LSTM layers if num_layers > 1)",
    )
    forcing_var_names: list[str] = Field(
        default_factory=lambda: ["P", "PET", "Temp"],
        description="Forcing variable names used as LSTM dynamic inputs",
    )
    input_var_names: list[str] = Field(
        description="Static attribute names used as LSTM inputs alongside forcings"
    )
    learnable_parameters: list[str] = Field(
        default_factory=lambda: ["d_gw"],
        description="Names of time-varying parameters the LSTM will learn to predict",
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
    gate_parameters: list[str] = Field(
        default_factory=list,
        description="Parameters that use binary STE gating (bias initialized to OFF)",
    )
    off_parameters: list[str] = Field(
        default_factory=list,
        description="Parameters with bias initialized to -2.0 (sigmoid ≈ 0.12, default OFF). "
        "Unlike gate_parameters, these remain continuous (no binary STE).",
    )


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
    max_area_diff_sqkm: float | None = Field(
        default=50,
        description="Maximum absolute drainage area difference (km²) between USGS gage and COMID. "
        "Gages exceeding this threshold are excluded from training/evaluation. None disables filtering. "
        "For MERIT geodataset, the DA_VALID column in gage CSVs is preferred.",
    )
    learning_rate: dict[int, float] = Field(
        default_factory=lambda: {1: 0.001, 5: 0.0005, 9: 0.0001},
        description="Learning rate schedule mapping epoch number to LR. "
        "At each epoch, the most recent entry at or before the current epoch is used.",
    )
    loss: LossConfig = Field(
        default_factory=LossConfig,
        description="Multi-component hydrograph loss weights and thresholds",
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
    geodataset: GeoDataset = Field(description="The geospatial dataset used in predictions and routing")
    mode: Mode = Field(description="Operating mode: training, testing, or routing")
    params: Params = Field(description="Physical and numerical parameters for the routing model")
    kan: Kan = Field(description="Architecture and configuration settings for the Kolmogorov-Arnold Network")
    cuda_lstm: CudaLstm = Field(
        description="CudaLSTM config for time-varying parameter prediction (d_gw).",
    )
    np_seed: int = Field(default=42, description="Random seed for NumPy operations to ensure reproducibility")
    seed: int = Field(default=42, description="Random seed for PyTorch operations to ensure reproducibility")
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

        # LSTM and KAN learnable_parameters must not overlap
        lstm_params = set(self.cuda_lstm.learnable_parameters)
        kan_params = set(self.kan.learnable_parameters)
        overlap = lstm_params & kan_params
        if overlap:
            raise ValueError(f"LSTM and KAN learnable_parameters must not overlap. Found in both: {overlap}")

        # All LSTM + KAN params must exist in parameter_ranges
        all_learned = lstm_params | kan_params
        missing = [p for p in all_learned if p not in self.params.parameter_ranges]
        if missing:
            raise ValueError(
                f"Parameters {missing} are in learnable_parameters but missing from parameter_ranges"
            )

        # Forcings store required for LSTM dynamic inputs
        if self.data_sources.forcings is None:
            raise ValueError(
                "data_sources.forcings must be set "
                "(path to icechunk store with meteorological forcings for the LSTM)"
            )

        # When use_leakance=True, K_D_delta, d_gw, and leakance_gate must be in parameter_ranges
        if self.params.use_leakance:
            required_leakance_params = ["K_D_delta", "d_gw", "leakance_gate"]
            missing_ranges = [p for p in required_leakance_params if p not in self.params.parameter_ranges]
            if missing_ranges:
                raise ValueError(f"use_leakance=True requires {missing_ranges} in params.parameter_ranges")

            # leakance_gate must be in KAN learnable_parameters when leakance is enabled
            if "leakance_gate" not in self.kan.learnable_parameters:
                raise ValueError("use_leakance=True requires 'leakance_gate' in kan.learnable_parameters")

        # When use_reservoir=True, reservoir_params path must be provided
        if self.params.use_reservoir:
            if self.data_sources.reservoir_params is None:
                raise ValueError("use_reservoir=True requires data_sources.reservoir_params")

        # All gate_parameters must be in kan.learnable_parameters
        invalid_gates = [g for g in self.kan.gate_parameters if g not in self.kan.learnable_parameters]
        if invalid_gates:
            raise ValueError(
                f"gate_parameters {invalid_gates} not found in kan.learnable_parameters. "
                f"gate_parameters must be a subset of learnable_parameters."
            )

        # All off_parameters must be in kan.learnable_parameters
        invalid_off = [p for p in self.kan.off_parameters if p not in self.kan.learnable_parameters]
        if invalid_off:
            raise ValueError(
                f"off_parameters {invalid_off} not found in kan.learnable_parameters. "
                f"off_parameters must be a subset of learnable_parameters."
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
