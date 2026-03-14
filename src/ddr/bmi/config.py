"""BMI initialization config schema.

Defines the YAML config format for DDR's BMI wrapper. This config points to
the full DDR Hydra config and trained KAN checkpoint, keeping BMI-specific
settings separate from DDR's internal configuration.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class BmiInitConfig(BaseModel):
    """Schema for the BMI initialization YAML config file.

    Parameters
    ----------
    ddr_config : Path
        Path to DDR's Hydra YAML config file.
    kan_checkpoint : Path
        Path to trained KAN .pt checkpoint file.
    hydrofabric_gpkg : Path or None
        Override hydrofabric GeoPackage path from ddr_config.
    conus_adjacency : Path or None
        Override adjacency matrix path from ddr_config.
    device : str
        Compute device ("cpu", "cuda", "cuda:0", etc.).
    timestep_seconds : float
        Internal MC routing timestep in seconds. Can be smaller than
        ngen's coupling interval for sub-stepping (e.g., 900s routing
        with 3600s ngen_dt gives 4 sub-steps per coupling).
    interpolation : {"constant", "linear"}
        Lateral inflow interpolation between ngen coupling intervals
        when sub-stepping. "constant" holds inflows fixed (zeroth-order);
        "linear" interpolates from previous to current inflows across
        sub-steps. See ``data/diagrams/bmi_testing_guide.txt`` for
        mass conservation implications.
    """

    ddr_config: Path = Field(description="Path to DDR Hydra YAML config")
    kan_checkpoint: Path = Field(description="Path to trained KAN checkpoint")
    hydrofabric_gpkg: Path | None = Field(default=None, description="Override hydrofabric GeoPackage path")
    conus_adjacency: Path | None = Field(default=None, description="Override adjacency matrix path")
    device: str = Field(default="cpu", description="Compute device")
    timestep_seconds: float = Field(default=3600.0, description="Internal MC routing timestep in seconds")
    interpolation: Literal["constant", "linear"] = Field(
        default="constant",
        description="Lateral inflow interpolation for sub-stepping: 'constant' or 'linear'",
    )
