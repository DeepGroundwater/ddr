"""DiffRoute-specific configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DiffRouteConfig(BaseModel):
    """Configuration for DiffRoute LTI routing model."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable/disable DiffRoute comparison",
    )
    irf_fn: str = Field(
        default="muskingum",
        description="IRF model: muskingum, linear_storage, nash_cascade, pure_lag, hayami",
    )
    max_delay: int = Field(
        default=100,
        description="Maximum delay timesteps for LTI router",
    )
    dt: float = Field(
        default=0.0416667,  # 1 hour in days
        description="Timestep in days (must match DDR's internal timestep)",
    )
    k: float | None = Field(
        default=None,
        description="Muskingum k (wave travel time) in days. None = use dt",
    )
    x: float = Field(
        default=0.3,
        description="Muskingum x weighting factor (0-0.5)",
    )
