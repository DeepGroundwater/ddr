"""Benchmark configuration that extends DDR Config with model-specific configs."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ddr.validation import Config as DDRConfig
from ddr.validation.configs import _save_cfg, _set_seed

from .diffroute import DiffRouteConfig

log = logging.getLogger(__name__)


class BenchmarkConfig(BaseModel):
    """Benchmark configuration wrapping DDR Config + model configs."""

    model_config = ConfigDict(extra="forbid")

    # DDR core config (all standard DDR fields)
    ddr: DDRConfig = Field(description="DDR core configuration")

    # Model-specific configs
    diffroute: DiffRouteConfig = Field(
        default_factory=DiffRouteConfig,
        description="DiffRoute model configuration",
    )
    # Future: rapid: RAPIDConfig, etc.


def validate_benchmark_config(cfg: DictConfig, save_config: bool = True) -> BenchmarkConfig:
    """Validate benchmark config, separating DDR fields from model-specific fields.

    Parameters
    ----------
    cfg : DictConfig
        The Hydra DictConfig with all fields (DDR + diffroute + etc.)
    save_config : bool
        Whether to save the validated config

    Returns
    -------
    BenchmarkConfig
        Validated config with ddr and model-specific sections
    """
    try:
        config_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)

        # Extract model-specific configs
        diffroute_cfg = config_dict.pop("diffroute", {})
        # Future: rapid_cfg = config_dict.pop("rapid", {})

        # Remaining fields are DDR config
        ddr_config = DDRConfig(**config_dict)
        _set_seed(cfg=ddr_config)

        # Build benchmark config
        benchmark_config = BenchmarkConfig(
            ddr=ddr_config,
            diffroute=DiffRouteConfig(**diffroute_cfg),
        )

        if save_config:
            _save_cfg(cfg=ddr_config)

        return benchmark_config

    except ValidationError as e:
        log.exception(e)
        raise e
