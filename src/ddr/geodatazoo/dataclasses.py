"""The geospatial dataclasses used for storing context for routing"""

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import xarray as xr

from ddr.dataset.Dates import Dates

log = logging.getLogger(__name__)


@dataclass
class RoutingDataclass:
    """RoutingDataclass data class."""

    adjacency_matrix: torch.Tensor | None = field(default=None)
    spatial_attributes: torch.Tensor | None = field(default=None)
    length: torch.Tensor | None = field(default=None)
    slope: torch.Tensor | None = field(default=None)
    side_slope: torch.Tensor | None = field(default=None)
    top_width: torch.Tensor | None = field(default=None)
    x: torch.Tensor | None = field(default=None)
    dates: Dates | None = field(default=None)
    normalized_spatial_attributes: torch.Tensor | None = field(default=None)
    observations: xr.Dataset | None = field(default=None)
    divide_ids: np.ndarray | None = field(default=None)
    outflow_idx: list[np.ndarray] | None = field(
        default=None
    )  # Has to be list[np.ndarray] since idx are ragged arrays
    gage_wb: list[str] | None = field(default=None)
