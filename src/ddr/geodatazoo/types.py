"""Data types for dataset module."""

import numpy as np
from pydantic import BaseModel


class ChannelParameters(BaseModel):
    """Channel parameter data"""

    length: np.ndarray  # Channel length (meters)
    slope: np.ndarray  # Channel slope (dimensionless)
    top_width: np.ndarray  # Top width (meters)
    side_slope: np.ndarray  # Side slope ratio (horizontal:vertical)
    x: np.ndarray  # Muskingum X parameter (0-0.5)


class NetworkMetadata(BaseModel):
    """Metadata about a river network"""

    name: str
    version: str
    num_reaches: int
    coordinate_system: str
    id_prefix: str
    description: str
