"""This module contains providers for the NextGen Hydrofabric v2.2+"""

from .attributes import NextGenAttributeProvider
from .network import NextGenNetworkProvider
from .streamflow import NextGenStreamflowProvider

__all__ = [
    "NextGenNetworkProvider",
    "NextGenAttributeProvider",
    "NextGenStreamflowProvider",
]
