from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd
import numpy as np
import torch
import xarray as xr
from scipy import sparse
from torch.utils.data import Dataset

from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.validation.configs import Config
from ddr.validation.enums import Mode


class BaseGeoDataset(Dataset, ABC):
    """Abstract base class for PyTorch datasets."""

    cfg: Config
    dates: Dates
    gage_ids: np.ndarray | None
    routing_dataclass: RoutingDataclass | None
    attribute_ds: xr.Dataset

    def __len__(self) -> int:
        """A shared __len__ for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            assert self.gage_ids is not None, "No gage IDs found, cannot batch"
            return len(self.gage_ids)
        return len(self.dates.daily_time_range)

    def __getitem__(self, idx: int) -> str | int:
        """A shared __getitem__ for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            assert self.gage_ids is not None, "No gage IDs found, cannot batch"
            return str(self.gage_ids[idx])
        return idx

    def collate_fn(self, *args: Any, **kwargs: Any) -> RoutingDataclass:
        """A shared collate_fn for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            self.dates.calculate_time_period()
            return self._collate_gages(np.array(args[0]))

        assert self.routing_dataclass is not None, "No RoutingDataclass, cannot batch"
        indices = list(args[0])
        if 0 not in indices:
            indices.insert(0, indices[0] - 1)
        self.dates.set_date_range(np.array(indices))
        return self.routing_dataclass

    @abstractmethod
    def _load_attributes(self) -> xr.Dataset:
        """Load the attribute dataset into memory."""
        raise NotImplementedError

    @abstractmethod
    def _get_attributes(
        self,
        catchment_ids: np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Fetch attributes for the given divide IDs."""
        raise NotImplementedError

    @abstractmethod
    def _collate_gages(self, batch: np.ndarray) -> RoutingDataclass:
        """Build routing data for a batch of gages (training)."""
        raise NotImplementedError

    @abstractmethod
    def _init_training(self) -> None:
        """Initialize dataset for training mode."""
        raise NotImplementedError

    @abstractmethod
    def _init_inference(self) -> None:
        """Initialize dataset for inference mode."""
        raise NotImplementedError

    @abstractmethod
    def _build_common_tensors(
        self,
        csr_matrix: sparse.csr_matrix,
        catchment_ids: np.ndarray,
        flowpath_attr: gpd.GeoDataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build tensors common to all collate methods."""
