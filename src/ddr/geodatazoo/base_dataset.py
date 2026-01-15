from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset

from ddr.geodatazoo.dataclasses import RoutingDataclass


class BaseDataset(Dataset, ABC):
    """Abstract base class for PyTorch datasets."""

    @abstractmethod
    def __len__(self) -> int:
        """Abstract __len__ function"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """Abstract __getitem__ function"""
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args: Any, **kwargs: Any) -> RoutingDataclass:
        """Abstract collate function"""
        raise NotImplementedError
