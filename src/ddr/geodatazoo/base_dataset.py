"""A base ABC dataclass for other geospatial fabrics to replicate from for making routing datasets"""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset

from ddr.geodatazoo.dataclasses import RoutingDataclass


class BaseDataset(Dataset, ABC):
    """Abstract base class for PyTorch datasets."""

    @abstractmethod
    def __len__(self):
        """Abstract __len__ function"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """Abstract __getitem__ function"""
        raise NotImplementedError

    @abstractmethod
    def collate_fn(self, *args, **kwargs) -> RoutingDataclass:
        """Abstract collate function"""
        raise NotImplementedError
