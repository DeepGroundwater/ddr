import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset as TorchDataset

from ddr.dataset.attributes import AttributesReader
from ddr.dataset.Dates import Dates
from ddr.dataset.statistics import set_statistics
from ddr.dataset.utils import (
    Hydrofabric,
    fill_nans,
    naninfmean,
    read_zarr,
)
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class LargeScaleDataset(TorchDataset):
    """Runs through all data sequentially over a specific amount of timesteps"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.experiment.model_dump())

        self.attr_reader = AttributesReader(cfg=self.cfg)
        self.attr_stats = set_statistics(self.cfg, self.attr_reader.ds)
        # Convert to tensor after collecting all valid data
        self.means = torch.tensor(
            [self.attr_stats[attr].iloc[2] for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)  # Mean is always idx 2
        self.stds = torch.tensor(
            [self.attr_stats[attr].iloc[3] for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)  # Mean is always idx 3

        _flowpath_attr = gpd.read_file(
            self.cfg.data_sources.hydrofabric_gpkg, layer="flowpath-attributes-ml"
        ).set_index("id")
        self.flowpath_attr = _flowpath_attr[~_flowpath_attr.index.duplicated(keep="first")]

        self.phys_means = torch.tensor(
            [
                naninfmean(self.flowpath_attr[attr].values)
                for attr in ["Length_m", "So", "TopWdth", "ChSlp", "MusX"]
            ],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)  # Creating mean values for physical parameters within the HF

        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.hf_ids = self.conus_adjacency["order"][:]  # type: ignore

        coordinates = set()  # (indices_0, indices_1)
        _r = self.conus_adjacency["indices_0"][:].tolist()  # type: ignore
        _c = self.conus_adjacency["indices_1"][:].tolist()  # type: ignore
        for row, col in zip(_r, _c, strict=False):
            coordinates.add((row, col))
        _attrs: dict[str, Any] = dict(self.conus_adjacency.attrs)
        if coordinates:
            rows, cols = zip(*coordinates, strict=False)
            rows = list(rows)
            cols = list(cols)
        else:
            raise ValueError("No coordinate-pairs found. Cannot construct a matrix")
        shape = tuple(_attrs["shape"])  # type: ignore
        csr = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=shape,
        ).tocsr()

        # Create waterbody and divide IDs for the compressed matrix
        wb_ids = np.array([f"wb-{_id}" for _id in self.hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in self.hf_ids])

        # Get subset of flowpath attributes for this batch
        compressed_flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        # Create PyTorch sparse tensor with compressed 135x135 matrix
        adjacency_matrix = torch.sparse_csr_tensor(
            crow_indices=csr.indptr,
            col_indices=csr.indices,
            values=csr.data,
            size=csr.shape,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        spatial_attributes = self.attr_reader(
            divide_ids=divide_ids,
            attr_means=self.means,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        for r in range(spatial_attributes.shape[0]):
            row_means = torch.nanmean(spatial_attributes[r])
            nan_mask = torch.isnan(spatial_attributes[r])
            spatial_attributes[r, nan_mask] = row_means

        normalized_spatial_attributes = (spatial_attributes - self.means) / self.stds
        normalized_spatial_attributes = normalized_spatial_attributes.T  # transposing for NN inputs

        length = fill_nans(
            torch.tensor(compressed_flowpath_attr["Length_m"].values, dtype=torch.float32),
            row_means=self.phys_means[0],
        )
        slope = fill_nans(
            torch.tensor(compressed_flowpath_attr["So"].values, dtype=torch.float32),
            row_means=self.phys_means[1],
        )
        top_width = fill_nans(
            torch.tensor(compressed_flowpath_attr["TopWdth"].values, dtype=torch.float32),
            row_means=self.phys_means[2],
        )
        side_slope = fill_nans(
            torch.tensor(compressed_flowpath_attr["ChSlp"].values, dtype=torch.float32),
            row_means=self.phys_means[3],
        )
        x = fill_nans(
            torch.tensor(compressed_flowpath_attr["MusX"].values, dtype=torch.float32),
            row_means=self.phys_means[4],
        )

        self.hydrofabric = Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=length,
            slope=slope,
            side_slope=side_slope,
            top_width=top_width,
            x=x,
            dates=self.dates,
            adjacency_matrix=adjacency_matrix,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=None,
            divide_ids=divide_ids,
            gage_idx=None,
            gage_wb=None,
        )

    def __len__(self) -> int:
        """Returns the total number of days that we're evaluating over"""
        return len(self.hf_ids)

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs) -> None:
        """Since there is no batching here, we have no collate_fn"""
        pass
