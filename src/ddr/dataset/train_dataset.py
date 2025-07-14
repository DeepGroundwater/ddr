import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset as TorchDataset

from ddr.dataset.attributes import AttributesReader
from ddr.dataset.Dates import Dates
from ddr.dataset.observations import IcechunkUSGSReader
from ddr.dataset.statistics import set_statistics
from ddr.dataset.utils import (
    Hydrofabric,
    construct_network_matrix,
    create_hydrofabric_observations,
    fill_nans,
    read_zarr,
)

log = logging.getLogger(__name__)


class train_dataset(TorchDataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.train)

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

        self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
        self.observations = self.obs_reader.read_data(dates=self.dates)
        self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])

        # self.network = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="network")
        # self.divides = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divides").set_index(
        #     "divide_id"
        # )
        # self.divide_attr = gpd.read_file(
        #     cfg.data_sources.local_hydrofabric, layer="divide-attributes"
        # ).set_index("divide_id")
        self.flowpath_attr = gpd.read_file(
            cfg.data_sources.hydrofabric_gpkg, layer="flowpath-attributes-ml"
        ).set_index("id")
        # self.flowpaths = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpaths").set_index("id")
        # self.nexus = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="nexus")

        # TODO add logic for multiple gauges
        # TODO add sparse logic
        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.gages_adjacency = read_zarr(Path(cfg.data_sources.gages_adjacency))
        # self.adjacency_matrix, root_coo = read_coo(Path(cfg.data_sources.network), self.gage_ids[0])
        # self.order = root_coo["order"][:]
        # self.network_matrix = torch.tensor(
        #     self.adjacency_matrix.todense(), dtype=torch.float32, device=cfg.device
        # )

        # wb_ordered_index = [f"wb-{_id}" for _id in self.order]
        # cat_ordered_index = [f"cat-{_id}" for _id in self.order]
        # self.divides_sorted = self.divides.reindex(cat_ordered_index)
        # self.divide_attr_sorted = self.divide_attr.reindex(self.divides_sorted.index)

        # self.flowpaths_sorted = self.flowpaths.reindex(wb_ordered_index)
        # self.flowpath_attr = self.flowpath_attr[~self.flowpath_attr.index.duplicated(keep="first")]
        # self.flowpath_attr_sorted = self.flowpath_attr.reindex(wb_ordered_index)

        # self.idx_mapper = {_id: idx for idx, _id in enumerate(self.divides_sorted.index)}
        # self.catchment_mapper = {_id : idx for idx, _id in enumerate(self.divides_sorted["divide_id"])}

        # self.length = torch.tensor(self.flowpath_attr_sorted["Length_m"].values, dtype=torch.float32)
        # self.slope = torch.tensor(self.flowpath_attr_sorted["So"].values, dtype=torch.float32)
        # self.top_width = torch.tensor(self.flowpath_attr_sorted["TopWdth"].values, dtype=torch.float32)
        # self.side_slope = torch.tensor(self.flowpath_attr_sorted["ChSlp"].values, dtype=torch.float32)
        # self.x = torch.tensor(self.flowpath_attr_sorted["MusX"].values, dtype=torch.float32)

        # self.length = fill_nans(self.length)
        # self.slope = fill_nans(self.slope)
        # self.top_width = fill_nans(self.top_width)
        # self.side_slope = fill_nans(self.side_slope)
        # self.x = fill_nans(self.x)

    def __len__(self) -> int:
        """Returns the total number of gauges in the gages.csv file"""
        return len(self.gage_ids)

    def __getitem__(self, idx) -> str:
        return self.gage_ids[idx].item()

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        """Collate function for the dataset."""
        self.dates.calculate_time_period()

        batch: list[str] = args[0]
        coo, gage_idx, gage_wb = construct_network_matrix(batch, self.gages_adjacency)
        adjacency_matrix = torch.sparse_coo_tensor(
            torch.vstack([torch.from_numpy(coo.row), torch.from_numpy(coo.col)]),
            torch.from_numpy(coo.data),
            device=self.cfg.device,
            dtype=torch.float32,
            size=coo.shape,
        ).to_sparse_csr()

        all_ids = np.unique(np.concatenate([coo.row, coo.col]))
        wb_ids = np.array([f"wb-{_id}" for _id in self.conus_adjacency["order"][:][all_ids]])  # type: ignore
        divide_ids = np.array([f"cat-{_id}" for _id in self.conus_adjacency["order"][:][all_ids]])  # type: ignore
        _spatial_attributes = self.attr_reader(divide_ids=divide_ids)
        spatial_attributes = torch.tensor(
            [_spatial_attributes[attr].values for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
            dtype=torch.float32,
        )

        for r in range(spatial_attributes.shape[0]):
            row_means = torch.nanmean(spatial_attributes[r])
            nan_mask = torch.isnan(spatial_attributes[r])
            spatial_attributes[r, nan_mask] = row_means

        normalized_spatial_attributes = (spatial_attributes - self.means) / self.stds
        normalized_spatial_attributes = normalized_spatial_attributes.T  # transposing for NN inputs

        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=self.gage_ids,
            observations=self.observations,
        )

        self.flowpath_attr_sorted = self.flowpath_attr.loc[wb_ids]

        length = fill_nans(torch.tensor(self.flowpath_attr_sorted["Length_m"].values, dtype=torch.float32))
        slope = fill_nans(torch.tensor(self.flowpath_attr_sorted["So"].values, dtype=torch.float32))
        top_width = fill_nans(torch.tensor(self.flowpath_attr_sorted["TopWdth"].values, dtype=torch.float32))
        side_slope = fill_nans(torch.tensor(self.flowpath_attr_sorted["ChSlp"].values, dtype=torch.float32))
        x = fill_nans(torch.tensor(self.flowpath_attr_sorted["MusX"].values, dtype=torch.float32))

        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=length,
            slope=slope,
            side_slope=side_slope,
            top_width=top_width,
            x=x,
            dates=self.dates,
            adjacency_matrix=adjacency_matrix,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
            divide_ids=divide_ids,
            gage_idx=gage_idx,
            gage_wb=gage_wb,
        )
