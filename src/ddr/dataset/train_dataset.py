from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import xarray as xr
from omegaconf import DictConfig
from scipy.sparse import csc_matrix
from torch.utils.data import Dataset as TorchDataset

from ddr.dataset.Dates import Dates
from ddr.dataset.observations import ZarrUSGSReader
from ddr.dataset.statistics import set_statistics
from ddr.dataset.utils import fill_nans, read_coo

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    adjacency_matrix: Union[torch.Tensor, None] = field(default=None)
    spatial_attributes: Union[torch.Tensor, None] = field(default=None)
    length: Union[torch.Tensor, None] = field(default=None)
    slope: Union[torch.Tensor, None] = field(default=None)
    side_slope: Union[torch.Tensor, None] = field(default=None)
    top_width: Union[torch.Tensor, None] = field(default=None)
    x: Union[torch.Tensor, None] = field(default=None)
    dates: Union[Dates, None] = field(default=None)
    normalized_spatial_attributes: Union[torch.Tensor, None] = field(default=None)
    observations: Union[xr.Dataset, None] = field(default=None)
    transition_matrix: Union[csc_matrix, None] = field(default=None)
    merit_basins: Union[np.ndarray, None] = field(default=None)
    
    
def create_hydrofabric_observations(
    dates: Dates,
    gage_ids: np.ndarray,
    observations: xr.Dataset,
) -> xr.Dataset:
    ds = observations.sel(time=dates.batch_daily_time_range, gage_id=gage_ids)
    return ds


class train_dataset(TorchDataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.train)

        self.obs_reader = ZarrUSGSReader(cfg=self.cfg)
        self.observations = self.obs_reader.read_data(dates=self.dates)
        self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])

        self.network = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="network")
        self.divides = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divides").set_index("divide_id")
        self.divide_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divide-attributes").set_index("divide_id")
        self.flowpath_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpath-attributes-ml").set_index("id")
        self.flowpaths = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpaths").set_index("id")
        self.nexus = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="nexus")

         # TODO add logic for multiple gauges
         # TODO add sparse logic
        self.adjacency_matrix, root_coo = read_coo(Path(cfg.data_sources.network), self.gage_ids[0])
        self.order = root_coo["order"][:]
        self.network_matrix = torch.tensor(self.adjacency_matrix.todense(), dtype=torch.float32, device=cfg.device)
        
        # TODO get mike johnson et al. to fix the subset bug: https://github.com/owp-spatial/hfsubsetR/issues/9
        wb_ordered_index = [f"wb-{_id}" for _id in self.order]
        cat_ordered_index = [f"cat-{_id}" for _id in self.order]
        self.divides_sorted = self.divides.reindex(cat_ordered_index)
        self.divide_attr_sorted = self.divide_attr.reindex(self.divides_sorted.index)
        
        self.flowpaths_sorted = self.flowpaths.reindex(wb_ordered_index)
        self.flowpath_attr = self.flowpath_attr[~self.flowpath_attr.index.duplicated(keep='first')]
        self.flowpath_attr_sorted = self.flowpath_attr.reindex(wb_ordered_index)
        
        # self.idx_mapper = {_id: idx for idx, _id in enumerate(self.divides_sorted.index)}
        # self.catchment_mapper = {_id : idx for idx, _id in enumerate(self.divides_sorted["divide_id"])}
        
        self.length = torch.tensor(self.flowpath_attr_sorted["Length_m"].values, dtype=torch.float32)
        self.slope = torch.tensor(self.flowpath_attr_sorted["So"].values, dtype=torch.float32)
        self.top_width = torch.tensor(self.flowpath_attr_sorted["TopWdth"].values, dtype=torch.float32)
        self.side_slope = torch.tensor(self.flowpath_attr_sorted["ChSlp"].values, dtype=torch.float32)
        self.x = torch.tensor(self.flowpath_attr_sorted["MusX"].values, dtype=torch.float32)
    
        self.length = fill_nans(self.length)
        self.slope = fill_nans(self.slope)
        self.top_width = fill_nans(self.top_width)
        self.side_slope = fill_nans(self.side_slope)
        self.x = fill_nans(self.x)
    
        self.attribute_stats = set_statistics(self.cfg)

        # Convert to tensor after collecting all valid data        
        self.means = torch.tensor([self.attribute_stats[attr].iloc[2] for attr in self.cfg.kan.input_var_names], device=self.cfg.device, dtype=torch.float32).unsqueeze(1)  # Mean is always idx 2
        self.stds = torch.tensor([self.attribute_stats[attr].iloc[3] for attr in self.cfg.kan.input_var_names], device=self.cfg.device, dtype=torch.float32).unsqueeze(1)  # Mean is always idx 3

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        self.dates.calculate_time_period()
        
        spatial_attributes = torch.tensor(
            np.array([self.divide_attr_sorted[attr].values for attr in self.cfg.kan.input_var_names]),
            device=self.cfg.device,
            dtype=torch.float32
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

        tm, tm_root_coo = read_coo(Path(self.cfg.data_sources.transition_matrix), "73")
        csc_tm = tm.tocsc()
        merit_basins_order = tm_root_coo["merit_basins_order"][:] 
        comid_order = tm_root_coo["comid_order"][:]
        col_idx = np.where(np.isin(comid_order, self.order))[0]
        _transition_matrix = csc_tm[:, col_idx]
        mask = np.sum(_transition_matrix, axis=1).A1 > 0 
        transition_matrix = _transition_matrix[mask]      
        merit_basins = merit_basins_order[mask]
        
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=self.length,
            slope=self.slope,
            side_slope=self.side_slope,
            top_width=self.top_width,
            x=self.x,
            dates=self.dates,
            adjacency_matrix=self.network_matrix,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
            transition_matrix=transition_matrix,
            merit_basins=merit_basins
        )
