from dataclasses import dataclass, field
import logging
from typing import Union

import geopandas as gpd
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from ddr.dataset.Dates import Dates
from ddr.dataset.observations import ZarrUSGSReader
from ddr.dataset.statistics import set_statistics

log = logging.getLogger(__name__)


@dataclass
class Hydrofabric:
    leakance_attributes: Union[torch.Tensor, None] = field(default=None)
    
def create_hydrofabric_observations(
    dates: Dates,
    gage_ids: np.ndarray,
    observations: xr.Dataset,
) -> xr.Dataset:
    ds = observations.sel(time=dates.batch_daily_time_range, gage_id=gage_ids)
    return ds


class train_dataset(torch.utils.data.Dataset):
    """train_dataset class for handling dataset operations for training dMC models"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.train)

        self.network = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="network")
        self.divides = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divides")
        self.divide_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="divide-attributes")
        self.flowpath_attr = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpath-attributes-ml")
        self.flowpaths = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="flowpaths")
        self.nexus = gpd.read_file(cfg.data_sources.local_hydrofabric, layer="nexus")

        self.adjacency_matrix = np.load(cfg.data_sources.network)
        
        self.divides_sorted = self.divides.sort_values('tot_drainage_areasqkm')
        self.idx_mapper = {_id: idx for idx, _id in enumerate(self.divides_sorted["id"])}
        self.catchment_mapper = {_id : idx for idx, _id in enumerate(self.divides_sorted["divide_id"])}
        
                
        self.length = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["Length_m"]] * 1000 for _id in self.flowpaths_sorted["id"])  
        self.slope = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["So"]] for _id in self.flowpaths_sorted["id"])      
        self.width = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["TopWdth"]] for _id in self.flowpaths_sorted["id"])
        self.x = torch.tensor([self.flowpath_attr.iloc[self.idx_mapper[_id]]["MusX"]] for _id in self.flowpaths_sorted["id"])         
    
        self.attribute_stats = set_statistics()
        
        self.obs_reader = ZarrUSGSReader(self.cfg)
        self.observations = self._observation_reader.read_data(dates=self.dates)
        self.gage_ids = np.array(str(self.obs_reader.gage_dict["STAID"].zfill(8)))

    def __len__(self) -> int:
        """Returns the total number of gauges."""
        return 1

    def __getitem__(self, idx) -> tuple[int, str, str]:
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        self.dates.calculate_time_period()
        
        spatial_attributes = torch.tensor(
            [[self.divide_attr.iloc[self.catchment_mapper[_id]][attr]] for _id in self.flowpaths_sorted["id"] for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
        )
        
        means = torch.tensor([self.attribute_stats[attr].iloc[2] for attr in self.cfg.kan.input_var_names], device=self.cfg.device)  # Mean is always idx 2
        stds = torch.tensor([self.attribute_stats[attr].iloc[3] for attr in self.cfg.kan.input_var_names], device=self.cfg.device)  # Mean is always idx 3
        
        normalized_spatial_attributes = (spatial_attributes - means) / stds
        
        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=self.gage_ids,
            observations=self.observations,
        )
        
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=self.length,
            slope=self.slope,
            dates=self.dates,
            mapping=self.idx_mapper,
            network=self.network,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
        )
