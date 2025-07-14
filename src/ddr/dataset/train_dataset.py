import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
from omegaconf import DictConfig
from scipy import sparse
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

        _flowpath_attr = gpd.read_file(
            self.cfg.data_sources.hydrofabric_gpkg, layer="flowpath-attributes-ml"
        ).set_index("id")
        self.flowpath_attr = _flowpath_attr[~_flowpath_attr.index.duplicated(keep="first")]

        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.hf_ids = self.conus_adjacency["order"][:]  # type: ignore
        self.gages_adjacency = read_zarr(Path(cfg.data_sources.gages_adjacency))

    def __len__(self) -> int:
        """Returns the total number of gauges in the gages.csv file"""
        return len(self.gage_ids)

    def __getitem__(self, idx) -> str:
        return self.gage_ids[idx].item()

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        """
        Collate function for the dataset.

        NOTE: For doing indexing, we are using the values that are the col_indices from the CSR matrix
        """
        self.dates.calculate_time_period()

        batch: list[str] = args[0]
        # Combines all gauge information together into one large matrix where the CONUS hydrofabric is the indexing
        coo, _gage_idx, gage_wb = construct_network_matrix(batch, self.gages_adjacency)
        local_col_idx = []
        for _i, _idx in enumerate(_gage_idx):
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            local_col_idx.append(coo.col[local_gage_inflow_idx])

        active_indices = np.unique(np.concatenate([coo.row, coo.col]))
        index_mapping = {orig_idx: compressed_idx for compressed_idx, orig_idx in enumerate(active_indices)}

        compressed_rows = np.array([index_mapping[idx] for idx in coo.row])
        compressed_cols = np.array([index_mapping[idx] for idx in coo.col])

        compressed_size = len(active_indices)
        compressed_coo = sparse.coo_matrix(
            (coo.data, (compressed_rows, compressed_cols)), shape=(compressed_size, compressed_size)
        )
        compressed_csr = compressed_coo.tocsr()
        compressed_hf_ids = self.hf_ids[active_indices]

        # Create waterbody and divide IDs for the compressed matrix
        wb_ids = np.array([f"wb-{_id}" for _id in compressed_hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in compressed_hf_ids])

        # Get subset of flowpath attributes for this batch
        compressed_flowpath_attr = self.flowpath_attr.loc[wb_ids]

        # Update local_col_idx to use compressed indices
        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            # Map original column indices to compressed indices
            original_col_indices = coo.col[local_gage_inflow_idx]
            compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            outflow_idx.append(compressed_col_indices)

        # Create PyTorch sparse tensor with compressed 135x135 matrix
        adjacency_matrix = torch.sparse_csr_tensor(
            crow_indices=compressed_csr.indptr,
            col_indices=compressed_csr.indices,
            values=compressed_csr.data,
            size=compressed_csr.shape,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        # NOTE: You can check the accuracy of the CSR compression through the following lines. The "to" should be the same number as gage_wb
        # compressed_flowpath_attr.iloc[np.concatenate(outflow_idx)]
        # gage_wb

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
            gage_ids=np.array(batch),
            observations=self.observations,
        )

        length = fill_nans(torch.tensor(compressed_flowpath_attr["Length_m"].values, dtype=torch.float32))
        slope = fill_nans(torch.tensor(compressed_flowpath_attr["So"].values, dtype=torch.float32))
        top_width = fill_nans(torch.tensor(compressed_flowpath_attr["TopWdth"].values, dtype=torch.float32))
        side_slope = fill_nans(torch.tensor(compressed_flowpath_attr["ChSlp"].values, dtype=torch.float32))
        x = fill_nans(torch.tensor(compressed_flowpath_attr["MusX"].values, dtype=torch.float32))

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
            gage_idx=outflow_idx,
            gage_wb=gage_wb,
        )
