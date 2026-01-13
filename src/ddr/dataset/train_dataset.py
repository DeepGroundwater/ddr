import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch
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
    naninfmean,
    read_zarr,
)
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class TrainDataset(TorchDataset):
    """train_dataset class for handling dataset operations for training dMC models"""

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

        # Determine routing mode
        if cfg.data_sources.target_catchments is not None:
            # Training catchments mode: specific catchments, no observations
            self.mode = "target_catchments"
            self.target_catchments = cfg.data_sources.target_catchments
            self.obs_reader = None
            self.gage_ids = None
            self.observations = None
            self.gages_adjacency = None
            log.info(
                f"Target Catchments mode: routing outputs for target catchments: {self.target_catchments}"
            )

        elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
            # Gages mode: route to gauge locations with observations
            self.mode = "gages"
            self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
            self.observations = self.obs_reader.read_data(dates=self.dates)
            self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])
            self.gages_adjacency = read_zarr(Path(cfg.data_sources.gages_adjacency))
            log.info(f"Gages mode: routing for {len(self.gage_ids)} gauged locations")

        else:
            # All segments mode: route entire domain
            self.mode = "all"
            self.target_ids = None
            self.obs_reader = None
            self.gage_ids = None
            self.observations = None
            self.gages_adjacency = None
            log.info("All segments mode: across all catchments within the hydrofabric")

    def __len__(self) -> int:
        """Returns the total number of gauges in the gages.csv file"""
        assert self.gage_ids is not None, "Cannot train model if no Gage IDs"
        return len(self.gage_ids)

    def __getitem__(self, idx) -> str:
        assert self.gage_ids is not None, "Cannot train model if no Gage IDs"
        return self.gage_ids[idx].item()

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        """
        Collate function for the dataset.
        Routes to appropriate collate method based on mode.
        """
        self.dates.calculate_time_period()

        if self.mode == "target_catchments":
            return self._collate_target_catchments(np.array(args[0]))
        elif self.mode == "gages":
            return self._collate_gages(np.array(args[0]))
        else:
            return self._collate_all_segments()

    def _collate_target_catchments(self, batch: np.ndarray) -> Hydrofabric:
        """Route to specific target catchments without observations"""
        raise NotImplementedError("Target catchments mode not implemented for training basins")

    def _collate_gages(self, batch: np.ndarray) -> Hydrofabric:
        """Route to gauge locations with observations"""
        # Filter observations based on batch and what gauges exist in the zarr store/HF
        valid_gauges_mask = np.isin(batch, list(self.gages_adjacency.keys()))
        batch = batch[valid_gauges_mask].tolist()

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
        compressed_flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        # Update local_col_idx to use compressed indices
        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            original_col_indices = coo.col[local_gage_inflow_idx]
            compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            outflow_idx.append(compressed_col_indices)

        assert (
            np.array(
                [
                    _id.split("-")[1]
                    for _id in compressed_flowpath_attr.iloc[np.concatenate(outflow_idx)]["to"]
                    .drop_duplicates(keep="first")
                    .values
                ]
            )
            == np.array([_id.split("-")[1] for _id in gage_wb])
        ).all(), (
            "Gage WB don't match up with indices. There is something wrong with your batching and how it's loading in sparse matrices from the engine"
        )

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, divide_ids, compressed_flowpath_attr)
        )

        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=np.array(batch),
            observations=self.observations,
        )

        log.info(f"Created an adjacency matrix of shape: {adjacency_matrix.shape}")
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=flowpath_tensors["length"],
            slope=flowpath_tensors["slope"],
            side_slope=flowpath_tensors["side_slope"],
            top_width=flowpath_tensors["top_width"],
            x=flowpath_tensors["x"],
            dates=self.dates,
            adjacency_matrix=adjacency_matrix,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=hydrofabric_observations,
            divide_ids=divide_ids,
            outflow_idx=outflow_idx,
            gage_wb=gage_wb,
        )

    def _collate_all_segments(self) -> Hydrofabric:
        """Route over all segments using full CONUS adjacency matrix"""
        # Build adjacency matrix from full CONUS adjacency
        csr_matrix = sparse.csr_matrix(
            (
                self.conus_adjacency["data"][:],
                self.conus_adjacency["indices"][:],
                self.conus_adjacency["indptr"][:],
            ),
            shape=(len(self.hf_ids), len(self.hf_ids)),
        )

        wb_ids = np.array([f"wb-{_id}" for _id in self.hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in self.hf_ids])
        flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(csr_matrix, divide_ids, flowpath_attr)
        )

        log.info(f"Created full adjacency matrix of shape: {adjacency_matrix.shape}")
        return Hydrofabric(
            spatial_attributes=spatial_attributes,
            length=flowpath_tensors["length"],
            slope=flowpath_tensors["slope"],
            side_slope=flowpath_tensors["side_slope"],
            top_width=flowpath_tensors["top_width"],
            x=flowpath_tensors["x"],
            dates=self.dates,
            adjacency_matrix=adjacency_matrix,
            normalized_spatial_attributes=normalized_spatial_attributes,
            observations=None,
            divide_ids=divide_ids,
            gage_idx=None,
            gage_wb=None,
        )

    def _build_common_tensors(
        self,
        csr_matrix: sparse.csr_matrix,
        divide_ids: np.ndarray,
        flowpath_attr: gpd.GeoDataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build tensors common to all collate methods"""
        adjacency_matrix = torch.sparse_csr_tensor(
            crow_indices=csr_matrix.indptr,
            col_indices=csr_matrix.indices,
            values=csr_matrix.data,
            size=csr_matrix.shape,
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
        normalized_spatial_attributes = normalized_spatial_attributes.T

        flowpath_tensors = {
            "length": fill_nans(
                torch.tensor(flowpath_attr["Length_m"].values, dtype=torch.float32),
                row_means=self.phys_means[0],
            ),
            "slope": fill_nans(
                torch.tensor(flowpath_attr["So"].values, dtype=torch.float32),
                row_means=self.phys_means[1],
            ),
            "top_width": fill_nans(
                torch.tensor(flowpath_attr["TopWdth"].values, dtype=torch.float32),
                row_means=self.phys_means[2],
            ),
            "side_slope": fill_nans(
                torch.tensor(flowpath_attr["ChSlp"].values, dtype=torch.float32),
                row_means=self.phys_means[3],
            ),
            "x": fill_nans(
                torch.tensor(flowpath_attr["MusX"].values, dtype=torch.float32),
                row_means=self.phys_means[4],
            ),
        }

        return adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors
