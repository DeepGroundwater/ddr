import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rustworkx as rx
import torch
from scipy import sparse
from torch.utils.data import Dataset as TorchDataset

from ddr.dataset.attributes import AttributesReader
from ddr.dataset.Dates import Dates
from ddr.dataset.observations import IcechunkUSGSReader
from ddr.dataset.statistics import set_statistics
from ddr.dataset.utils import (
    Hydrofabric,
    _build_network_graph,
    construct_network_matrix,
    create_hydrofabric_observations,
    fill_nans,
    naninfmean,
    read_zarr,
)
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class TestDataset(TorchDataset):
    """Runs through all data sequentially over a specific amount of timesteps"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dates = Dates(**self.cfg.experiment.model_dump())

        self.attr_reader = AttributesReader(cfg=self.cfg)
        self.attr_stats = set_statistics(self.cfg, self.attr_reader.ds)

        self.means = torch.tensor(
            [self.attr_stats[attr].iloc[2] for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        self.stds = torch.tensor(
            [self.attr_stats[attr].iloc[3] for attr in self.cfg.kan.input_var_names],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

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
        ).unsqueeze(1)

        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.hf_ids = self.conus_adjacency["order"][:]

        # Determine mode and build hydrofabric
        if cfg.data_sources.target_catchments is not None:
            self.mode = "target_catchments"
            self.target_catchments = cfg.data_sources.target_catchments
            self.network_graph, self.hf_id_to_node = _build_network_graph(self.conus_adjacency)
            self.gage_ids = None
            self.observations = None
            self.gages_adjacency = None
            log.info(f"Target catchments mode: routing flow upstream of the {self.target_catchments} outlets")
            self.hydrofabric = self._build_target_catchments_hydrofabric()

        elif cfg.data_sources.gages is not None and cfg.data_sources.gages_adjacency is not None:
            self.mode = "gages"
            self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
            self.observations = self.obs_reader.read_data(dates=self.dates)
            self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])
            self.gages_adjacency = read_zarr(Path(cfg.data_sources.gages_adjacency))
            log.info(f"Gages mode: {len(self.gage_ids)} gauged locations")
            self.hydrofabric = self._build_gages_hydrofabric()

        else:
            self.mode = "all"
            self.gage_ids = None
            self.observations = None
            self.gages_adjacency = None
            log.info("All segments mode")
            self.hydrofabric = self._build_all_segments_hydrofabric()

    def __len__(self) -> int:
        """Returns the total number of days that we're evaluating over"""
        return len(self.dates.daily_time_range)

    def __getitem__(self, idx) -> int:
        return idx

    def collate_fn(self, *args, **kwargs) -> Hydrofabric:
        """Batching by timesteps, not gauge IDs"""
        indices = list(args[0])
        if 0 not in indices:
            prev_day = indices[0] - 1
            indices.insert(0, prev_day)

        self.dates.set_date_range(indices)
        return self.hydrofabric

    def _build_gages_hydrofabric(self) -> Hydrofabric:
        """Build hydrofabric for all gages."""
        valid_gauges_mask = np.isin(self.gage_ids, list(self.gages_adjacency.keys()))
        batch = self.gage_ids[valid_gauges_mask].tolist()

        coo, _gage_idx, gage_wb = construct_network_matrix(batch, self.gages_adjacency)

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

        wb_ids = np.array([f"wb-{_id}" for _id in compressed_hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in compressed_hf_ids])
        compressed_flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            original_col_indices = coo.col[local_gage_inflow_idx]
            compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            outflow_idx.append(compressed_col_indices)

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, divide_ids, compressed_flowpath_attr)
        )

        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=np.array(batch),
            observations=self.observations,
        )

        log.info(f"Created gages adjacency matrix of shape: {adjacency_matrix.shape}")
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

    def _build_target_catchments_hydrofabric(self) -> Hydrofabric:
        """Build hydrofabric for target catchments by finding all upstream segments."""
        all_ancestor_indices = set()
        target_node_groups = []

        for target in self.target_catchments:
            target_id = int(target.split("-")[1])

            assert target_id in self.hf_id_to_node, (
                f"{target_id} not found in Hydrofabric graph. Use a different target ID"
            )

            target_node = self.hf_id_to_node[target_id]
            ancestors = rx.ancestors(self.network_graph, target_node)
            ancestors.add(target_node)

            all_ancestor_indices.update(ancestors)
            target_node_groups.append((target_node, ancestors))

        if not all_ancestor_indices:
            raise ValueError("No valid target catchments found in hydrofabric")

        rows = self.conus_adjacency["indices_0"][:]
        cols = self.conus_adjacency["indices_1"][:]
        data = self.conus_adjacency["values"][:]

        active_set = all_ancestor_indices
        mask = np.array([r in active_set and c in active_set for r, c in zip(rows, cols, strict=False)])
        filtered_rows = rows[mask]
        filtered_cols = cols[mask]
        filtered_data = data[mask]

        coo = sparse.coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)), shape=(len(self.hf_ids), len(self.hf_ids))
        )

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

        wb_ids = np.array([f"wb-{_id}" for _id in compressed_hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in compressed_hf_ids])
        compressed_flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        outflow_idx = [np.array([i]) for i in range(compressed_size)]

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, divide_ids, compressed_flowpath_attr)
        )

        log.info(f"Created target catchments adjacency matrix of shape: {adjacency_matrix.shape}")
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
            outflow_idx=outflow_idx,
            gage_wb=None,
        )

    def _build_all_segments_hydrofabric(self) -> Hydrofabric:
        """Build hydrofabric for all segments."""
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

        log.info(f"Created all segments adjacency matrix of shape: {adjacency_matrix.shape}")
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
            outflow_idx=None,
            gage_wb=None,
        )

    def _build_common_tensors(
        self,
        csr_matrix: sparse.csr_matrix,
        divide_ids: np.ndarray,
        flowpath_attr: gpd.GeoDataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build tensors common to all modes."""
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
