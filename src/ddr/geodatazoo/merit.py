"""A file to have all train/test geospatial datasets for the MERIT Hydro dataset"""

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rustworkx as rx
import torch
import xarray as xr
from scipy import sparse

from ddr.geodatazoo.base_geodataset import BaseGeoDataset
from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.io.builders import (
    _build_network_graph,
    construct_network_matrix,
    create_hydrofabric_observations,
)
from ddr.io.readers import (
    IcechunkUSGSReader,
    build_flow_scale_tensor,
    fill_nans,
    filter_gages_by_area_threshold,
    filter_gages_by_da_valid,
    naninfmean,
    read_zarr,
)
from ddr.io.statistics import set_statistics
from ddr.validation.configs import Config
from ddr.validation.enums import Mode

log = logging.getLogger(__name__)


class Merit(BaseGeoDataset):
    """An implementation of the BaseDataset for the MERIT Hydro dataset."""

    def __init__(self, cfg: "Config") -> None:
        self.cfg = cfg
        self.dates = Dates(**self.cfg.experiment.model_dump())
        self.gage_ids: np.ndarray | None = None
        self.routing_dataclass: RoutingDataclass | None = None
        self.network_graph: rx.PyDiGraph | None = None
        self.hf_id_to_node: dict[int, int] | None = None
        self.observations: Any = None
        self.gages_adjacency: Any = None
        self.target_catchments: list[str] | None = None

        # Load attributes
        self.attribute_ds = self._load_attributes()
        self.attr_stats = set_statistics(self.cfg, self.attribute_ds)
        self.id_to_index = {comid: idx for idx, comid in enumerate(self.attribute_ds.COMID.values)}
        all_names = list(self.cfg.kan.input_var_names)
        if self.cfg.cuda_lstm is not None:
            for name in self.cfg.cuda_lstm.input_var_names:
                if name not in all_names:
                    all_names.append(name)
        self.attributes_list = all_names

        # Precompute mean/std tensors for normalization
        self.means = torch.tensor(
            [self.attr_stats[attr].iloc[2] for attr in self.attributes_list],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        self.stds = torch.tensor(
            [self.attr_stats[attr].iloc[3] for attr in self.attributes_list],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Load flowpath attributes
        _flowpath_attr = gpd.read_file(
            self.cfg.data_sources.geospatial_fabric_gpkg,
        ).set_index("COMID")
        self.flowpath_attr = _flowpath_attr[~_flowpath_attr.index.duplicated(keep="first")]

        self.phys_means = torch.tensor(
            [naninfmean(self.flowpath_attr[attr].values) for attr in ["lengthkm", "slope"]],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        # Load adjacency data
        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.merit_ids = self.conus_adjacency["order"][:]

        # Initialize based on mode
        if cfg.mode == Mode.TRAINING:
            self._init_training()
        else:
            self._init_inference()

    def _load_attributes(self) -> xr.Dataset:
        """Load attributes from NetCDF/Zarr files."""
        return xr.open_mfdataset(self.cfg.data_sources.attributes)

    def _get_attributes(
        self,
        catchment_ids: np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Fetch attributes for the given divide IDs (COMIDs)."""
        valid_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(catchment_ids):
            comid = int(divide_id)

            if comid in self.id_to_index:
                valid_indices.append(self.id_to_index[comid])
                divide_idx_mask.append(i)
            else:
                log.debug(f"COMID {comid} missing from the loaded attributes")

        assert valid_indices, "No valid COMIDs found in this batch"

        output = torch.full(
            (len(self.attributes_list), len(catchment_ids)),
            np.nan,
            device=device,
            dtype=dtype,
        )

        _ds = self.attribute_ds[self.attributes_list].isel(COMID=valid_indices).compute()
        data_tensor = torch.from_numpy(_ds.to_array(dim="COMID").values).to(device=device, dtype=dtype)

        output[:, divide_idx_mask] = data_tensor
        return fill_nans(attr=output, row_means=self.means)

    def _init_training(self) -> None:
        """Initialize dataset for training mode."""
        if self.cfg.data_sources.gages is None or self.cfg.data_sources.gages_adjacency is None:
            raise ValueError("Training requires gages and gages_adjacency to be defined")

        self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
        self.observations = self.obs_reader.read_data(dates=self.dates)
        self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])
        if "DA_VALID" in self.obs_reader.gage_dict:
            self.gage_ids, n_removed = filter_gages_by_da_valid(
                self.gage_ids,
                self.obs_reader.gage_dict,
            )
            log.info(
                f"Filtered {n_removed}/{len(self.obs_reader.gage_dict['STAID'])} gages with DA_VALID=False"
            )
        elif self.cfg.experiment.max_area_diff_sqkm is not None:
            log.warning("DA_VALID not found in gage CSV, falling back to max_area_diff_sqkm")
            self.gage_ids, n_removed = filter_gages_by_area_threshold(
                self.gage_ids,
                self.obs_reader.gage_dict,
                self.cfg.experiment.max_area_diff_sqkm,
            )
            log.info(
                f"Filtered {n_removed}/{len(self.obs_reader.gage_dict['STAID'])} gages exceeding area diff threshold "
                f"of {self.cfg.experiment.max_area_diff_sqkm} km²"
            )
        self.gages_adjacency = read_zarr(Path(self.cfg.data_sources.gages_adjacency))
        log.info(f"Training mode: routing for {len(self.gage_ids)} gauged locations")

    def _init_inference(self) -> None:
        """Initialize dataset for testing or routing mode."""
        if self.cfg.data_sources.target_catchments is not None:
            self.target_catchments = self.cfg.data_sources.target_catchments
            self.network_graph, self.hf_id_to_node, _ = _build_network_graph(self.conus_adjacency)
            log.info(f"Target catchments mode: routing upstream of {self.target_catchments}")
            self.routing_dataclass = self._build_routing_data_target_catchments()

        elif self.cfg.data_sources.gages is not None and self.cfg.data_sources.gages_adjacency is not None:
            self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
            self.observations = self.obs_reader.read_data(dates=self.dates)
            self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])
            if "DA_VALID" in self.obs_reader.gage_dict:
                self.gage_ids, n_removed = filter_gages_by_da_valid(
                    self.gage_ids,
                    self.obs_reader.gage_dict,
                )
                log.info(f"Filtered {n_removed} gages with DA_VALID=False")
            elif self.cfg.experiment.max_area_diff_sqkm is not None:
                log.warning("DA_VALID not found in gage CSV, falling back to max_area_diff_sqkm")
                self.gage_ids, n_removed = filter_gages_by_area_threshold(
                    self.gage_ids,
                    self.obs_reader.gage_dict,
                    self.cfg.experiment.max_area_diff_sqkm,
                )
                log.info(
                    f"Filtered {n_removed} gages exceeding area diff threshold "
                    f"of {self.cfg.experiment.max_area_diff_sqkm} km²"
                )
            self.gages_adjacency = read_zarr(Path(self.cfg.data_sources.gages_adjacency))
            log.info(f"Gages mode: {len(self.gage_ids)} gauged locations")
            self.routing_dataclass = self._build_routing_data_gages()

        else:
            log.info("All segments mode")
            self.routing_dataclass = self._build_routing_data_all_catchments()

    def _collate_gages(self, batch: np.ndarray) -> RoutingDataclass:
        """Route to gauge locations with observations."""
        valid_gauges_mask = np.isin(batch, list(self.gages_adjacency.keys()))
        batch = batch[valid_gauges_mask].tolist()

        coo, _gage_idx, gage_catchment = construct_network_matrix(batch, self.gages_adjacency)

        edge_indices = (
            np.unique(np.concatenate([coo.row, coo.col])) if coo.nnz > 0 else np.array([], dtype=int)
        )
        gage_indices = np.array(_gage_idx, dtype=int)
        active_indices = np.unique(np.concatenate([edge_indices, gage_indices]))
        index_mapping = {orig_idx: compressed_idx for compressed_idx, orig_idx in enumerate(active_indices)}

        if coo.nnz > 0:
            compressed_rows = np.array([index_mapping[idx] for idx in coo.row])
            compressed_cols = np.array([index_mapping[idx] for idx in coo.col])
        else:
            compressed_rows = np.array([], dtype=int)
            compressed_cols = np.array([], dtype=int)

        compressed_size = len(active_indices)
        compressed_coo = sparse.coo_matrix(
            (coo.data[: len(compressed_rows)], (compressed_rows, compressed_cols)),
            shape=(compressed_size, compressed_size),
        )
        compressed_csr = compressed_coo.tocsr()
        compressed_merit_ids = self.merit_ids[active_indices]

        compressed_flowpath_attr = self.flowpath_attr.reindex(compressed_merit_ids)

        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            original_col_indices = coo.col[local_gage_inflow_idx]
            if len(original_col_indices) > 0:
                compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            else:
                compressed_col_indices = np.array([index_mapping[int(_idx)]])
            outflow_idx.append(compressed_col_indices)

        gage_compressed_indices = [index_mapping[int(idx)] for idx in _gage_idx]
        flow_scale = build_flow_scale_tensor(
            batch=batch,
            gage_dict=self.obs_reader.gage_dict,
            gage_compressed_indices=gage_compressed_indices,
            num_segments=compressed_size,
        )

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, compressed_merit_ids, compressed_flowpath_attr)
        )

        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=np.array(batch),
            observations=self.observations,
        )

        log.info(f"Created adjacency matrix of shape: {adjacency_matrix.shape}")
        return RoutingDataclass(
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
            divide_ids=compressed_merit_ids,
            outflow_idx=outflow_idx,
            gage_catchment=gage_catchment,
            flow_scale=flow_scale,
            attribute_names=self.attributes_list,
        )

    def _build_common_tensors(
        self,
        csr_matrix: sparse.csr_matrix,
        catchment_ids: np.ndarray,
        flowpath_attr: gpd.GeoDataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Build tensors common to all collate methods."""
        adjacency_matrix = torch.sparse_csr_tensor(
            crow_indices=csr_matrix.indptr,
            col_indices=csr_matrix.indices,
            values=csr_matrix.data,
            size=csr_matrix.shape,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        spatial_attributes = self._get_attributes(
            catchment_ids=catchment_ids,
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
                torch.tensor(flowpath_attr["lengthkm"].values, dtype=torch.float32),
                row_means=self.phys_means[0],
            )
            * 1000,  # convert from km to m
            "slope": fill_nans(
                torch.tensor(flowpath_attr["slope"].values, dtype=torch.float32),
                row_means=self.phys_means[1],
            ),
        }
        flowpath_tensors["x"] = torch.full_like(
            flowpath_tensors["length"], fill_value=0.3, dtype=torch.float32
        )
        flowpath_tensors["top_width"] = torch.empty(0)
        flowpath_tensors["side_slope"] = torch.empty(0)

        return adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors

    def _build_routing_data_target_catchments(self) -> RoutingDataclass:
        """Build hydrofabric for target catchments by finding all upstream segments."""
        if self.target_catchments is None:
            raise ValueError("target_catchments must be set")
        if self.hf_id_to_node is None or self.network_graph is None:
            raise ValueError("network_graph and hf_id_to_node must be initialized")

        all_ancestor_indices: set[int] = set()

        for target in self.target_catchments:
            target_id = int(target.split("-")[1])
            assert target_id in self.hf_id_to_node, f"{target_id} not found in graph"

            target_node = self.hf_id_to_node[target_id]
            ancestors = rx.ancestors(self.network_graph, target_node)
            ancestors.add(target_node)
            all_ancestor_indices.update(ancestors)

        if not all_ancestor_indices:
            raise ValueError("No valid target catchments found")

        rows = self.conus_adjacency["indices_0"][:]
        cols = self.conus_adjacency["indices_1"][:]
        data = self.conus_adjacency["values"][:]

        mask = np.array(
            [
                r in all_ancestor_indices and c in all_ancestor_indices
                for r, c in zip(rows, cols, strict=False)
            ]
        )
        filtered_rows = rows[mask]
        filtered_cols = cols[mask]
        filtered_data = data[mask]

        coo = sparse.coo_matrix(
            (filtered_data, (filtered_rows, filtered_cols)),
            shape=(len(self.merit_ids), len(self.merit_ids)),
        )

        active_indices = np.unique(np.concatenate([coo.row, coo.col]))
        index_mapping = {orig_idx: compressed_idx for compressed_idx, orig_idx in enumerate(active_indices)}

        compressed_rows = np.array([index_mapping[idx] for idx in coo.row])
        compressed_cols = np.array([index_mapping[idx] for idx in coo.col])

        compressed_size = len(active_indices)
        compressed_coo = sparse.coo_matrix(
            (coo.data, (compressed_rows, compressed_cols)),
            shape=(compressed_size, compressed_size),
        )
        compressed_csr = compressed_coo.tocsr()
        compressed_merit_ids = self.merit_ids[active_indices]

        compressed_flowpath_attr = self.flowpath_attr.reindex(compressed_merit_ids)

        outflow_idx = [np.array([i]) for i in range(compressed_size)]

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, compressed_merit_ids, compressed_flowpath_attr)
        )

        log.info(f"Created target catchments adjacency matrix of shape: {adjacency_matrix.shape}")
        return RoutingDataclass(
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
            divide_ids=compressed_merit_ids,
            outflow_idx=outflow_idx,
            gage_catchment=None,
            attribute_names=self.attributes_list,
        )

    def _build_routing_data_all_catchments(self) -> RoutingDataclass:
        """Build hydrofabric for all segments."""
        rows = self.conus_adjacency["indices_0"][:].tolist()
        cols = self.conus_adjacency["indices_1"][:].tolist()
        _attrs: dict[str, Any] = dict(self.conus_adjacency.attrs)

        if not rows:
            raise ValueError("No coordinate-pairs found. Cannot construct a matrix")

        shape = tuple(_attrs["shape"])
        csr_matrix = sparse.coo_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=shape,
        ).tocsr()

        flowpath_attr = self.flowpath_attr.reindex(self.merit_ids)

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(csr_matrix, self.merit_ids, flowpath_attr)
        )

        log.info(f"Created all segments adjacency matrix of shape: {adjacency_matrix.shape}")
        return RoutingDataclass(
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
            divide_ids=self.merit_ids,
            outflow_idx=None,
            gage_catchment=None,
            attribute_names=self.attributes_list,
        )

    def _build_routing_data_gages(self) -> RoutingDataclass:
        """Build hydrofabric for all gages."""
        if self.gage_ids is None:
            raise ValueError("gage_ids must be set for gages mode")

        valid_gauges_mask = np.isin(self.gage_ids, list(self.gages_adjacency.keys()))
        batch = self.gage_ids[valid_gauges_mask].tolist()

        coo, _gage_idx, gage_catchment = construct_network_matrix(batch, self.gages_adjacency)

        edge_indices = (
            np.unique(np.concatenate([coo.row, coo.col])) if coo.nnz > 0 else np.array([], dtype=int)
        )
        gage_indices = np.array(_gage_idx, dtype=int)
        active_indices = np.unique(np.concatenate([edge_indices, gage_indices]))
        index_mapping = {orig_idx: compressed_idx for compressed_idx, orig_idx in enumerate(active_indices)}

        if coo.nnz > 0:
            compressed_rows = np.array([index_mapping[idx] for idx in coo.row])
            compressed_cols = np.array([index_mapping[idx] for idx in coo.col])
        else:
            compressed_rows = np.array([], dtype=int)
            compressed_cols = np.array([], dtype=int)

        compressed_size = len(active_indices)
        compressed_coo = sparse.coo_matrix(
            (coo.data[: len(compressed_rows)], (compressed_rows, compressed_cols)),
            shape=(compressed_size, compressed_size),
        )
        compressed_csr = compressed_coo.tocsr()
        compressed_merit_ids = self.merit_ids[active_indices]
        compressed_flowpath_attr = self.flowpath_attr.reindex(compressed_merit_ids)

        outflow_idx = []
        for _idx in _gage_idx:
            mask = np.isin(coo.row, _idx)
            local_gage_inflow_idx = np.where(mask)[0]
            original_col_indices = coo.col[local_gage_inflow_idx]
            if len(original_col_indices) > 0:
                compressed_col_indices = np.array([index_mapping[idx] for idx in original_col_indices])
            else:
                compressed_col_indices = np.array([index_mapping[int(_idx)]])
            outflow_idx.append(compressed_col_indices)

        gage_compressed_indices = [index_mapping[int(idx)] for idx in _gage_idx]
        flow_scale = build_flow_scale_tensor(
            batch=batch,
            gage_dict=self.obs_reader.gage_dict,
            gage_compressed_indices=gage_compressed_indices,
            num_segments=compressed_size,
        )

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, compressed_merit_ids, compressed_flowpath_attr)
        )

        hydrofabric_observations = create_hydrofabric_observations(
            dates=self.dates,
            gage_ids=np.array(batch),
            observations=self.observations,
        )

        log.info(f"Created gages adjacency matrix of shape: {adjacency_matrix.shape}")
        return RoutingDataclass(
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
            divide_ids=compressed_merit_ids,
            outflow_idx=outflow_idx,
            gage_catchment=gage_catchment,
            flow_scale=flow_scale,
            attribute_names=self.attributes_list,
        )
