"""A file to have all train/test geospatial datasets for the NOAA-OWP/Lynker Hydrofabric v2.2"""

import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rustworkx as rx
import torch
from scipy import sparse

from ddr.geodatazoo.base_geodataset import BaseGeoDataset
from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.io.builders import (
    _build_network_graph,
    construct_network_matrix,
    create_hydrofabric_observations,
)
from ddr.io.readers import IcechunkUSGSReader, fill_nans, naninfmean, read_ic, read_zarr
from ddr.io.statistics import set_statistics
from ddr.validation.configs import Config
from ddr.validation.enums import Mode

log = logging.getLogger(__name__)


class LynkerHydrofabric(BaseGeoDataset):
    """An implementation of the BaseGeoDataset for the LynkerHydrofabric."""

    def __init__(self, cfg: Config) -> None:
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
        self.id_to_index = {
            divide_id: idx for idx, divide_id in enumerate(self.attribute_ds.divide_id.values)
        }
        self.attributes_list = list(self.cfg.kan.input_var_names)

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
            self.cfg.data_sources.geospatial_fabric_gpkg, layer="flowpath-attributes-ml"
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

        # Load adjacency data
        self.conus_adjacency = read_zarr(Path(cfg.data_sources.conus_adjacency))
        self.hf_ids = self.conus_adjacency["order"][:]

        if cfg.mode == Mode.TRAINING:
            self._init_training()
        else:
            self._init_inference()

    def _load_attributes(self) -> Any:
        """Load attributes from icechunk repository."""
        return read_ic(self.cfg.data_sources.attributes, region=self.cfg.s3_region)

    def _get_attributes(
        self,
        catchment_ids: np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Fetch attributes for the given divide IDs."""
        valid_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(catchment_ids):
            if divide_id in self.id_to_index:
                valid_indices.append(self.id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                log.debug(f"{divide_id} missing from the loaded attributes")

        assert valid_indices, "No valid divide IDs found in this batch"

        output = torch.full(
            (len(self.attributes_list), len(catchment_ids)),
            np.nan,
            device=device,
            dtype=dtype,
        )

        _ds = self.attribute_ds[self.attributes_list].isel(divide_id=valid_indices).compute()
        data_tensor = torch.from_numpy(_ds.to_array(dim="divide_id").values).to(device=device, dtype=dtype)

        output[:, divide_idx_mask] = data_tensor
        return fill_nans(attr=output, row_means=self.means)

    def _init_training(self) -> None:
        """Initialize dataset for training mode."""
        if self.cfg.data_sources.gages is None or self.cfg.data_sources.gages_adjacency is None:
            raise ValueError("Training requires gages and gages_adjacency to be defined")

        self.obs_reader = IcechunkUSGSReader(cfg=self.cfg)
        self.observations = self.obs_reader.read_data(dates=self.dates)
        self.gage_ids = np.array([str(_id.zfill(8)) for _id in self.obs_reader.gage_dict["STAID"]])
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

        assert (
            np.array(
                [
                    _id.split("-")[1]
                    for _id in compressed_flowpath_attr.iloc[np.concatenate(outflow_idx)]["to"]
                    .drop_duplicates(keep="first")
                    .values
                ]
            )
            == np.array([_id.split("-")[1] for _id in gage_catchment])
        ).all(), "Gage WB don't match up with indices"

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, divide_ids, compressed_flowpath_attr)
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
            divide_ids=divide_ids,
            outflow_idx=outflow_idx,
            gage_catchment=gage_catchment,
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

    def _build_routing_data_target_catchments(self) -> RoutingDataclass:
        """Build hydrofabric for target catchments by finding all upstream segments."""
        if self.target_catchments is None:
            raise ValueError("target_catchments must be set")
        if self.hf_id_to_node is None or self.network_graph is None:
            raise ValueError("network_graph and hf_id_to_node must be initialized")

        all_ancestor_indices: set[int] = set()

        for target in self.target_catchments:
            target_id = int(target.split("-")[1])
            assert target_id in self.hf_id_to_node, f"{target_id} not found in Hydrofabric graph"

            target_node = self.hf_id_to_node[target_id]
            ancestors = rx.ancestors(self.network_graph, target_node)
            ancestors.add(target_node)
            all_ancestor_indices.update(ancestors)

        if not all_ancestor_indices:
            raise ValueError("No valid target catchments found in hydrofabric")

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
            shape=(len(self.hf_ids), len(self.hf_ids)),
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
        compressed_hf_ids = self.hf_ids[active_indices]

        wb_ids = np.array([f"wb-{_id}" for _id in compressed_hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in compressed_hf_ids])
        compressed_flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        outflow_idx = [np.array([i]) for i in range(compressed_size)]

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(compressed_csr, divide_ids, compressed_flowpath_attr)
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
            divide_ids=divide_ids,
            outflow_idx=outflow_idx,
            gage_catchment=None,
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

        wb_ids = np.array([f"wb-{_id}" for _id in self.hf_ids])
        divide_ids = np.array([f"cat-{_id}" for _id in self.hf_ids])
        flowpath_attr = self.flowpath_attr.reindex(wb_ids)

        adjacency_matrix, spatial_attributes, normalized_spatial_attributes, flowpath_tensors = (
            self._build_common_tensors(csr_matrix, divide_ids, flowpath_attr)
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
            divide_ids=divide_ids,
            outflow_idx=None,
            gage_catchment=None,
        )

    def _build_routing_data_gages(self) -> RoutingDataclass:
        """Build hydrofabric for all gages."""
        if self.gage_ids is None:
            raise ValueError("gage_ids must be set for gages mode")

        valid_gauges_mask = np.isin(self.gage_ids, list(self.gages_adjacency.keys()))
        batch = self.gage_ids[valid_gauges_mask].tolist()

        coo, _gage_idx, gage_catchment = construct_network_matrix(batch, self.gages_adjacency)

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
            divide_ids=divide_ids,
            outflow_idx=outflow_idx,
            gage_catchment=gage_catchment,
        )
