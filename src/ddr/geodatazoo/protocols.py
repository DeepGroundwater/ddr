"""Protocol definitions for dataset providers. These define the interfaces geospatial data objects must implement."""

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch.utils.data import Dataset as TorchDataset

from ddr.geodatazoo.Dates import Dates
from ddr.validation.validate_configs import Config


class BaseGeoDataset(TorchDataset, ABC):
    """Lays out the base implementation of a geospatial routing dataset"""

    def __init__(self, cfg: Config, is_train: bool):
        self.cfg = cfg
        self.is_train = is_train
        self.dates = Dates(**self.cfg.experiment.model_dump())

        self.observation_provider = None

        # Load data via abstract methods
        self._load_network_data()
        self._load_attribute_data()
        self._load_observation_data()

        # Setup statistics for normalization
        self._setup_statistics()

        # For test mode, build network once
        if not is_train:
            self._setup_test_network()
        else:
            self.hydrofabric = None

    # ========================================================================
    # Abstract methods - subclasses MUST implement these
    # ========================================================================

    @abstractmethod
    def _load_network_data(self):
        """Load network topology and channel parameters.

        Subclasses should set self.network_provider to an object implementing
        the NetworkProvider protocol.

        Examples
        --------
        >>> def _load_network_data(self):
        ...     from ddr.providers.nextgen import NextGenNetworkProvider
        ...
        ...     self.network_provider = NextGenNetworkProvider(
        ...         gpkg_path=self.cfg.data_sources.hydrofabric_gpkg,
        ...         adjacency_path=self.cfg.data_sources.conus_adjacency,
        ...         gages_adjacency_path=self.cfg.data_sources.gages_adjacency,
        ...     )
        """
        raise NotImplementedError

    @abstractmethod
    def _load_attribute_data(self):
        """Load catchment attributes.

        Subclasses should set self.attribute_provider to an object implementing
        the AttributeProvider protocol.

        Examples
        --------
        >>> def _load_attribute_data(self):
        ...     from ddr.dataset.attributes import AttributesReader
        ...
        ...     self.attribute_provider = AttributesReader(cfg=self.cfg)
        """
        raise NotImplementedError

    @abstractmethod
    def _load_observation_data(self):
        """Load observation data.

        Subclasses should set self.observation_provider to an object implementing
        the ObservationProvider protocol, and populate self.gauge_ids.

        Examples
        --------
        >>> def _load_observation_data(self):
        ...     from ddr.dataset.observations import IcechunkUSGSReader
        ...
        ...     self.observation_provider = IcechunkUSGSReader(cfg=self.cfg)
        ...     self.gauge_ids = self.observation_provider.get_gauge_ids()
        """
        raise NotImplementedError

    # ========================================================================
    # Common methods - implemented in base class
    # ========================================================================

    def _setup_statistics(self):
        """Setup normalization statistics from attribute provider."""
        from ddr.dataset.statistics import set_statistics

        # Get the underlying dataset from attribute provider if available
        attr_ds = getattr(self.attribute_provider, "ds", None)
        if attr_ds is not None:
            self.attr_stats = set_statistics(self.cfg, attr_ds)

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
        else:
            self.means = None
            self.stds = None

    def _setup_test_network(self):
        """Build network once for all gauges (test mode only)."""
        if not hasattr(self, "gauge_ids"):
            raise ValueError("gauge_ids must be set before calling _setup_test_network")

        # Get all reach IDs for all gauges
        all_reach_ids = []
        valid_gauge_ids = []

        for gid in self.gauge_ids:
            reach_id = f"wb-{gid}"
            try:
                upstream = self.network_provider.find_upstream_reaches(reach_id)
                if len(upstream) > 0:
                    all_reach_ids.extend(upstream)
                    valid_gauge_ids.append(gid)
            except Exception as e:
                log.info(f"Cannot find upstream reaches for gauge {gid}: {e}")
                continue

        # Remove duplicates
        reach_ids = list(dict.fromkeys(all_reach_ids))

        # Build hydrofabric
        self.hydrofabric = self._build_hydrofabric(reach_ids, valid_gauge_ids)

    def _build_hydrofabric(self, reach_ids: list[str], gauge_ids: list[str] = None) -> Hydrofabric:
        """Build hydrofabric object from reach IDs.

        Parameters
        ----------
        reach_ids : list[str]
            List of reach IDs to include in network
        gauge_ids : list[str], optional
            List of gauge IDs for observations

        Returns
        -------
        Hydrofabric
            Complete hydrofabric object ready for model input
        """
        # Get network topology
        adjacency = self.network_provider.get_topology(reach_ids)

        # Get catchment IDs
        catchment_ids = self.network_provider.get_catchment_ids(reach_ids)

        # Get channel parameters
        channel_params = self.network_provider.get_channel_parameters(reach_ids)

        # Get spatial attributes
        raw_attrs, normalized_attrs = self.attribute_provider.get_attributes(
            catchment_ids, self.cfg.kan.input_var_names
        )

        # Fill NaNs in raw attributes
        for r in range(raw_attrs.shape[0]):
            row_means = torch.nanmean(raw_attrs[r])
            nan_mask = torch.isnan(raw_attrs[r])
            raw_attrs[r, nan_mask] = row_means

        # Create sparse adjacency tensor
        adjacency_matrix = torch.sparse_csr_tensor(
            crow_indices=adjacency.indptr,
            col_indices=adjacency.indices,
            values=adjacency.data,
            size=adjacency.shape,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        # Convert channel parameters to tensors with NaN filling
        phys_means = torch.tensor(
            [
                naninfmean(channel_params["length"]),
                naninfmean(channel_params["slope"]),
                naninfmean(channel_params["top_width"]),
                naninfmean(channel_params["side_slope"]),
                naninfmean(channel_params["x"]),
            ],
            device=self.cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        length = fill_nans(
            torch.tensor(channel_params["length"], dtype=torch.float32),
            row_means=phys_means[0],
        )
        slope = fill_nans(
            torch.tensor(channel_params["slope"], dtype=torch.float32),
            row_means=phys_means[1],
        )
        top_width = fill_nans(
            torch.tensor(channel_params["top_width"], dtype=torch.float32),
            row_means=phys_means[2],
        )
        side_slope = fill_nans(
            torch.tensor(channel_params["side_slope"], dtype=torch.float32),
            row_means=phys_means[3],
        )
        x = fill_nans(
            torch.tensor(channel_params["x"], dtype=torch.float32),
            row_means=phys_means[4],
        )

        # Handle observations
        observations = None
        gage_idx = None
        gage_wb = None

        if self.observation_provider and gauge_ids:
            observations = self.observation_provider.get_observations(self.dates)

            # Map gauge IDs to reach indices
            gage_idx = []
            gage_wb = []
            for gid in gauge_ids:
                wb_id = f"wb-{gid}"
                if wb_id in reach_ids:
                    idx = reach_ids.index(wb_id)
                    gage_idx.append(np.array([idx]))
                    gage_wb.append(wb_id)

        return Hydrofabric(
            spatial_attributes=raw_attrs,
            length=length,
            slope=slope,
            side_slope=side_slope,
            top_width=top_width,
            x=x,
            dates=self.dates,
            adjacency_matrix=adjacency_matrix,
            normalized_spatial_attributes=normalized_attrs,
            observations=observations,
            divide_ids=np.array(catchment_ids),
            gage_idx=gage_idx,
            gage_wb=gage_wb,
        )

    # ========================================================================
    # PyTorch Dataset interface
    # ========================================================================

    def __len__(self) -> int:
        """Return number of samples based on training mode."""
        if self.is_train:
            return len(self.gauge_ids)
        else:
            return len(self.dates.daily_time_range)

    def __getitem__(self, idx: int) -> str | int:
        """Get item at index.

        Returns gauge ID if training, timestep index if testing.
        """
        if self.is_train:
            return self.gauge_ids[idx]
        else:
            return idx

    def collate_fn(self, batch: list) -> Hydrofabric:
        """Collate batch into Hydrofabric object.

        Parameters
        ----------
        batch : list
            List of gauge IDs (if is_train=True) or timestep indices (if is_train=False)

        Returns
        -------
        Hydrofabric
            Hydrofabric object ready for model input
        """
        if self.is_train:
            return self._collate_train(batch)
        else:
            return self._collate_test(batch)

    def _collate_train(self, gauge_ids: list[str]) -> Hydrofabric:
        """Collate training batch (sample by gauge).

        Parameters
        ----------
        gauge_ids : list[str]
            Batch of gauge IDs

        Returns
        -------
        Hydrofabric
            Hydrofabric for these gauges with random time period
        """
        # Calculate random time period if rho is set
        self.dates.calculate_time_period()

        # Find all upstream reaches for these gauges
        all_reach_ids = []
        valid_gauge_ids = []

        for gid in gauge_ids:
            reach_id = f"wb-{gid}"
            try:
                upstream = self.network_provider.find_upstream_reaches(reach_id)
                if len(upstream) > 0:
                    all_reach_ids.extend(upstream)
                    valid_gauge_ids.append(gid)
            except Exception as e:
                log.info(f"Cannot find upstream reaches for gauge {gid}: {e}")
                continue

        # Remove duplicates while preserving order
        reach_ids = list(dict.fromkeys(all_reach_ids))

        # Build and return hydrofabric
        return self._build_hydrofabric(reach_ids, valid_gauge_ids)

    def _collate_test(self, time_indices: list[int]) -> Hydrofabric:
        """Collate test batch (sample by time).

        Parameters
        ----------
        time_indices : list[int]
            Batch of timestep indices

        Returns
        -------
        Hydrofabric
            Pre-built hydrofabric with updated dates
        """
        # Add previous day for interpolation if needed
        indices = list(time_indices)
        if 0 not in indices and len(indices) > 0:
            prev_day = indices[0] - 1
            if prev_day >= 0:
                indices.insert(0, prev_day)

        # Update dates
        self.dates.set_date_range(np.array(indices))

        # Return pre-built hydrofabric (network doesn't change)
        return self.hydrofabric


class AttributeProvider(Protocol):
    """Protocol for catchment attribute providers."""

    def get_attributes(
        self, catchment_ids: list[str], attribute_names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_attributes, normalized_attributes)."""
        ...


class StreamflowProvider(Protocol):
    """Protocol for streamflow providers."""

    def get_streamflow(self, catchment_ids: list[str], dates: Dates) -> torch.Tensor:
        """Return streamflow (num_timesteps, num_catchments)."""
        ...
