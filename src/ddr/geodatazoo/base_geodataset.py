from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd
import numpy as np
import torch
import xarray as xr
from scipy import sparse
from torch.utils.data import Dataset

from ddr.geodatazoo.dataclasses import Dates, RoutingDataclass
from ddr.validation.configs import Config
from ddr.validation.enums import Mode


class BaseGeoDataset(Dataset, ABC):
    """Abstract base class for PyTorch datasets."""

    cfg: Config
    dates: Dates
    gage_ids: np.ndarray | None
    routing_dataclass: RoutingDataclass | None
    attribute_ds: xr.Dataset

    def __len__(self) -> int:
        """A shared __len__ for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            assert self.gage_ids is not None, "No gage IDs found, cannot batch"
            return len(self.gage_ids)
        return len(self.dates.daily_time_range)

    def __getitem__(self, idx: int) -> str | int:
        """A shared __getitem__ for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            assert self.gage_ids is not None, "No gage IDs found, cannot batch"
            return str(self.gage_ids[idx])
        return idx

    def collate_fn(self, *args: Any, **kwargs: Any) -> RoutingDataclass:
        """A shared collate_fn for all Datasets"""
        if self.cfg.mode == Mode.TRAINING:
            self.dates.calculate_time_period()
            return self._collate_gages(np.array(args[0]))

        assert self.routing_dataclass is not None, "No RoutingDataclass, cannot batch"
        indices = list(args[0])
        if 0 not in indices:
            indices.insert(0, indices[0] - 1)
        self.dates.set_date_range(np.array(indices))
        return self.routing_dataclass

    @abstractmethod
    def _load_attributes(self) -> xr.Dataset:
        """Load the catchment attribute dataset into memory.

        Returns
        -------
        xr.Dataset
            Dataset containing all input variables listed in
            ``cfg.kan.input_var_names``.  Must expose a spatial coordinate
            (e.g. ``COMID`` for MERIT, ``divide_id`` for Lynker) so that
            individual catchments can be selected by index.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_attributes(
        self,
        catchment_ids: np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Fetch raw (un-normalised) attributes for a set of catchment IDs.

        Parameters
        ----------
        catchment_ids : np.ndarray
            Ordered array of N catchment identifiers whose attributes are
            required.  IDs not present in the attribute store should be filled
            with the per-attribute mean rather than left as NaN.
        device : str or torch.device, optional
            Target device for the returned tensor.
        dtype : torch.dtype, optional
            Target dtype for the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape ``(num_attributes, N)`` — attributes are the *row* axis,
            catchments are the *column* axis.  This is the *un-normalised*
            form; ``_build_common_tensors`` handles z-score normalisation and
            transposition before the tensor is passed to the KAN.
        """
        raise NotImplementedError

    @abstractmethod
    def _collate_gages(self, batch: np.ndarray) -> RoutingDataclass:
        """Build a fully-populated RoutingDataclass for a mini-batch of gages.

        Called by the shared ``collate_fn`` during training.
        ``self.dates.calculate_time_period()`` has already been called before
        this method is invoked, so ``self.dates`` reflects the current random
        time window.

        Parameters
        ----------
        batch : np.ndarray
            Array of gage IDs (same dtype as ``self.gage_ids``) selected by
            the DataLoader sampler for this mini-batch.

        Returns
        -------
        RoutingDataclass
            All fields must be populated for training:

            * ``adjacency_matrix`` — sparse CSR tensor ``(N, N)`` on
              ``cfg.device``.
            * ``spatial_attributes`` — raw attributes ``(num_attrs, N)``.
            * ``normalized_spatial_attributes`` — z-score normalised and
              *transposed* to ``(N, num_attrs)`` — fed directly into the KAN.
            * ``length``, ``slope``, ``x``, ``top_width``, ``side_slope`` —
              per-reach physical tensors, each ``(N,)``.  Set unused tensors
              to ``torch.empty(0)`` (e.g. ``top_width`` for MERIT).
            * ``dates`` — reference to ``self.dates``.
            * ``observations`` — ``xr.Dataset`` with dims ``(gage_id, time)``
              aligned to the gages in ``batch`` and the current time window.
            * ``divide_ids`` — ``np.ndarray`` of N catchment IDs in network
              order (compressed index space).
            * ``outflow_idx`` — ragged ``list[np.ndarray]`` where element *i*
              holds the compressed column indices of segments draining into
              gage *i*.
            * ``gage_catchment`` — ``list[str]`` of gage IDs that were
              successfully matched to network segments.
            * ``flow_scale`` — ``torch.Tensor`` ``(N,)`` scaling factors that
              account for partial-area gage coverage.

        Notes
        -----
        Gages absent from ``self.gages_adjacency`` should be silently dropped
        via a validity mask before constructing the subgraph.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_training(self) -> None:
        """Initialise the dataset for training mode.

        Called from ``__init__`` when ``cfg.mode == Mode.TRAINING``.  Must set
        the following instance attributes that are consumed by the shared
        ``collate_fn`` and ``_collate_gages``:

        * ``self.gage_ids`` (``np.ndarray``) — gage IDs returned by
          ``__getitem__`` and sampled by the DataLoader.
        * ``self.observations`` — loaded observation dataset (typically the
          return value of ``IcechunkUSGSReader.read_data``).
        * ``self.gages_adjacency`` — pre-built per-gage adjacency structure
          (loaded from ``cfg.data_sources.gages_adjacency``).
        * ``self.obs_reader`` — the observation reader instance, kept as an
          attribute so ``_collate_gages`` can access ``gage_dict`` for area
          scaling.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_inference(self) -> None:
        """Initialise the dataset for testing or routing mode.

        Called from ``__init__`` when ``cfg.mode != Mode.TRAINING``.  Must set
        ``self.routing_dataclass`` to a fully-populated ``RoutingDataclass``
        covering the entire inference domain.  The shared ``collate_fn`` reads
        directly from this attribute during inference iteration.

        Three sub-cases must be handled, checked in priority order:

        1. **Target catchments** (``cfg.data_sources.target_catchments`` is
           set) — traverse the network graph upstream of each target and build
           the subgraph.
        2. **Gages mode** (``cfg.data_sources.gages`` and
           ``cfg.data_sources.gages_adjacency`` are both set) — build the
           union of subgraphs draining to each gage.
        3. **All segments** (fallback) — use the full CONUS adjacency matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def _build_common_tensors(
        self,
        csr_matrix: sparse.csr_matrix,
        catchment_ids: np.ndarray,
        flowpath_attr: gpd.GeoDataFrame,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Convert a compressed subgraph into the tensors needed by the routing engine.

        This is the core tensor-assembly step shared by all collate helpers
        (``_collate_gages``, ``_build_routing_data_gages``, etc.).

        Parameters
        ----------
        csr_matrix : sparse.csr_matrix
            Compressed-sparse-row adjacency matrix of shape ``(N, N)`` for the
            subgraph, where N is the number of active segments.
        catchment_ids : np.ndarray
            Array of N catchment identifiers in compressed-index order
            (i.e. ``catchment_ids[i]`` is the dataset ID for segment *i*).
        flowpath_attr : gpd.GeoDataFrame
            GeoDataFrame indexed by catchment ID containing at minimum the
            physical reach attributes needed to populate the routing tensors
            (length, slope, and optionally top-width, side-slope, and
            Muskingum *x*).

        Returns
        -------
        adjacency_matrix : torch.Tensor
            Sparse CSR tensor, shape ``(N, N)``, ``float32``, on
            ``cfg.device``.  Encodes the directed flow network used by the
            MC routing engine.
        spatial_attributes : torch.Tensor
            Shape ``(num_attributes, N)``, ``float32``.  Raw (un-normalised)
            catchment attributes with any remaining NaNs replaced by the
            per-attribute batch mean.
        normalized_spatial_attributes : torch.Tensor
            Shape ``(N, num_attributes)``, ``float32``.  Z-score normalised
            and **transposed** relative to ``spatial_attributes`` — this is
            fed directly as input to the KAN.
        flowpath_tensors : dict[str, torch.Tensor]
            Physical reach properties, each ``(N,)`` ``float32``:

            * ``"length"`` — reach length in **metres**.
            * ``"slope"`` — dimensionless channel slope.
            * ``"x"`` — Muskingum weighting factor (0 ≤ x ≤ 0.5).
            * ``"top_width"`` — bankfull top width in metres; set to
              ``torch.empty(0)`` when using Leopold & Maddock geometry
              (MERIT).
            * ``"side_slope"`` — trapezoid side slope *z* (H:V); set to
              ``torch.empty(0)`` when using Leopold & Maddock geometry
              (MERIT).

        Notes
        -----
        The transposition of ``normalized_spatial_attributes`` relative to
        ``spatial_attributes`` is intentional: the KAN expects shape
        ``(N, num_features)`` while the normalisation arithmetic is more
        convenient in ``(num_features, N)`` form.
        """
        raise NotImplementedError
