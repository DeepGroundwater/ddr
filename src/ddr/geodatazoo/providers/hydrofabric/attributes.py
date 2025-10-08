"""NextGen attribute provider."""

import logging

import torch

from ddr.dataset.attributes import AttributesReader
from ddr.dataset.statistics import set_statistics
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class NextGenAttributeProvider:
    """Provider for NextGen catchment attributes.

    This reads attributes from icechunk stores that use NextGen
    catchment IDs (cat-XXX format).

    The icechunk store must have:
    - Dimension: divide_id (with cat-XXX format IDs)
    - Variables: Specified in cfg.kan.input_var_names

    Parameters
    ----------
    cfg : Config
        Configuration with:
        - data_sources.attributes: Path to icechunk store
        - kan.input_var_names: List of attribute names to load
        - device: Device for tensors

    Examples
    --------
    >>> provider = NextGenAttributeProvider(cfg)
    >>> raw, normalized = provider.get_attributes(
    ...     catchment_ids=["cat-1", "cat-2"], attribute_names=["elevation", "slope"]
    ... )
    >>> print(f"Raw shape: {raw.shape}")  # (2, 2) - (n_attrs, n_catchments)
    >>> print(f"Normalized shape: {normalized.shape}")  # (2, 2) transposed
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Initialize icechunk reader
        self.reader = AttributesReader(cfg)

        # Compute statistics for normalization
        self.stats = set_statistics(cfg, self.reader.ds)

        # Pre-compute normalization parameters
        self.means = torch.tensor(
            [self.stats[attr].iloc[2] for attr in cfg.kan.input_var_names],
            device=cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        self.stds = torch.tensor(
            [self.stats[attr].iloc[3] for attr in cfg.kan.input_var_names],
            device=cfg.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        log.info(f"Initialized NextGenAttributeProvider with {len(cfg.kan.input_var_names)} attributes")

    def get_attributes(
        self, catchment_ids: list[str], attribute_names: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get attributes for NextGen catchments.

        Parameters
        ----------
        catchment_ids : list[str]
            NextGen catchment IDs (cat-XXX format)
        attribute_names : list[str]
            Names of attributes to retrieve (must match cfg.kan.input_var_names)

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (raw_attributes, normalized_attributes)
            - raw: shape (n_attributes, n_catchments)
            - normalized: shape (n_catchments, n_attributes) - transposed for NN input
        """
        # Read from icechunk
        raw = self.reader(
            divide_ids=catchment_ids,
            attr_means=self.means,
            device=self.cfg.device,
            dtype=torch.float32,
        )

        # Fill NaNs with row means
        for r in range(raw.shape[0]):
            row_mean = torch.nanmean(raw[r])
            nan_mask = torch.isnan(raw[r])
            raw[r, nan_mask] = row_mean

        # Normalize: (x - mean) / std
        normalized = (raw - self.means) / self.stds

        # Transpose for NN input: (n_catchments, n_attributes)
        return raw, normalized.T
