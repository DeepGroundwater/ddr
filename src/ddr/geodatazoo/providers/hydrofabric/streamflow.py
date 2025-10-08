"""NextGen streamflow provider."""

import logging

import torch

from ddr.dataset.Dates import Dates
from ddr.dataset.streamflow import StreamflowReader
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class NextGenStreamflowProvider:
    """Provider for NextGen streamflow/runoff data.

    This reads streamflow from icechunk stores that use NextGen
    catchment IDs (cat-XXX format).

    The icechunk store must have:
    - Dimensions: divide_id, time
    - Variable: Qr (runoff in m³/s)
    - divide_id values in cat-XXX format

    Parameters
    ----------
    cfg : Config
        Configuration with:
        - data_sources.streamflow: Path to icechunk store
        - device: Device for tensors

    Examples
    --------
    >>> provider = NextGenStreamflowProvider(cfg)
    >>> streamflow = provider.get_streamflow(catchment_ids=["cat-1", "cat-2"], dates=dates_object)
    >>> print(f"Shape: {streamflow.shape}")  # (n_timesteps, n_catchments)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.reader = StreamflowReader(cfg)
        log.info("Initialized NextGenStreamflowProvider")

    def get_streamflow(self, catchment_ids: list[str], dates: Dates) -> torch.Tensor:
        """Get streamflow for NextGen catchments.

        Parameters
        ----------
        catchment_ids : list[str]
            NextGen catchment IDs (cat-XXX format)
        dates : Dates
            Time period for streamflow data

        Returns
        -------
        torch.Tensor
            Streamflow data with shape (n_timesteps, n_catchments)
        """
        # Create temporary hydrofabric for reader interface
        # (StreamflowReader expects Hydrofabric object)
        import numpy as np

        from ddr.dataset.utils import Hydrofabric

        temp_hydrofabric = Hydrofabric(
            divide_ids=np.array(catchment_ids),
            dates=dates,
            # Other fields not needed by StreamflowReader
            spatial_attributes=None,
            length=None,
            slope=None,
            side_slope=None,
            top_width=None,
            x=None,
            adjacency_matrix=None,
            normalized_spatial_attributes=None,
            observations=None,
            gage_idx=None,
            gage_wb=None,
        )

        return self.reader(
            hydrofabric=temp_hydrofabric,
            device=self.cfg.device,
            dtype=torch.float32,
        )
