import logging

import numpy as np
import torch
from omegaconf import DictConfig

from ddr.dataset.utils import read_ic

log = logging.getLogger(__name__)


class AttributesReader(torch.nn.Module):
    """A class to read streamflow from a local zarr store or icechunk repo"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.attributes_list = self.cfg.params.attributes
        self.ds = read_ic(self.cfg.data_sources.attributes, region=self.cfg.s3_region)

    def forward(self, **kwargs) -> dict[str, np.ndarray]:
        """The forward function of the module for generating streamflow values

        Returns
        -------
        dict[str, np.ndarray]
            streamflow predictions for the given timesteps and divides

        Raises
        ------
        IndexError
            The basin you're searching for is not in the sample
        """
        hydrofabric = kwargs["hydrofabric"]
        divide_indices = np.where(np.isin(self.ds.divide_id.values, hydrofabric.divide_ids))[0]
        try:
            _ds = self.ds.isel(divide_id=divide_indices)[self.attributes]
        except IndexError as e:
            msg = "index out of bounds. This means you're trying to find a basin that there is no data for."
            log.exception(msg=msg)
            raise IndexError(msg) from e

        attributes_data = _ds.compute().values.astype(np.float32)
        return {"attributes": attributes_data}
