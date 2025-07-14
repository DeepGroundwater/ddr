import logging

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from ddr.dataset.utils import read_ic

log = logging.getLogger(__name__)


class AttributesReader(torch.nn.Module):
    """A class to read attributes from a local zarr store or icechunk repo"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.attributes_list = list(
            self.cfg.kan.input_var_names
        )  # Have to cast to list for this to work with xarray
        self.ds = read_ic(self.cfg.data_sources.attributes, region=self.cfg.s3_region)

    def forward(self, **kwargs) -> xr.Dataset:
        """The forward function of the module for generating attributes

        Returns
        -------
        xr.Dataset
            attributes for the given divides

        Raises
        ------
        IndexError
            The basin you're searching for is not in the sample
        """
        divide_ids = kwargs["divide_ids"]
        divide_indices = np.where(np.isin(self.ds.divide_id.values, divide_ids))[0]
        try:
            _ds = self.ds[self.attributes_list].isel(divide_id=divide_indices)
        except IndexError as e:
            msg = "index out of bounds. This means you're trying to find a basin that there is no data for."
            log.exception(msg=msg)
            raise IndexError(msg) from e

        attributes_data = _ds.compute()
        return attributes_data
