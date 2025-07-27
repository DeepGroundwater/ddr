import logging

import numpy as np
import torch

from ddr.dataset.utils import fill_nans, read_ic
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class AttributesReader(torch.nn.Module):
    """A class to read attributes from a local zarr store or icechunk repo"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attributes_list = list(
            self.cfg.kan.input_var_names
        )  # Have to cast to list for this to work with xarray
        self.ds = read_ic(self.cfg.data_sources.attributes, region=self.cfg.s3_region)

        # Index Lookup Dictionary
        self.divide_id_to_index = {divide_id: idx for idx, divide_id in enumerate(self.ds.divide_id.values)}

    def forward(self, **kwargs) -> torch.Tensor:
        """The forward function of the module for generating attributes

        Returnsq
        -------
        torch.Tensor
            attributes for the given divides in the shape (n_attributes, n_divides)

        Raises
        ------
        IndexError
            The basin you're searching for is not in the sample
        """
        divide_ids = kwargs["divide_ids"]
        attr_means = kwargs["attr_means"]
        device = kwargs.get("device", "cpu")  # defaulting to a CPU tensor
        dtype = kwargs.get("dtype", torch.float32)  # defaulting to float32

        valid_divide_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(divide_ids):
            if divide_id in self.divide_id_to_index:
                valid_divide_indices.append(self.divide_id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                log.info(f"{divide_id} missing from the loaded attributes")

        assert len(valid_divide_indices) != 0, "No valid divide IDs found in this batch. Throwing error"

        output = torch.full((len(self.attributes_list), len(divide_ids)), np.nan, device=device, dtype=dtype)

        _ds = self.ds[self.attributes_list].isel(divide_id=valid_divide_indices).compute()
        data_array = _ds.to_array(dim="divide_id").values
        data_tensor = torch.from_numpy(data_array).to(device=device, dtype=dtype)
        output[:, divide_idx_mask] = data_tensor

        output = fill_nans(
            attr=output, row_means=attr_means
        )  # Filling missing attributes with the mean values
        return output
