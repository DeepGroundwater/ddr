import logging

import numpy as np
import torch

from ddr.dataset.utils import read_ic
from ddr.validation.validate_configs import Config

log = logging.getLogger(__name__)


class StreamflowReader(torch.nn.Module):
    """A class to read streamflow from a local zarr store or icechunk repo"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ds = read_ic(self.cfg.data_sources.streamflow, region=self.cfg.s3_region)
        # Index Lookup Dictionary
        self.divide_id_to_index = {divide_id: idx for idx, divide_id in enumerate(self.ds.divide_id.values)}

    def forward(self, **kwargs) -> torch.Tensor:
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
        device = kwargs.get("device", "cpu")  # defaulting to a CPU tensor
        dtype = kwargs.get("dtype", torch.float32)  # defaulting to float32
        use_hourly = kwargs.get("use_hourly", False)
        valid_divide_indices = []
        divide_idx_mask = []

        for i, divide_id in enumerate(hydrofabric.divide_ids):
            if divide_id in self.divide_id_to_index:
                valid_divide_indices.append(self.divide_id_to_index[divide_id])
                divide_idx_mask.append(i)
            else:
                log.info(f"{divide_id} missing from the streamflow dataset")

        assert len(valid_divide_indices) != 0, "No valid divide IDs found in this batch. Throwing error"

        _ds = self.ds.isel(time=hydrofabric.dates.numerical_time_range, divide_id=valid_divide_indices)["Qr"]

        if use_hourly is False:
            _ds = _ds.interp(
                time=hydrofabric.dates.batch_hourly_time_range,
                method="nearest",
            )
        streamflow_data = (
            _ds.compute().values.astype(np.float32).T
        )  # Transposing to (num_timesteps, num_features)

        # Creating an output tensor where we're filling any missing data with minimum flow
        output = torch.full(
            (streamflow_data.shape[0], len(hydrofabric.divide_ids)),
            fill_value=0.001,
            device=device,
            dtype=dtype,
        )
        output[:, divide_idx_mask] = torch.tensor(streamflow_data, device=device, dtype=dtype)  # type: ignore
        return output
