import logging

import icechunk
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class StreamflowReader(torch.nn.Module):
    """A class to read streamflow from a local zarr store or icechunk repo"""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        if "s3" in self.cfg.data_sources.streamflow:
            # Getting the bucket and prefix from an s3:// URI
            bucket = self.cfg.data_sources.streamflow[5:].split("/")[0]
            prefix = self.cfg.data_sources.streamflow[5:].split("/")[1]
            storage_config = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, region=self.cfg.s3_region, anonymous=True
            )
            repo = icechunk.Repository.open(storage_config)
            session = repo.readonly_session("main")
            self.file_path = session.store
        else:
            self.file_path = self.cfg.data_sources.streamflow

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
        xr_streamflow_data = xr.open_zarr(
            self.file_path,
            consolidated=False,
        )
        divide_indices = np.where(np.isin(xr_streamflow_data.divide_id.values, hydrofabric.divide_ids))[0]
        try:
            lazy_flow_data = xr_streamflow_data.isel(
                time=hydrofabric.dates.numerical_time_range, divide_id=divide_indices
            )["Qr"]
        except IndexError as e:
            msg = "index out of bounds. This means you're trying to find a basin that there is no data for."
            log.exception(msg=msg)
            raise IndexError(msg) from e

        lazy_flow_data_interpolated = lazy_flow_data.interp(
            time=hydrofabric.dates.batch_hourly_time_range,
            method="nearest",
        )
        streamflow_data = (
            lazy_flow_data_interpolated.compute().values.astype(np.float32).T
        )  # Transposing to (num_timesteps, num_features)
        return {"streamflow": streamflow_data}
