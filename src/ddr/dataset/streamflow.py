import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import xarray as xr

log = logging.getLogger(__name__)

class StreamflowReader(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.cfg = kwargs["cfg"]

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        hydrofabric = kwargs["hydrofabric"]
        xr_streamflow_data = xr.open_zarr(
            Path(self.cfg.data_sources.streamflow) / "73",
            chunks="auto",
        )
        comid_indices = np.where(xr_streamflow_data.COMID.values == hydrofabric.comids)[0]
        try:
            lazy_flow_data = xr_streamflow_data.isel(
                time=hydrofabric.dates.numerical_time_range, COMID=comid_indices
            ).streamflow
        except IndexError:
            msg = "index out of bounds. This means you're trying to find a basin that there is no data for."
            log.exception(msg=msg)
            raise IndexError(msg)

        lazy_flow_data_interpolated = lazy_flow_data.interp(
            time=hydrofabric.dates.batch_hourly_time_range,
            method="nearest",
        )
        streamflow_data = lazy_flow_data_interpolated.compute().values.astype(np.float32)
        streamflow_predictions = torch.tensor(
            streamflow_data,
        )
        return {"streamflow": streamflow_predictions}
