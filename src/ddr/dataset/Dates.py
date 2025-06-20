import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, model_validator

log = logging.getLogger(__name__)


class Dates(BaseModel):
    """Dates class for handling time operations for training dMC models"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    daily_format: str = "%Y/%m/%d"
    hourly_format: str = "%Y/%m/%d %H:%M:%S"
    origin_start_date: str = "1980/01/01"
    start_time: str
    end_time: str
    rho: int | None = None
    batch_daily_time_range: pd.DatetimeIndex | None = pd.DatetimeIndex([], dtype="datetime64[ns]")
    batch_hourly_time_range: pd.DatetimeIndex | None = pd.DatetimeIndex([], dtype="datetime64[ns]")
    daily_time_range: pd.DatetimeIndex | None = pd.DatetimeIndex([], dtype="datetime64[ns]")
    daily_indices: np.ndarray = np.empty(0)
    hourly_time_range: pd.DatetimeIndex | None = pd.DatetimeIndex([], dtype="datetime64[ns]")
    hourly_indices: torch.Tensor | None = torch.empty(0)
    numerical_time_range: np.ndarray = np.empty(0)

    def __init__(self, **kwargs):
        super().__init__(
            start_time=kwargs["start_time"],
            end_time=kwargs["end_time"],
            rho=kwargs["rho"],
        )

    @model_validator(mode="after")
    @classmethod
    def validate_dates(cls, dates: Any) -> Any:
        """Validates that the number of days you select is within the range of the start and end times"""
        rho = dates.rho
        if isinstance(rho, int):
            if rho > len(dates.daily_time_range):
                log.exception(
                    ValueError("Rho needs to be smaller than the routed period between start and end times")
                )
                raise ValueError("Rho needs to be smaller than the routed period between start and end times")
        return dates

    def model_post_init(self, __context: Any) -> None:
        """Initializes the Dates object and time ranges"""
        self.daily_time_range = pd.date_range(
            datetime.strptime(self.start_time, self.daily_format),
            datetime.strptime(self.end_time, self.daily_format),
            freq="D",
            inclusive="both",
        )
        self.hourly_time_range = pd.date_range(
            start=self.daily_time_range[0],
            end=self.daily_time_range[-1],
            freq="h",
            inclusive="left",
        )
        self.batch_daily_time_range = self.daily_time_range
        self.set_batch_time(self.daily_time_range)

    def set_batch_time(self, daily_time_range: pd.DatetimeIndex) -> None:
        """Sets the time range for the batch you're train/test/simulating

        Parameters
        ----------
        daily_time_range : pd.DatetimeIndex
            The daily time range you want to select
        """
        self.batch_hourly_time_range = pd.date_range(
            start=daily_time_range[0],
            end=daily_time_range[-1],
            freq="h",
            inclusive="left",
        )
        origin_start_date = datetime.strptime(self.origin_start_date, self.daily_format)
        origin_base_start_time = int(
            (daily_time_range[0].to_pydatetime() - origin_start_date).total_seconds() / 86400
        )
        origin_base_end_time = int(
            (daily_time_range[-1].to_pydatetime() - origin_start_date).total_seconds() / 86400
        )

        # The indices for the dates in your selected routing time range
        self.numerical_time_range = np.arange(origin_base_start_time, origin_base_end_time + 1, 1)

        self._create_daily_indices()
        self._create_hourly_indices()

    def _create_hourly_indices(self) -> None:
        common_elements = self.hourly_time_range.intersection(self.batch_hourly_time_range)
        self.hourly_indices = torch.tensor([self.hourly_time_range.get_loc(time) for time in common_elements])

    def _create_daily_indices(self):
        common_elements = self.daily_time_range.intersection(self.batch_daily_time_range)
        self.daily_indices = np.array([self.daily_time_range.get_loc(time) for time in common_elements])

    def calculate_time_period(self) -> None:
        """Calculates the time period for the dataset using rho"""
        if self.rho is not None:
            sample_size = len(self.daily_time_range)
            random_start = torch.randint(low=0, high=sample_size - self.rho, size=(1, 1))[0][0].item()
            self.batch_daily_time_range = self.daily_time_range[random_start : (random_start + self.rho)]
            self.set_batch_time(self.batch_daily_time_range)

    def set_date_range(self, chunk: np.ndarray) -> None:
        """Sets the date range for the dataset

        Parameters
        ----------
        chunk : np.ndarray
            The chunk of the date range you want to select
        """
        self.batch_daily_time_range = self.daily_time_range[chunk]
        self.set_batch_time(self.batch_daily_time_range)

    def create_time_windows(self) -> np.ndarray:
        """Creates the time slices, or windows, for testing the model"""
        num_pieces = self.daily_time_range.shape[0] // self.rho
        last_time = num_pieces * self.rho
        reshaped_arr = np.reshape(self.daily_time_range[:last_time], (num_pieces, self.rho))
        return reshaped_arr
