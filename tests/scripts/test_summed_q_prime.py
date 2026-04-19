"""Tests for summed_q_prime.eval_q_prime — validates that the optimised hourly
aggregation path (isel + reshape + CuPy) produces correct predictions and
matches a pure-NumPy reference implementation."""

import cupy as cp
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage
from omegaconf import OmegaConf

from ddr.validation import GeoDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_streamflow_ds(
    divide_ids: np.ndarray,
    start: str,
    n_days: int,
    freq: str = "D",
    seed: int = 42,
) -> xr.Dataset:
    """Build a minimal streamflow xr.Dataset at daily or hourly frequency."""
    rng = np.random.default_rng(seed)
    n_divides = len(divide_ids)

    if freq == "h":
        n_steps = n_days * 24
        time = pd.date_range(start, periods=n_steps, freq="h")
        # Build hourly data that has a known daily mean:
        # repeat a daily pattern across 24 hours with small hourly noise
        daily_base = rng.uniform(0.1, 10.0, size=(n_divides, n_days)).astype(np.float32)
        data = np.repeat(daily_base, 24, axis=1)
        # Add small noise so hourly != constant (mean still ≈ daily_base)
        noise = rng.normal(0, 0.01, size=data.shape).astype(np.float32)
        data = np.maximum(data + noise, 1e-6)
    else:
        time = pd.date_range(start, periods=n_days, freq="D")
        data = rng.uniform(0.1, 10.0, size=(n_divides, n_days)).astype(np.float32)

    return xr.Dataset(
        {"Qr": (["divide_id", "time"], data, {"units": "m^3/s"})},
        coords={"divide_id": divide_ids, "time": time},
    )


def _make_observations_ds(
    gage_ids: list[str],
    time_range: pd.DatetimeIndex,
    seed: int = 99,
) -> xr.Dataset:
    """Build a minimal USGS observations xr.Dataset."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(1.0, 50.0, size=(len(gage_ids), len(time_range))).astype(np.float32)
    return xr.Dataset(
        {"streamflow": (["gage_id", "time"], data)},
        coords={"gage_id": gage_ids, "time": time_range},
    )


def _make_gages_adjacency(
    gage_ids: list[str],
    divide_ids: np.ndarray,
    divides_per_gage: int = 3,
) -> zarr.Group:
    """Build a minimal in-memory gages adjacency zarr group."""
    store = zarr.storage.MemoryStore()
    root = zarr.open_group(store, mode="w")
    for i, gid in enumerate(gage_ids):
        g = root.create_group(gid)
        start = (i * divides_per_gage) % len(divide_ids)
        order = divide_ids[start : start + divides_per_gage]
        g.create_array("order", data=order)
    return root


def _make_cfg(
    start_time: str,
    end_time: str,
    is_hourly: bool = False,
) -> "OmegaConf":
    """Build a minimal OmegaConf DictConfig for eval_q_prime."""
    return OmegaConf.create(
        {
            "geodataset": GeoDataset.MERIT.value,
            "experiment": {"start_time": start_time, "end_time": end_time},
            "data_sources": {"is_hourly": is_hourly},
            "params": {"save_path": "/tmp"},
        }
    )


def _reference_preds_numpy(
    streamflow: xr.Dataset,
    gage_ids: list[str],
    divide_ids: np.ndarray,
    eval_time_range: pd.DatetimeIndex,
    is_hourly: bool = False,
) -> np.ndarray:
    """Pure-NumPy reference: the old resample path for cross-checking.

    This is the pre-optimisation logic, known to be correct.
    """
    if is_hourly:
        streamflow = (
            streamflow.sel(
                time=slice(eval_time_range[0], eval_time_range[-1] + pd.Timedelta(hours=23)),
            )
            .resample(time="D")
            .mean()
            .transpose("divide_id", "time")
            .compute()
        )

    conus_divide_ids = streamflow.divide_id.values
    n_eval_days = len(eval_time_range)

    gages_adjacency = _make_gages_adjacency(gage_ids, divide_ids)
    valid_gauges = np.array(gage_ids)
    preds = np.zeros([len(valid_gauges), n_eval_days], dtype=np.float32)

    conus_time_range = streamflow.time.values
    time_indices = np.where(np.isin(conus_time_range, eval_time_range))[0]

    for i, gauge in enumerate(valid_gauges):
        basins = gages_adjacency[gauge]["order"][:]
        divide_indices = np.where(np.isin(conus_divide_ids, basins))[0]
        qr = streamflow.isel(time=time_indices, divide_id=divide_indices)["Qr"].values.astype(np.float32)
        preds[i] = np.nansum(qr, axis=0)

    return preds


def _optimised_preds_cupy(
    streamflow: xr.Dataset,
    gage_ids: list[str],
    divide_ids: np.ndarray,
    eval_time_range: pd.DatetimeIndex,
    is_hourly: bool = False,
) -> np.ndarray:
    """Mirrors the new optimised eval_q_prime logic (isel + CuPy)."""
    gages_adjacency = _make_gages_adjacency(gage_ids, divide_ids)
    valid_gauges = np.array(gage_ids)

    # Pre-collect all upstream divides
    gauge_basins: dict[str, np.ndarray] = {}
    all_needed_basins: set = set()
    for gauge in valid_gauges:
        basins = gages_adjacency[gauge]["order"][:]
        gauge_basins[gauge] = basins
        all_needed_basins.update(basins)

    conus_divide_ids = streamflow.divide_id.values
    needed_divide_indices = np.where(np.isin(conus_divide_ids, list(all_needed_basins)))[0]

    if is_hourly:
        store_start = pd.Timestamp(streamflow.time.values[0])
        start_idx = int((eval_time_range[0] - store_start).total_seconds() // 3600)
        end_idx = int((eval_time_range[-1] + pd.Timedelta(hours=23) - store_start).total_seconds() // 3600)
        hourly = streamflow.isel(
            time=slice(start_idx, end_idx + 1), divide_id=needed_divide_indices
        ).compute()
        qr_hourly = hourly["Qr"].values  # stays on CPU
        n_days = qr_hourly.shape[1] // 24
        qr_daily = qr_hourly[:, : n_days * 24].reshape(qr_hourly.shape[0], n_days, 24).mean(axis=2)
        filtered_divide_ids = conus_divide_ids[needed_divide_indices]
        qr_gpu = cp.asarray(qr_daily.astype(np.float32))  # daily result → GPU
    else:
        conus_time_range = streamflow.time.values
        time_indices = np.where(np.isin(conus_time_range, eval_time_range))[0]
        filtered_divide_ids = conus_divide_ids[needed_divide_indices]
        qr_gpu = cp.asarray(
            streamflow.isel(time=time_indices, divide_id=needed_divide_indices)["Qr"].values.astype(
                np.float32
            )
        )

    preds = cp.zeros([len(valid_gauges), qr_gpu.shape[1]], dtype=cp.float32)
    for i, gauge in enumerate(valid_gauges):
        divide_indices = np.where(np.isin(filtered_divide_ids, gauge_basins[gauge]))[0]
        preds[i] = cp.nansum(qr_gpu[divide_indices], axis=0)

    return cp.asnumpy(preds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvalQPrimeHourly:
    """Validate that the optimised hourly aggregation produces correct daily predictions."""

    DIVIDE_IDS = np.arange(1000, 1012)
    GAGE_IDS = ["00000001", "00000002", "00000003"]
    START = "1990/10/01"
    END = "1990/10/30"
    N_DAYS = 30

    def _eval_time_range(self) -> pd.DatetimeIndex:
        return pd.date_range("1990-10-01", "1990-10-30", freq="D")

    def test_hourly_constant_matches_daily(self) -> None:
        """Hourly data constant within each day must produce identical preds to the daily store."""
        eval_range = self._eval_time_range()

        # Build daily store
        ds_daily = _make_streamflow_ds(self.DIVIDE_IDS, "1990-10-01", self.N_DAYS, freq="D", seed=42)

        # Build hourly store by repeating daily values (no noise → mean == original)
        daily_data = ds_daily["Qr"].values
        hourly_data = np.repeat(daily_data, 24, axis=1)
        hourly_time = pd.date_range("1990-10-01", periods=self.N_DAYS * 24, freq="h")
        ds_hourly = xr.Dataset(
            {"Qr": (["divide_id", "time"], hourly_data, {"units": "m^3/s"})},
            coords={"divide_id": self.DIVIDE_IDS, "time": hourly_time},
        )

        preds_daily = _optimised_preds_cupy(
            ds_daily, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=False
        )
        preds_hourly = _optimised_preds_cupy(
            ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True
        )

        np.testing.assert_allclose(preds_hourly, preds_daily, rtol=1e-5)

    def test_hourly_aggregation_computes_daily_mean(self) -> None:
        """Hourly values that vary within a day should produce their daily mean."""
        eval_range = self._eval_time_range()

        # hours 0-11 = 2.0, hours 12-23 = 4.0 → daily mean = 3.0
        n_divides = len(self.DIVIDE_IDS)
        hourly_time = pd.date_range("1990-10-01", periods=self.N_DAYS * 24, freq="h")
        hourly_data = np.zeros((n_divides, self.N_DAYS * 24), dtype=np.float32)
        for d in range(self.N_DAYS):
            hourly_data[:, d * 24 : d * 24 + 12] = 2.0
            hourly_data[:, d * 24 + 12 : d * 24 + 24] = 4.0
        ds_hourly = xr.Dataset(
            {"Qr": (["divide_id", "time"], hourly_data, {"units": "m^3/s"})},
            coords={"divide_id": self.DIVIDE_IDS, "time": hourly_time},
        )

        preds = _optimised_preds_cupy(ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True)

        # Each gage covers 3 divides, each contributing daily mean of 3.0
        expected = np.full((len(self.GAGE_IDS), self.N_DAYS), 9.0, dtype=np.float32)
        np.testing.assert_allclose(preds, expected, rtol=1e-5)

    def test_hourly_output_shape_matches_eval_range(self) -> None:
        """Hourly path output should have exactly n_eval_days columns."""
        eval_range = self._eval_time_range()
        # Hourly store covers more than the eval period
        ds_hourly = _make_streamflow_ds(self.DIVIDE_IDS, "1990-09-01", 90, freq="h", seed=42)

        preds = _optimised_preds_cupy(ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True)

        assert preds.shape == (len(self.GAGE_IDS), self.N_DAYS)
        assert not np.any(np.isnan(preds))

    def test_optimised_matches_reference_daily(self) -> None:
        """Optimised CuPy path must match pure-NumPy reference on daily data."""
        eval_range = self._eval_time_range()
        ds = _make_streamflow_ds(self.DIVIDE_IDS, "1990-10-01", self.N_DAYS, freq="D", seed=42)

        ref = _reference_preds_numpy(ds, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=False)
        opt = _optimised_preds_cupy(ds, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=False)

        np.testing.assert_allclose(opt, ref, rtol=1e-5)

    def test_optimised_matches_reference_hourly(self) -> None:
        """Optimised CuPy+isel path must match pure-NumPy resample reference on hourly data."""
        eval_range = self._eval_time_range()

        # Use constant-within-day hourly data so resample mean == reshape mean exactly
        ds_daily = _make_streamflow_ds(self.DIVIDE_IDS, "1990-10-01", self.N_DAYS, freq="D", seed=42)
        daily_data = ds_daily["Qr"].values
        hourly_data = np.repeat(daily_data, 24, axis=1)
        hourly_time = pd.date_range("1990-10-01", periods=self.N_DAYS * 24, freq="h")
        ds_hourly = xr.Dataset(
            {"Qr": (["divide_id", "time"], hourly_data, {"units": "m^3/s"})},
            coords={"divide_id": self.DIVIDE_IDS, "time": hourly_time},
        )

        ref = _reference_preds_numpy(ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True)
        opt = _optimised_preds_cupy(ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True)

        np.testing.assert_allclose(opt, ref, rtol=1e-5)

    def test_pre_filtered_divides_excludes_unused(self) -> None:
        """Extra divides not upstream of any gauge should not affect predictions."""
        eval_range = self._eval_time_range()

        # Core divides used by gauges
        core_ids = self.DIVIDE_IDS
        ds_core = _make_streamflow_ds(core_ids, "1990-10-01", self.N_DAYS, freq="D", seed=42)

        # Extended store with 20 extra divides that no gauge references
        extra_ids = np.arange(2000, 2020)
        all_ids = np.concatenate([core_ids, extra_ids])
        rng = np.random.default_rng(42)
        core_data = ds_core["Qr"].values
        extra_data = rng.uniform(100.0, 200.0, size=(len(extra_ids), self.N_DAYS)).astype(np.float32)
        all_data = np.concatenate([core_data, extra_data], axis=0)
        daily_time = pd.date_range("1990-10-01", periods=self.N_DAYS, freq="D")
        ds_extended = xr.Dataset(
            {"Qr": (["divide_id", "time"], all_data, {"units": "m^3/s"})},
            coords={"divide_id": all_ids, "time": daily_time},
        )

        preds_core = _optimised_preds_cupy(ds_core, self.GAGE_IDS, core_ids, eval_range, is_hourly=False)
        preds_extended = _optimised_preds_cupy(
            ds_extended, self.GAGE_IDS, core_ids, eval_range, is_hourly=False
        )

        np.testing.assert_allclose(preds_extended, preds_core, rtol=1e-6)

    def test_store_offset_hourly(self) -> None:
        """Hourly store starting before eval period should produce correct indices."""
        eval_range = self._eval_time_range()

        # Store starts 30 days before eval period
        ds_hourly = _make_streamflow_ds(self.DIVIDE_IDS, "1990-09-01", 90, freq="h", seed=42)

        # Extract the eval-period slice and build an aligned daily store for reference
        eval_hourly = ds_hourly.sel(time=slice(eval_range[0], eval_range[-1] + pd.Timedelta(hours=23)))
        daily_data = eval_hourly["Qr"].values.reshape(len(self.DIVIDE_IDS), self.N_DAYS, 24).mean(axis=2)
        ds_daily_aligned = xr.Dataset(
            {"Qr": (["divide_id", "time"], daily_data, {"units": "m^3/s"})},
            coords={"divide_id": self.DIVIDE_IDS, "time": eval_range},
        )

        preds_hourly = _optimised_preds_cupy(
            ds_hourly, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=True
        )
        preds_daily = _optimised_preds_cupy(
            ds_daily_aligned, self.GAGE_IDS, self.DIVIDE_IDS, eval_range, is_hourly=False
        )

        np.testing.assert_allclose(preds_hourly, preds_daily, rtol=1e-5)
