"""Tests for mass-conservative linear interpolation of q_prime.

Verifies that mass_conservative_rescale:
1. Preserves daily mass balance exactly
2. Produces within-day variability (not flat like nearest)
3. Keeps all values non-negative
4. Handles edge cases (constant data, single day, near-zero flows)
"""

import numpy as np

from ddr.io.functions import mass_conservative_rescale


class TestMassConservativeRescale:
    """Tests for mass_conservative_rescale()."""

    def test_daily_means_preserved_exactly(self) -> None:
        """After rescaling, each day's 24-hour mean must equal the original daily value."""
        rng = np.random.default_rng(42)
        num_days = 30
        num_divides = 10
        daily_data = rng.uniform(1.0, 100.0, size=(num_days, num_divides)).astype(np.float32)

        # Simulate linear interpolation: ramp between consecutive daily values
        num_complete_days = num_days - 1
        hourly_data = np.empty((num_complete_days * 24, num_divides), dtype=np.float32)
        for d in range(num_complete_days):
            for h in range(24):
                frac = h / 24.0
                hourly_data[d * 24 + h] = daily_data[d] * (1 - frac) + daily_data[d + 1] * frac

        result = mass_conservative_rescale(hourly_data, daily_data)

        # Check each day's mean matches the original daily value
        for d in range(num_complete_days):
            day_slice = result[d * 24 : (d + 1) * 24]
            day_mean = day_slice.mean(axis=0)
            np.testing.assert_allclose(
                day_mean,
                daily_data[d],
                rtol=1e-5,
                err_msg=f"Day {d}: mean {day_mean} != daily {daily_data[d]}",
            )

    def test_within_day_variability(self) -> None:
        """Linearly interpolated + rescaled data should have within-day variance."""
        num_divides = 5
        # Two days with different values → ramp between them
        daily_data = np.array([[10.0] * num_divides, [50.0] * num_divides], dtype=np.float32)

        # Linear ramp from 10 to 50 over 24 hours
        hourly_data = np.empty((24, num_divides), dtype=np.float32)
        for h in range(24):
            frac = h / 24.0
            hourly_data[h] = daily_data[0] * (1 - frac) + daily_data[1] * frac

        result = mass_conservative_rescale(hourly_data, daily_data)

        # The first hour should differ from the last hour
        assert not np.allclose(result[0], result[23]), "Expected within-day variability, got flat signal"

        # Variance should be > 0 for each divide
        for d in range(num_divides):
            assert result[:, d].std() > 0, f"Divide {d} has zero within-day std"

    def test_all_values_non_negative(self) -> None:
        """Rescaled values should remain non-negative when daily inputs are non-negative."""
        rng = np.random.default_rng(123)
        num_days = 20
        num_divides = 8
        daily_data = rng.uniform(0.1, 200.0, size=(num_days, num_divides)).astype(np.float32)

        # Linear interpolation
        num_complete_days = num_days - 1
        hourly_data = np.empty((num_complete_days * 24, num_divides), dtype=np.float32)
        for d in range(num_complete_days):
            for h in range(24):
                frac = h / 24.0
                hourly_data[d * 24 + h] = daily_data[d] * (1 - frac) + daily_data[d + 1] * frac

        result = mass_conservative_rescale(hourly_data, daily_data)
        assert np.all(result >= 0), f"Found negative values: min={result.min()}"

    def test_constant_daily_values_unchanged(self) -> None:
        """If all daily values are identical, hourly values should be constant."""
        num_divides = 3
        daily_data = np.full((10, num_divides), 42.0, dtype=np.float32)

        # Constant interpolation → constant hourly
        hourly_data = np.full((9 * 24, num_divides), 42.0, dtype=np.float32)

        result = mass_conservative_rescale(hourly_data, daily_data)
        np.testing.assert_allclose(result, 42.0, rtol=1e-6)

    def test_single_complete_day(self) -> None:
        """A single day (24 hours) should still preserve mass."""
        daily_data = np.array([[25.0, 50.0]], dtype=np.float32)
        # Hourly data that doesn't average to the daily value
        hourly_data = np.full((24, 2), 30.0, dtype=np.float32)

        result = mass_conservative_rescale(hourly_data, daily_data)
        day_mean = result.mean(axis=0)
        np.testing.assert_allclose(day_mean, daily_data[0], rtol=1e-5)

    def test_near_zero_daily_values_safe(self) -> None:
        """Near-zero daily values should not cause division-by-zero."""
        daily_data = np.array([[1e-10, 50.0], [1e-10, 100.0]], dtype=np.float32)
        # Hourly data with near-zero values for first divide
        hourly_data = np.full((24, 2), dtype=np.float32, fill_value=0.0)
        hourly_data[:, 1] = np.linspace(50.0, 100.0, 24)

        result = mass_conservative_rescale(hourly_data, daily_data)
        assert np.all(np.isfinite(result)), "Found non-finite values (inf/nan)"

    def test_does_not_modify_input(self) -> None:
        """The function should not modify the input arrays."""
        daily_data = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        hourly_data = np.full((48, 1), 15.0, dtype=np.float32)
        daily_copy = daily_data.copy()
        hourly_copy = hourly_data.copy()

        mass_conservative_rescale(hourly_data, daily_data)

        np.testing.assert_array_equal(daily_data, daily_copy)
        np.testing.assert_array_equal(hourly_data, hourly_copy)

    def test_total_volume_preserved(self) -> None:
        """Sum of hourly values × dt should equal sum of daily values × dt.

        Since mean(hourly) = daily for each day, the total volume
        (sum over all days) is preserved exactly.
        """
        rng = np.random.default_rng(999)
        num_days = 50
        num_divides = 4
        daily_data = rng.uniform(5.0, 500.0, size=(num_days, num_divides)).astype(np.float32)

        num_complete_days = num_days - 1
        hourly_data = np.empty((num_complete_days * 24, num_divides), dtype=np.float32)
        for d in range(num_complete_days):
            for h in range(24):
                frac = h / 24.0
                hourly_data[d * 24 + h] = daily_data[d] * (1 - frac) + daily_data[d + 1] * frac

        result = mass_conservative_rescale(hourly_data, daily_data)

        # Total volume: sum of daily means × 24 hours = sum of all hourly values
        for div in range(num_divides):
            hourly_sum = result[:, div].sum()
            daily_sum = daily_data[:num_complete_days, div].sum() * 24
            np.testing.assert_allclose(
                hourly_sum,
                daily_sum,
                rtol=1e-4,
                err_msg=f"Divide {div}: hourly volume {hourly_sum} != daily volume {daily_sum}",
            )

    def test_large_day_to_day_jump(self) -> None:
        """Extreme daily changes (drought → flood) should still conserve mass."""
        daily_data = np.array([[1.0], [1000.0], [2.0]], dtype=np.float32)
        hourly_data = np.empty((48, 1), dtype=np.float32)
        for d in range(2):
            for h in range(24):
                frac = h / 24.0
                hourly_data[d * 24 + h] = daily_data[d] * (1 - frac) + daily_data[d + 1] * frac

        result = mass_conservative_rescale(hourly_data, daily_data)

        # Day 0 mean should be 1.0
        np.testing.assert_allclose(result[:24].mean(), 1.0, rtol=1e-5)
        # Day 1 mean should be 1000.0
        np.testing.assert_allclose(result[24:48].mean(), 1000.0, rtol=1e-5)

    def test_incomplete_last_day_untouched(self) -> None:
        """Hours beyond the last complete day should pass through unchanged."""
        daily_data = np.array([[10.0], [20.0]], dtype=np.float32)
        # 30 hours = 1 complete day + 6 leftover hours
        hourly_data = np.full((30, 1), 15.0, dtype=np.float32)
        hourly_data[24:] = 99.0  # sentinel for the leftover hours

        result = mass_conservative_rescale(hourly_data, daily_data)

        # Leftover hours should be unchanged
        np.testing.assert_array_equal(result[24:], 99.0)
        # Complete day should be rescaled
        np.testing.assert_allclose(result[:24].mean(), 10.0, rtol=1e-5)
