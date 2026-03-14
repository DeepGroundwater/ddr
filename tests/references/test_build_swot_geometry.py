"""Tests for references.geo_io.build_swot_geometry — MERIT-SWORD pipeline validation.

Test 1c: Diagnostic flag consistency — For each COMID in the rebuilt NetCDF, verify that
         at least one SWORD reach contributing to it has diag_flag=0 and width > 0.
Test 1d: No orphan COMIDs — Every COMID in the rebuilt NetCDF must exist in the
         MERIT CONUS adjacency matrix (merit_conus_adjacency.zarr).
Test 2:  Coverage check — verify COMIDs and value ranges in the rebuilt NetCDF.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import zarr

# The references module isn't installed as a package — add it to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "references" / "geo_io"))

from build_swot_geometry import (
    MB_PFAF_REGIONS,
    expand_sword_to_comids,
    filter_sword_reaches,
    load_sword_regions_for_mb,
    load_sword_translates,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
_MERIT_SWORD_DIR = Path("/mnt/ssd1/data/swot/merit_sword")
_GEOMETRY_NC = Path(__file__).resolve().parents[2] / "data" / "swot_merit_geometry.nc"
_ADJACENCY_ZARR = Path(__file__).resolve().parents[2] / "data" / "merit_conus_adjacency.zarr"

_SKIP_MSG = "MERIT-SWORD data files not available on this machine"


def _data_available() -> bool:
    return _MERIT_SWORD_DIR.exists() and _GEOMETRY_NC.exists() and _ADJACENCY_ZARR.exists()


pytestmark = pytest.mark.skipif(not _data_available(), reason=_SKIP_MSG)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures (module-scoped: loaded once)
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def geometry_ds() -> xr.Dataset:
    """The rebuilt swot_merit_geometry.nc."""
    return xr.open_dataset(_GEOMETRY_NC)


@pytest.fixture(scope="module")
def adjacency_comids() -> set[int]:
    """All COMIDs present in the MERIT CONUS adjacency matrix."""
    store = zarr.open(_ADJACENCY_ZARR)
    return {int(c) for c in store["order"][:]}


@pytest.fixture(scope="module")
def sword_filtered():
    """Filtered SWORD translate GeoDataFrame (type=1, diag_flag=0)."""
    sword_regions = load_sword_regions_for_mb(_MERIT_SWORD_DIR, MB_PFAF_REGIONS)
    sword_gdf = load_sword_translates(_MERIT_SWORD_DIR, sword_regions)
    return filter_sword_reaches(sword_gdf)


@pytest.fixture(scope="module")
def expected_comid_widths(sword_filtered):
    """COMIDs with top_width as produced by expand_sword_to_comids."""
    return expand_sword_to_comids(sword_filtered)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1c: Diagnostic flag consistency
# ──────────────────────────────────────────────────────────────────────────────
class TestDiagnosticFlagConsistency:
    """Every COMID in the output NetCDF must trace back to at least one type=1, flag=0 reach."""

    def test_every_comid_has_at_least_one_valid_source(
        self,
        geometry_ds: xr.Dataset,
        sword_filtered,
    ) -> None:
        """Verify all SWORD reaches that produced NetCDF COMIDs are type=1, flag=0.

        Since sword_filtered is already filtered, we just need to verify the
        NetCDF COMIDs are a subset of what those filtered reaches produce.
        """
        # Build the set of COMIDs reachable from filtered (type=1, flag=0) reaches
        valid_comids: set[int] = set()
        for _, reach in sword_filtered.iterrows():
            for j in range(1, 41):
                mb_col = f"mb_{j}"
                if mb_col not in reach.index:
                    break
                comid = int(reach[mb_col])
                if comid == 0:
                    break
                valid_comids.add(comid)

        # Every NC COMID with top_width must come from a valid reach
        tw = geometry_ds["top_width"].values
        comid_arr = geometry_ds["COMID"].values
        tw_comids = {int(comid_arr[i]) for i in range(len(tw)) if not np.isnan(tw[i])}

        orphans = tw_comids - valid_comids
        assert len(orphans) == 0, (
            f"{len(orphans)} COMIDs with top_width lack a type=1, flag=0 SWORD source. "
            f"First 20: {sorted(orphans)[:20]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 1d: No orphan COMIDs
# ──────────────────────────────────────────────────────────────────────────────
class TestNoOrphanCOMIDs:
    """Every COMID in the rebuilt NetCDF must exist in the MERIT CONUS adjacency matrix."""

    def test_all_comids_in_adjacency(
        self,
        geometry_ds: xr.Dataset,
        adjacency_comids: set[int],
    ) -> None:
        nc_comids = {int(c) for c in geometry_ds["COMID"].values}
        orphans = nc_comids - adjacency_comids

        assert len(orphans) == 0, (
            f"{len(orphans)} COMIDs in swot_merit_geometry.nc are NOT in "
            f"merit_conus_adjacency.zarr. First 20: {sorted(orphans)[:20]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Coverage and value checks
# ──────────────────────────────────────────────────────────────────────────────
class TestCoverageCount:
    """Verify the NetCDF matches the expected COMIDs from the SWORD translate pipeline."""

    def test_netcdf_width_comids_match_expand(
        self,
        geometry_ds: xr.Dataset,
        expected_comid_widths,
    ) -> None:
        """COMIDs with top_width in the NetCDF should match expand_sword_to_comids output."""
        tw = geometry_ds["top_width"].values
        comid_arr = geometry_ds["COMID"].values
        nc_tw_comids = {int(comid_arr[i]) for i in range(len(tw)) if not np.isnan(tw[i])}

        expected_tw_comids = set(expected_comid_widths.index.values)
        assert nc_tw_comids == expected_tw_comids, (
            f"Width COMID mismatch: NetCDF has {len(nc_tw_comids)}, "
            f"expand produced {len(expected_tw_comids)}. "
            f"Extra: {len(nc_tw_comids - expected_tw_comids)}, "
            f"Missing: {len(expected_tw_comids - nc_tw_comids)}"
        )

    def test_top_width_values_positive(self, geometry_ds: xr.Dataset) -> None:
        """All non-NaN top_width values must be positive."""
        tw = geometry_ds["top_width"].values
        valid = tw[~np.isnan(tw)]
        assert np.all(valid > 0), f"Found {(valid <= 0).sum()} non-positive top_width values"

    def test_side_slope_values_in_range(self, geometry_ds: xr.Dataset) -> None:
        """All non-NaN side_slope values must be in [0.5, 50.0]."""
        ss = geometry_ds["side_slope"].values
        valid = ss[~np.isnan(ss)]
        assert np.all(valid >= 0.5) and np.all(valid <= 50.0), (
            f"side_slope out of [0.5, 50.0]: min={valid.min()}, max={valid.max()}"
        )
