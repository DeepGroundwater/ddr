"""Attribute adapters for converting external datasets to MERIT-equivalent format.

The DDR spatial KAN is trained on MERIT-Hydro catchment attributes with specific
variable names, units, and normalization statistics. This module provides mappings
to convert attributes from other sources (e.g., HydroATLAS) into the expected
format so the trained KAN can be applied to new domains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# MERIT attribute names (the KAN's native input format)
# ──────────────────────────────────────────────────────────────────────────────
MERIT_ATTRIBUTE_NAMES = (
    "SoilGrids1km_clay",
    "aridity",
    "meanelevation",
    "meanP",
    "NDVI",
    "meanslope",
    "log10_uparea",
    "SoilGrids1km_sand",
    "ETPOT_Hargr",
    "Porosity",
)


@dataclass(frozen=True)
class AttributeMapping:
    """Defines how to convert one external attribute to its MERIT equivalent.

    Parameters
    ----------
    merit_name : str
        Target MERIT variable name.
    scale : float
        Multiplicative factor applied to the source value.
    offset : float
        Additive offset applied after scaling.
    log_transform : bool
        If True, apply ``log10()`` after scaling and offset (used for upstream area).
    """

    merit_name: str
    scale: float = 1.0
    offset: float = 0.0
    log_transform: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# HydroATLAS → MERIT mapping
# ──────────────────────────────────────────────────────────────────────────────
HYDROATLAS_TO_MERIT: dict[str, AttributeMapping] = {
    "cly_pc_sav": AttributeMapping(merit_name="SoilGrids1km_clay"),
    "ari_ix_sav": AttributeMapping(merit_name="aridity"),
    "ele_mt_sav": AttributeMapping(merit_name="meanelevation"),
    "pre_mm_syr": AttributeMapping(merit_name="meanP"),
    "ndv_ix_sav": AttributeMapping(merit_name="NDVI"),
    "slp_dg_sav": AttributeMapping(merit_name="meanslope"),
    "upa_sk_smx": AttributeMapping(merit_name="log10_uparea", log_transform=True),
    "snd_pc_sav": AttributeMapping(merit_name="SoilGrids1km_sand"),
    "pet_mm_syr": AttributeMapping(merit_name="ETPOT_Hargr"),
    "por_pc_sav": AttributeMapping(merit_name="Porosity"),
}

_KNOWN_SOURCES: dict[str, dict[str, AttributeMapping]] = {
    "hydroatlas": HYDROATLAS_TO_MERIT,
}


def detect_source(ds: xr.Dataset) -> str | None:
    """Detect the attribute source from variable names.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset whose variable names will be checked.

    Returns
    -------
    str or None
        Detected source name (e.g., ``"hydroatlas"``, ``"merit"``), or None
        if the source cannot be determined.
    """
    var_names = set(ds.data_vars)

    # Already in MERIT format?
    if var_names >= set(MERIT_ATTRIBUTE_NAMES):
        return "merit"

    # Check registered external sources
    for source_name, mapping in _KNOWN_SOURCES.items():
        if var_names >= set(mapping.keys()):
            return source_name

    return None


def adapt_attributes(ds: xr.Dataset, source: str = "auto") -> xr.Dataset:
    """Convert external attributes to MERIT-equivalent names and units.

    Parameters
    ----------
    ds : xr.Dataset
        Input attributes. May use MERIT, HydroATLAS, or other naming conventions.
    source : str
        Source format: ``"merit"`` (no-op), ``"hydroatlas"``, or ``"auto"``
        (detect from variable names).

    Returns
    -------
    xr.Dataset
        Dataset with MERIT-equivalent variable names and units, ready for
        z-score normalization and KAN inference.

    Raises
    ------
    ValueError
        If the source cannot be detected or required variables are missing.
    """
    if source == "auto":
        detected = detect_source(ds)
        if detected is None:
            msg = (
                f"Cannot auto-detect attribute source from variables: {sorted(ds.data_vars)}. "
                f"Expected MERIT names {MERIT_ATTRIBUTE_NAMES} or HydroATLAS names "
                f"{sorted(HYDROATLAS_TO_MERIT.keys())}. Specify source='merit' or source='hydroatlas'."
            )
            raise ValueError(msg)
        source = detected

    if source == "merit":
        # Validate all required attributes are present
        missing = set(MERIT_ATTRIBUTE_NAMES) - set(ds.data_vars)
        if missing:
            msg = f"Missing MERIT attributes: {sorted(missing)}"
            raise ValueError(msg)
        return ds[list(MERIT_ATTRIBUTE_NAMES)]

    mapping = _KNOWN_SOURCES.get(source)
    if mapping is None:
        msg = f"Unknown attribute source: {source!r}. Known sources: {sorted(_KNOWN_SOURCES.keys())}"
        raise ValueError(msg)

    # Validate all required source attributes are present
    missing = set(mapping.keys()) - set(ds.data_vars)
    if missing:
        msg = f"Missing {source} attributes: {sorted(missing)}"
        raise ValueError(msg)

    converted: dict[str, xr.DataArray] = {}
    for src_name, attr_map in mapping.items():
        values = ds[src_name].astype(np.float64) * attr_map.scale + attr_map.offset
        if attr_map.log_transform:
            values = np.log10(values.clip(min=1e-6))
        converted[attr_map.merit_name] = values

    log.info("Converted %d attributes from %s to MERIT format", len(converted), source)
    result = xr.Dataset(converted, coords=ds.coords)
    return result[list(MERIT_ATTRIBUTE_NAMES)]
