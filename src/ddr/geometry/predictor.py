"""Standalone geometry predictor using a trained DDR KAN.

Decouples the KAN spatial parameter prediction and trapezoidal geometry
computation from the full Muskingum-Cunge routing pipeline. Users provide
catchment attributes, discharge, and channel slope; the predictor returns
complete channel cross-section geometry.

Example
-------
>>> from ddr.geometry import GeometryPredictor
>>> predictor = GeometryPredictor.from_checkpoint(
...     checkpoint_path="runs/best_model.pt",
...     config_path="config/merit.yaml",
... )
>>> result = predictor.predict(attributes_ds, discharge, slope)
>>> result["top_width"]  # xr.DataArray of bankfull width per reach
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf

from ddr.geometry.adapters import adapt_attributes
from ddr.geometry.trapezoidal import compute_trapezoidal_geometry
from ddr.nn import kan
from ddr.routing.utils import denormalize
from ddr.scripts_utils import load_checkpoint
from ddr.validation.configs import validate_config

log = logging.getLogger(__name__)


class GeometryPredictor:
    """Predict trapezoidal channel geometry from catchment attributes and discharge.

    Wraps a trained DDR KAN to provide a simple interface for geometry estimation
    without requiring the full routing network. The KAN maps catchment attributes
    to physical channel parameters (Manning's n, Leopold & Maddock p and q), which
    are then combined with discharge and slope to compute cross-section geometry.

    Parameters
    ----------
    nn_model : kan
        Trained KAN neural network (already loaded with checkpoint weights).
    attribute_names : list[str]
        Ordered list of attribute variable names the KAN expects.
    means : torch.Tensor
        Z-score normalization means for each attribute. Shape ``(n_attrs,)``.
    stds : torch.Tensor
        Z-score normalization standard deviations. Shape ``(n_attrs,)``.
    parameter_ranges : dict[str, list[float]]
        Physical bounds for each learnable parameter (e.g., ``{"n": [0.015, 0.25]}``).
    log_space_parameters : list[str]
        Parameters that use log-space denormalization.
    defaults : dict[str, float]
        Default values for parameters not learned by the KAN.
    attribute_minimums : dict[str, float]
        Physical lower bounds for depth, bottom_width, etc.
    stats_ranges : dict[str, dict[str, float]] | None
        Per-attribute p10/p90 ranges for out-of-distribution warnings.
    device : str
        Torch device (``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        nn_model: kan,
        attribute_names: list[str],
        means: torch.Tensor,
        stds: torch.Tensor,
        parameter_ranges: dict[str, list[float]],
        log_space_parameters: list[str],
        defaults: dict[str, float],
        attribute_minimums: dict[str, float],
        stats_ranges: dict[str, dict[str, float]] | None = None,
        device: str = "cpu",
    ) -> None:
        self._nn = nn_model
        self._nn.eval()
        self._attribute_names = attribute_names
        self._means = means.to(device)
        self._stds = stds.to(device)
        self._parameter_ranges = parameter_ranges
        self._log_space_parameters = log_space_parameters
        self._defaults = defaults
        self._attribute_minimums = attribute_minimums
        self._stats_ranges = stats_ranges
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_path: str | Path,
        stats_path: str | Path | None = None,
        device: str = "cpu",
    ) -> GeometryPredictor:
        """Create a GeometryPredictor from a trained DDR checkpoint.

        Parameters
        ----------
        checkpoint_path : str or Path
            Path to the ``.pt`` checkpoint file containing KAN weights.
        config_path : str or Path
            Path to the Hydra YAML config used during training (needed for
            KAN architecture, parameter bounds, and attribute list).
        stats_path : str or Path or None
            Path to the attribute statistics JSON file. If None, auto-detected
            from the config's ``data_sources.statistics`` directory.
        device : str
            Torch device for inference. Default ``"cpu"``.

        Returns
        -------
        GeometryPredictor
            Ready-to-use predictor instance.
        """
        # Load and validate config
        omega_cfg = OmegaConf.load(str(config_path))
        omega_cfg.device = device
        omega_cfg.mode = "route"
        cfg = validate_config(omega_cfg, save_config=False)

        # Build the KAN with the same architecture as training
        attribute_names = list(cfg.kan.input_var_names)
        nn_model = kan(
            input_var_names=attribute_names,
            learnable_parameters=cfg.kan.learnable_parameters,
            hidden_size=cfg.kan.hidden_size,
            num_hidden_layers=cfg.kan.num_hidden_layers,
            grid=cfg.kan.grid,
            k=cfg.kan.k,
            seed=cfg.seed,
            device=device,
        )
        load_checkpoint(nn_model, checkpoint_path, device)
        nn_model.eval()

        # Load normalization statistics
        means, stds, stats_ranges = cls._load_normalization_stats(cfg, attribute_names, stats_path, device)

        return cls(
            nn_model=nn_model,
            attribute_names=attribute_names,
            means=means,
            stds=stds,
            parameter_ranges=cfg.params.parameter_ranges,
            log_space_parameters=cfg.params.log_space_parameters,
            defaults=cfg.params.defaults,
            attribute_minimums=cfg.params.attribute_minimums,
            stats_ranges=stats_ranges,
            device=device,
        )

    def predict(
        self,
        attributes: xr.Dataset,
        discharge: xr.DataArray,
        slope: xr.DataArray,
        source: str = "auto",
    ) -> xr.Dataset:
        """Predict channel geometry for all reaches.

        Parameters
        ----------
        attributes : xr.Dataset
            Catchment attributes with 10 variables per reach. Accepts MERIT
            native names or HydroATLAS names (auto-detected or specified via
            ``source``).
        discharge : xr.DataArray
            Representative discharge per reach (m^3/s).
        slope : xr.DataArray
            Channel bed slope per reach (m/m, dimensionless).
        source : str
            Attribute source format: ``"merit"``, ``"hydroatlas"``, or
            ``"auto"`` (detect from variable names). Default ``"auto"``.

        Returns
        -------
        xr.Dataset
            Geometry variables per reach: ``top_width``, ``depth``,
            ``bottom_width``, ``side_slope``, ``cross_sectional_area``,
            ``wetted_perimeter``, ``hydraulic_radius``, ``velocity``,
            ``n``, ``p_spatial``, ``q_spatial``.
        """
        # 1. Adapt attributes to MERIT format
        adapted = adapt_attributes(attributes, source=source)

        # 2. Check for out-of-distribution attributes
        self._check_distribution(adapted)

        # 3. Build attribute tensor and normalize
        attr_tensor = self._prepare_attributes(adapted)

        # 4. KAN inference → denormalize to physical parameters
        n, p_spatial, q_spatial = self._predict_parameters(attr_tensor)

        # 5. Compute geometry
        q_tensor = torch.tensor(np.asarray(discharge.values, dtype=np.float32), device=self._device)
        s_tensor = torch.tensor(np.asarray(slope.values, dtype=np.float32), device=self._device)

        # Clamp slope to physical minimum
        slope_lb = self._attribute_minimums.get("slope", 0.0001)
        q_tensor = torch.clamp(q_tensor, min=self._attribute_minimums.get("discharge", 0.0001))
        s_tensor = torch.clamp(s_tensor, min=slope_lb)

        geometry = compute_trapezoidal_geometry(
            n=n,
            p_spatial=p_spatial,
            q_spatial=q_spatial,
            discharge=q_tensor,
            slope=s_tensor,
            depth_lb=self._attribute_minimums.get("depth", 0.01),
            bottom_width_lb=self._attribute_minimums.get("bottom_width", 0.01),
        )

        # 6. Build output xr.Dataset
        coords = discharge.coords
        dim_name = discharge.dims[0] if discharge.dims else "reach"

        data_vars: dict[str, Any] = {}
        for name, tensor in geometry.items():
            data_vars[name] = (dim_name, tensor.detach().cpu().numpy())

        # Include learned parameters in output
        data_vars["n"] = (dim_name, n.detach().cpu().numpy())
        data_vars["p_spatial"] = (dim_name, p_spatial.detach().cpu().numpy())
        data_vars["q_spatial"] = (dim_name, q_spatial.detach().cpu().numpy())

        return xr.Dataset(data_vars, coords=coords)

    def _prepare_attributes(self, adapted: xr.Dataset) -> torch.Tensor:
        """Convert adapted xr.Dataset to a normalized torch tensor.

        Parameters
        ----------
        adapted : xr.Dataset
            Attributes in MERIT format.

        Returns
        -------
        torch.Tensor
            Normalized attribute tensor, shape ``(N, n_attrs)``.
        """
        arrays = []
        for attr_name in self._attribute_names:
            arr = np.asarray(adapted[attr_name].values, dtype=np.float32)
            arrays.append(arr)

        # Shape: (n_attrs, N)
        raw = torch.tensor(np.stack(arrays, axis=0), device=self._device)

        # Z-score normalize: (attr - mean) / std
        normalized = (raw - self._means.unsqueeze(1)) / self._stds.unsqueeze(1)

        # Transpose to (N, n_attrs) for KAN input
        return normalized.T

    def _predict_parameters(
        self, attr_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run KAN inference and denormalize outputs to physical parameters.

        Parameters
        ----------
        attr_tensor : torch.Tensor
            Normalized attributes, shape ``(N, n_attrs)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``(n, p_spatial, q_spatial)`` in physical units.
        """
        with torch.no_grad():
            raw_params = self._nn(inputs=attr_tensor)

        log_space = self._log_space_parameters

        n = denormalize(
            value=raw_params["n"],
            bounds=self._parameter_ranges["n"],
            log_space="n" in log_space,
        )
        q_spatial = denormalize(
            value=raw_params["q_spatial"],
            bounds=self._parameter_ranges["q_spatial"],
            log_space="q_spatial" in log_space,
        )

        if "p_spatial" in raw_params and "p_spatial" in self._parameter_ranges:
            p_spatial = denormalize(
                value=raw_params["p_spatial"],
                bounds=self._parameter_ranges["p_spatial"],
                log_space="p_spatial" in log_space,
            )
        else:
            default_p = self._defaults.get("p_spatial", 21.0)
            p_spatial = torch.full_like(n, default_p)
            log.info("p_spatial not learned; using default value %.1f", default_p)

        return n, p_spatial, q_spatial

    def _check_distribution(self, adapted: xr.Dataset) -> None:
        """Warn if input attributes fall outside the training distribution.

        Parameters
        ----------
        adapted : xr.Dataset
            Attributes in MERIT format.
        """
        if self._stats_ranges is None:
            return

        for attr_name in self._attribute_names:
            if attr_name not in self._stats_ranges:
                continue
            ranges = self._stats_ranges[attr_name]
            values = adapted[attr_name].values
            p10, p90 = ranges["p10"], ranges["p90"]
            below = np.sum(values < p10)
            above = np.sum(values > p90)
            total = len(values)
            if below > 0 or above > 0:
                log.warning(
                    "Attribute %s: %d/%d values below training p10 (%.3f), %d/%d above training p90 (%.3f)",
                    attr_name,
                    below,
                    total,
                    p10,
                    above,
                    total,
                    p90,
                )

    @staticmethod
    def _load_normalization_stats(
        cfg: Any,
        attribute_names: list[str],
        stats_path: str | Path | None,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]] | None]:
        """Load z-score normalization statistics from a JSON file.

        Parameters
        ----------
        cfg : Config
            Validated DDR config.
        attribute_names : list[str]
            Ordered attribute names matching the KAN input.
        stats_path : str, Path, or None
            Explicit path to stats JSON, or None for auto-detection.
        device : str
            Torch device.

        Returns
        -------
        tuple
            ``(means, stds, stats_ranges)`` where means and stds are tensors of
            shape ``(n_attrs,)`` and stats_ranges is a dict of p10/p90 per attribute.
        """
        if stats_path is not None:
            json_path = Path(stats_path)
        else:
            # Auto-detect from config
            stats_dir = Path(cfg.data_sources.statistics)
            attr_source = Path(cfg.data_sources.attributes).name
            json_path = stats_dir / f"{cfg.geodataset.value}_attribute_statistics_{attr_source}.json"

        if not json_path.exists():
            msg = (
                f"Attribute statistics file not found: {json_path}. "
                f"Provide stats_path explicitly or run training first to generate statistics."
            )
            raise FileNotFoundError(msg)

        log.info("Loading normalization statistics from %s", json_path)
        with open(json_path) as f:
            stats = json.load(f)

        means_list = []
        stds_list = []
        ranges: dict[str, dict[str, float]] = {}
        for attr in attribute_names:
            if attr not in stats:
                msg = f"Attribute {attr!r} not found in statistics file {json_path}"
                raise KeyError(msg)
            means_list.append(float(stats[attr]["mean"]))
            stds_list.append(float(stats[attr]["std"]))
            ranges[attr] = {
                "p10": float(stats[attr]["p10"]),
                "p90": float(stats[attr]["p90"]),
            }

        means = torch.tensor(means_list, device=device, dtype=torch.float32)
        stds = torch.tensor(stds_list, device=device, dtype=torch.float32)

        return means, stds, ranges
