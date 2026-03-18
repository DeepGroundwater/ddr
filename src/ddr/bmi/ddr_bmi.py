"""BMI v2.0 wrapper for DDR differentiable Muskingum-Cunge routing.

Drop-in replacement for t-route in the NGWPC/ngen NextGen framework.
Uses identical CSDMS Standard Names for BMI variables so ngen's realization
config needs only ``python_type`` changed from t-route to DDR.

Example ngen realization config::

    {
        "routing": {
            "ddr_routing": {
                "name": "bmi_py",
                "params": {
                    "python_type": "ddr.bmi.DdrBmi",
                    "init_config": "config/bmi_ddr_routing.yaml",
                    "main_output_variable": "channel_exit_water_x-section__volume_flow_rate",
                },
            }
        }
    }
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from bmipy import Bmi
from omegaconf import OmegaConf

from ddr.bmi.config import BmiInitConfig
from ddr.nn import kan
from ddr.routing.mmc import MuskingumCunge, compute_hotstart_discharge
from ddr.scripts_utils import load_checkpoint
from ddr.validation.configs import validate_config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSDMS Standard Names (matching t-route for drop-in compatibility)
# ---------------------------------------------------------------------------
_INPUT_VAR_NAMES = (
    "land_surface_water_source__id",
    "land_surface_water_source__volume_flow_rate",
    "ngen_dt",
)

_OUTPUT_VAR_NAMES = (
    "channel_water__id",
    "channel_exit_water_x-section__volume_flow_rate",
    "channel_water_flow__speed",
    "channel_water__mean_depth",
)

_VAR_UNITS: dict[str, str] = {
    "land_surface_water_source__id": "-",
    "land_surface_water_source__volume_flow_rate": "m3 s-1",
    "ngen_dt": "s",
    "channel_water__id": "-",
    "channel_exit_water_x-section__volume_flow_rate": "m3 s-1",
    "channel_water_flow__speed": "m s-1",
    "channel_water__mean_depth": "m",
}

_VAR_TYPES: dict[str, str] = {
    "land_surface_water_source__id": "int32",
    "land_surface_water_source__volume_flow_rate": "float64",
    "ngen_dt": "int32",
    "channel_water__id": "int64",
    "channel_exit_water_x-section__volume_flow_rate": "float32",
    "channel_water_flow__speed": "float32",
    "channel_water__mean_depth": "float32",
}


class DdrBmi(Bmi):
    """BMI v2.0 wrapper for DDR differentiable Muskingum-Cunge routing.

    Designed for integration with the NGWPC/ngen NextGen framework as a
    routing module (replacing t-route). Operates on the FULL river network
    simultaneously via sparse matrix solve, not per-catchment.

    The KAN neural network runs once during ``initialize()`` to predict
    static spatial parameters (Manning's n, q_spatial). All subsequent
    ``update_until()`` calls perform inference-only MC routing with
    ``torch.no_grad()`` to avoid memory leaks from graph accumulation.

    Output arrays (discharge, velocity, depth) are stored as persistent
    numpy arrays and updated in-place after each ``route_timestep()`` call,
    following the NGWPC/lstm BMI pattern. This ensures ``get_value_ptr``
    returns stable references that reflect the latest state.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._cold_started = False

        # Config
        self._bmi_cfg: BmiInitConfig | None = None
        self._cfg: Any = None
        self._device: str = "cpu"

        # Routing engine
        self._mc: MuskingumCunge | None = None
        self._mapper: Any = None  # PatternMapper
        self._spatial_params: dict[str, torch.Tensor] | None = None
        self._num_segments: int = 0

        # ID mapping (nexus → segment index)
        self._nexus_to_seg_idx: dict[int, int] = {}
        self._segment_ids: np.ndarray = np.empty(0, dtype=np.int64)

        # Per-timestep state
        self._lateral_inflow: np.ndarray = np.empty(0, dtype=np.float64)
        self._prev_lateral_inflow: np.ndarray = np.empty(0, dtype=np.float64)
        self._has_prev_inflow: bool = False
        self._nexus_ids: np.ndarray = np.empty(0, dtype=np.int32)
        self._current_time: float = 0.0
        self._timestep: float = 3600.0
        self._ngen_dt: int = 3600
        self._interpolation: str = "constant"

        # Persistent output arrays (mutated in-place, safe for get_value_ptr)
        self._discharge: np.ndarray = np.empty(0, dtype=np.float32)
        self._velocity: np.ndarray = np.empty(0, dtype=np.float32)
        self._depth: np.ndarray = np.empty(0, dtype=np.float32)

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def initialize(self, config_file: str) -> None:
        """Initialize DDR routing from a YAML config file.

        Loads the DDR config, builds the network topology from the hydrofabric,
        runs KAN inference once for static spatial parameters, and initializes
        the MuskingumCunge engine.
        """
        # 1. Parse BMI init config
        raw = yaml.safe_load(Path(config_file).read_text())
        self._bmi_cfg = BmiInitConfig(**raw)
        self._device = self._bmi_cfg.device
        self._timestep = self._bmi_cfg.timestep_seconds
        self._interpolation = self._bmi_cfg.interpolation

        # 2. Load DDR config (without Hydra)
        omega_cfg = OmegaConf.load(str(self._bmi_cfg.ddr_config))
        omega_cfg.device = self._device
        omega_cfg.mode = "route"
        if self._bmi_cfg.hydrofabric_gpkg is not None:
            omega_cfg.data_sources.geospatial_fabric_gpkg = str(self._bmi_cfg.hydrofabric_gpkg)
        if self._bmi_cfg.conus_adjacency is not None:
            omega_cfg.data_sources.conus_adjacency = str(self._bmi_cfg.conus_adjacency)
        self._cfg = validate_config(omega_cfg, save_config=False)

        # 3. Build network topology from hydrofabric
        dataset = self._cfg.geodataset.get_dataset_class(cfg=self._cfg)
        routing_dc = dataset.routing_dataclass
        if routing_dc is None or routing_dc.adjacency_matrix is None:
            msg = "Failed to build routing dataclass from hydrofabric"
            raise RuntimeError(msg)
        self._num_segments = routing_dc.adjacency_matrix.shape[0]

        # Extract segment IDs for output
        if routing_dc.divide_ids is not None:
            raw_ids = routing_dc.divide_ids
            # Convert cat-{id} / wb-{id} strings to integer IDs
            self._segment_ids = np.array(
                [int(str(s).replace("cat-", "").replace("wb-", "")) for s in raw_ids],
                dtype=np.int64,
            )
        else:
            self._segment_ids = np.arange(self._num_segments, dtype=np.int64)

        # Build nexus→segment index mapping from hydrofabric GeoPackage
        gpkg_path = Path(self._cfg.data_sources.geospatial_fabric_gpkg)
        self._nexus_to_seg_idx = self._build_nexus_mapping(gpkg_path, routing_dc.divide_ids)

        # 4. Load KAN and run inference once for static spatial parameters
        nn_model = kan(
            input_var_names=self._cfg.kan.input_var_names,
            learnable_parameters=self._cfg.kan.learnable_parameters,
            hidden_size=self._cfg.kan.hidden_size,
            num_hidden_layers=self._cfg.kan.num_hidden_layers,
            grid=self._cfg.kan.grid,
            k=self._cfg.kan.k,
            seed=self._cfg.seed,
            device=self._device,
        )
        load_checkpoint(nn_model, self._bmi_cfg.kan_checkpoint, self._device)
        nn_model.eval()
        with torch.no_grad():
            attrs = routing_dc.normalized_spatial_attributes
            if attrs is None:
                msg = "No normalized spatial attributes in routing dataclass"
                raise RuntimeError(msg)
            self._spatial_params = nn_model(inputs=attrs.to(self._device))
        del nn_model  # Free KAN memory — only spatial params needed

        # 5. Initialize MuskingumCunge engine
        self._mc = MuskingumCunge(self._cfg, device=self._device)
        # Override timestep to match BMI config
        self._mc.t = torch.tensor(self._timestep, device=self._device)

        # Setup network context and denormalize parameters using a dummy inflow
        dummy_q_prime = torch.zeros(1, self._num_segments, device=self._device)
        self._mc.setup_inputs(
            routing_dataclass=routing_dc,
            streamflow=dummy_q_prime,
            spatial_parameters=self._spatial_params,
        )

        # Cache the PatternMapper (immutable for the network topology)
        self._mapper, _, _ = self._mc.create_pattern_mapper()

        # 6. Initialize persistent output and state arrays (in-place mutation)
        self._lateral_inflow = np.zeros(self._num_segments, dtype=np.float64)
        self._prev_lateral_inflow = np.zeros(self._num_segments, dtype=np.float64)
        self._has_prev_inflow = False
        self._nexus_ids = np.empty(0, dtype=np.int32)
        self._discharge = np.zeros(self._num_segments, dtype=np.float32)
        self._velocity = np.zeros(self._num_segments, dtype=np.float32)
        self._depth = np.zeros(self._num_segments, dtype=np.float32)
        self._current_time = 0.0
        self._cold_started = False
        self._initialized = True

        log.info(
            "DdrBmi initialized: %d segments, %d nexus mappings, device=%s, dt=%.0fs, interpolation=%s",
            self._num_segments,
            len(self._nexus_to_seg_idx),
            self._device,
            self._timestep,
            self._interpolation,
        )

    def update(self) -> None:
        """Advance the model by one timestep."""
        self.update_until(self._current_time + self._timestep)

    def update_until(self, time: float) -> None:
        """Advance the model to the given time.

        When ``timestep_seconds < ngen_dt``, multiple routing sub-steps are
        taken per ngen coupling interval. The ``interpolation`` config controls
        how lateral inflows are distributed across sub-steps:

        - ``"constant"``: same inflows for every sub-step (zeroth-order hold).
        - ``"linear"``: linearly interpolate from previous to current inflows.
          Falls back to constant on the first coupling interval (no previous
          data). Both methods conserve mass differently — see
          ``data/diagrams/bmi_testing_guide.txt``.
        """
        if not self._initialized or self._mc is None:
            msg = "Model not initialized. Call initialize() first."
            raise RuntimeError(msg)

        # Compute number of sub-steps for this coupling interval
        remaining = time - self._current_time
        if remaining <= 0.0:
            return  # No-op: don't consume inflows or update state
        n_steps = max(1, round(remaining / self._timestep))
        use_linear = self._interpolation == "linear" and self._has_prev_inflow and n_steps > 1

        for step in range(n_steps):
            if self._current_time >= time - 1e-6:
                break

            # Interpolate lateral inflow for this sub-step
            if use_linear:
                alpha = (step + 1) / n_steps
                inflow = (1.0 - alpha) * self._prev_lateral_inflow + alpha * self._lateral_inflow
            else:
                inflow = self._lateral_inflow

            q_prime = torch.tensor(inflow, dtype=torch.float32, device=self._device)

            # Cold-start on first real inflow
            if not self._cold_started:
                self._mc._discharge_t = compute_hotstart_discharge(
                    q_prime,
                    self._mapper,
                    self._mc.discharge_lb,
                    self._device,
                )
                self._cold_started = True

            # Clamp lateral inflow to minimum
            q_prime_clamp = torch.clamp(
                q_prime,
                min=self._cfg.params.attribute_minimums["discharge"],
            )

            # Route one timestep (no autograd — inference only)
            with torch.no_grad():
                q_t1 = self._mc.route_timestep(
                    q_prime_clamp=q_prime_clamp,
                    mapper=self._mapper,
                )

            # Update state
            self._mc._discharge_t = q_t1
            self._current_time += self._timestep

        # Cache outputs into persistent arrays (in-place for get_value_ptr stability)
        self._update_output_cache()

        # Store current inflows as previous for next linear interpolation
        self._prev_lateral_inflow[:] = self._lateral_inflow
        self._has_prev_inflow = True

        # Clear lateral inflows after routing (ngen re-sends each step)
        self._lateral_inflow[:] = 0.0

    def finalize(self) -> None:
        """Clean up model resources."""
        self._mc = None
        self._mapper = None
        self._spatial_params = None
        self._initialized = False
        log.info("DdrBmi finalized")

    # -----------------------------------------------------------------------
    # Variable info
    # -----------------------------------------------------------------------

    def get_component_name(self) -> str:
        return "DDR-MuskingumCunge"

    def get_input_item_count(self) -> int:
        return len(_INPUT_VAR_NAMES)

    def get_output_item_count(self) -> int:
        return len(_OUTPUT_VAR_NAMES)

    def get_input_var_names(self) -> tuple[str, ...]:
        return _INPUT_VAR_NAMES

    def get_output_var_names(self) -> tuple[str, ...]:
        return _OUTPUT_VAR_NAMES

    def get_var_grid(self, name: str) -> int:
        return 0

    def get_var_type(self, name: str) -> str:
        return _VAR_TYPES.get(name, "float64")

    def get_var_units(self, name: str) -> str:
        return _VAR_UNITS.get(name, "-")

    def get_var_itemsize(self, name: str) -> int:
        dtype = np.dtype(self.get_var_type(name))
        return int(dtype.itemsize)

    def get_var_nbytes(self, name: str) -> int:
        raise NotImplementedError

    def get_var_location(self, name: str) -> str:
        return "node"

    # -----------------------------------------------------------------------
    # Time
    # -----------------------------------------------------------------------

    def get_current_time(self) -> float:
        return self._current_time

    def get_start_time(self) -> float:
        return 0.0

    def get_end_time(self) -> float:
        return float("inf")  # ngen controls the simulation end time

    def get_time_units(self) -> str:
        return "s"

    def get_time_step(self) -> float:
        return self._timestep

    # -----------------------------------------------------------------------
    # Getters / Setters
    # -----------------------------------------------------------------------

    def get_value(self, name: str, dest: np.ndarray) -> np.ndarray:
        """Copy a BMI output variable into ``dest`` (follows NGWPC/lstm pattern)."""
        dest[:] = self.get_value_ptr(name)[: len(dest)]
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Return a reference to a variable's persistent numpy array.

        Arrays are updated in-place after each ``update_until()`` call,
        so pointers remain stable across the simulation lifetime
        (same contract as NGWPC/lstm's ``get_value_ptr``).
        """
        if name == "channel_exit_water_x-section__volume_flow_rate":
            return self._discharge
        if name == "channel_water__id":
            return self._segment_ids
        if name == "channel_water_flow__speed":
            return self._velocity
        if name == "channel_water__mean_depth":
            return self._depth
        msg = f"Unknown output variable: {name}"
        raise ValueError(msg)

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """Get values at specific indices."""
        dest[:] = self.get_value_ptr(name)[inds]
        return dest

    def set_value(self, name: str, src: np.ndarray) -> None:
        """Set a BMI input variable value."""
        if name == "land_surface_water_source__volume_flow_rate":
            # Remap nexus-indexed inflows to segment-indexed array
            if len(self._nexus_ids) > 0 and len(src) > 0:
                n_nexus = len(self._nexus_ids)
                flows = src.flat[:n_nexus]
                for i, nex_id in enumerate(self._nexus_ids):
                    seg_idx = self._nexus_to_seg_idx.get(int(nex_id))
                    if seg_idx is not None:
                        self._lateral_inflow[seg_idx] = flows[i]
            else:
                # Direct array assignment if no nexus IDs set
                n = min(len(src.flat), self._num_segments)
                self._lateral_inflow[:n] = src.flat[:n]
        elif name == "land_surface_water_source__id":
            self._nexus_ids = src.astype(np.int32).flatten()
        elif name == "ngen_dt":
            self._ngen_dt = int(src.flat[0])
        else:
            # BMI convention: unknown set_value calls should not crash
            log.debug("Unknown input variable ignored: %s", name)

    def set_value_at_indices(self, name: str, inds: np.ndarray, src: np.ndarray) -> None:
        """Set values at specific indices."""
        if name == "land_surface_water_source__volume_flow_rate":
            for i, idx in enumerate(inds):
                if idx < self._num_segments:
                    self._lateral_inflow[idx] = src[i]
        else:
            log.debug("set_value_at_indices not supported for: %s", name)

    # -----------------------------------------------------------------------
    # Grid info (unstructured network)
    # -----------------------------------------------------------------------

    def get_grid_rank(self, grid: int) -> int:
        return 1

    def get_grid_size(self, grid: int) -> int:
        return self._num_segments

    def get_grid_type(self, grid: int) -> str:
        return "unstructured"

    def get_grid_shape(self, grid: int, shape: np.ndarray) -> np.ndarray:
        shape[0] = self._num_segments
        return shape

    def get_grid_spacing(self, grid: int, spacing: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Spacing not defined for unstructured grid")

    def get_grid_origin(self, grid: int, origin: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Origin not defined for unstructured grid")

    def get_grid_x(self, grid: int, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Grid coordinates not available")

    def get_grid_y(self, grid: int, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Grid coordinates not available")

    def get_grid_z(self, grid: int, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Grid coordinates not available")

    def get_grid_node_count(self, grid: int) -> int:
        return self._num_segments

    def get_grid_edge_count(self, grid: int) -> int:
        if self._mc is not None and self._mc.network is not None:
            return int(self._mc.network._nnz())
        return 0

    def get_grid_face_count(self, grid: int) -> int:
        return 0

    def get_grid_edge_nodes(self, grid: int, edge_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_edges(self, grid: int, face_edges: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_face_nodes(self, grid: int, face_nodes: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_nexus_mapping(
        self,
        gpkg_path: Path,
        divide_ids: list[str] | np.ndarray | None,
    ) -> dict[int, int]:
        """Build nexus→segment-index mapping from the hydrofabric GeoPackage.

        Reads the ``flowpaths`` table (``id``, ``toid``) to find which nexus
        each flowpath drains into, then inverts to get nexus→flowpath.
        This mirrors the logic in ``engine/.../graph.py:preprocess_river_network``
        but uses stdlib ``sqlite3`` to avoid a polars dependency at runtime.

        Parameters
        ----------
        gpkg_path : Path
            Path to the hydrofabric GeoPackage.
        divide_ids : list or ndarray or None
            Ordered divide IDs (cat-{id}) from the RoutingDataclass.

        Returns
        -------
        dict[int, int]
            Mapping from nexus integer ID to segment array index.
        """
        # Build a lookup: segment integer ID → array index
        seg_id_to_idx: dict[int, int] = {}
        if divide_ids is not None:
            for idx, did in enumerate(divide_ids):
                seg_int = int(str(did).replace("cat-", "").replace("wb-", ""))
                seg_id_to_idx[seg_int] = idx
        else:
            for idx in range(self._num_segments):
                seg_id_to_idx[idx] = idx

        nexus_to_seg: dict[int, int] = {}
        try:
            con = sqlite3.connect(str(gpkg_path))
            # flowpaths table: each flowpath wb-{id} has toid = nex-{nexus_id}
            # Invert: nex-{nexus_id} → wb-{id} (the flowpath draining INTO that nexus)
            rows = con.execute("SELECT id, toid FROM flowpaths WHERE toid LIKE 'nex-%'").fetchall()
            con.close()

            for fp_id, nex_id in rows:
                # Extract integer IDs
                fp_str = str(fp_id)
                nex_str = str(nex_id)
                if not (fp_str.startswith("wb-") or fp_str.startswith("cat-")):
                    continue
                fp_int = int(fp_str.replace("wb-", "").replace("cat-", ""))
                nex_int = int(nex_str.replace("nex-", ""))
                seg_idx = seg_id_to_idx.get(fp_int)
                if seg_idx is not None:
                    nexus_to_seg[nex_int] = seg_idx

            log.info(
                "Built nexus mapping from GeoPackage: %d nexus→segment entries from %s",
                len(nexus_to_seg),
                gpkg_path.name,
            )
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            log.warning(
                "Could not read flowpaths table from %s; falling back to identity mapping",
                gpkg_path,
            )
            # Fallback: assume nexus ID == segment ID
            nexus_to_seg = {int(sid): idx for sid, idx in seg_id_to_idx.items()}

        return nexus_to_seg

    def _update_output_cache(self) -> None:
        """Update persistent output arrays from current MC routing state.

        Called after each ``update_until()`` to snapshot discharge, velocity,
        and depth into the numpy arrays that ``get_value_ptr`` returns.
        Uses in-place ``[:] =`` so pointers remain stable.
        """
        if self._mc is None:
            return

        # Discharge — direct copy from torch tensor
        if self._mc._discharge_t is not None:
            self._discharge[:] = self._mc._discharge_t.detach().cpu().numpy().astype(np.float32)

        # Velocity and depth — derived from MC engine's stored geometry
        # route_timestep already computed top_width/side_slope via Leopold & Maddock;
        # reuse those cached values rather than re-deriving.
        if (
            self._mc._discharge_t is not None
            and self._mc.n is not None
            and self._mc.slope is not None
            and self._mc.top_width is not None
            and self._mc.side_slope is not None
        ):
            with torch.no_grad():
                q_t = self._mc._discharge_t
                n = self._mc.n
                s0 = self._mc.slope
                tw = self._mc.top_width
                z = self._mc.side_slope
                q_eps = self._mc.q_spatial + 1e-6 if self._mc.q_spatial is not None else 1e-6
                p = self._mc.p_spatial

                # Depth from Leopold & Maddock inversion (same as _get_trapezoid_velocity)
                num = q_t * n * (q_eps + 1)
                den = p * torch.pow(s0, 0.5)
                depth = torch.pow(
                    torch.div(num, den + 1e-8),
                    torch.div(3.0, 5.0 + 3.0 * q_eps),
                )
                depth = torch.clamp(depth, min=self._mc.depth_lb)

                # Hydraulic radius from trapezoidal cross-section
                bw = torch.clamp(tw - 2 * z * depth, min=self._mc.bottom_width_lb)
                area = (tw + bw) * depth / 2
                wp = bw + 2 * depth * torch.sqrt(1 + z**2)
                R = area / wp

                # Manning's velocity
                v = (1.0 / n) * torch.pow(R, 2.0 / 3.0) * torch.pow(s0, 0.5)
                v = torch.clamp(v, min=0.0, max=15.0)

            self._velocity[:] = v.detach().cpu().numpy().astype(np.float32)
            self._depth[:] = depth.detach().cpu().numpy().astype(np.float32)
