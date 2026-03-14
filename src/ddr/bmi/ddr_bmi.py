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

        # Cached output arrays (avoid re-allocation per get_value call)
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
            # Convert cat-{id} strings to integer IDs
            self._segment_ids = np.array(
                [int(str(s).replace("cat-", "").replace("wb-", "")) for s in raw_ids],
                dtype=np.int64,
            )
        else:
            self._segment_ids = np.arange(self._num_segments, dtype=np.int64)

        # Build nexus→segment index mapping
        # Nexus nex-{id} feeds into segment cat-{id} (downstream flowpath)
        self._nexus_to_seg_idx = {}
        for idx, seg_id in enumerate(self._segment_ids):
            # In ngen, nex-{id} flows into the downstream cat-{id}
            # t-route uses downstream_flowpath_dict for this mapping
            # For now, assume nexus ID == segment ID (common in NextGen hydrofabric)
            self._nexus_to_seg_idx[int(seg_id)] = idx

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

        # 6. Initialize per-timestep arrays
        self._lateral_inflow = np.zeros(self._num_segments, dtype=np.float64)
        self._prev_lateral_inflow = np.zeros(self._num_segments, dtype=np.float64)
        self._has_prev_inflow = False
        self._nexus_ids = np.empty(0, dtype=np.int32)
        self._velocity = np.zeros(self._num_segments, dtype=np.float32)
        self._depth = np.zeros(self._num_segments, dtype=np.float32)
        self._current_time = 0.0
        self._cold_started = False
        self._initialized = True

        log.info(
            "DdrBmi initialized: %d segments, device=%s, dt=%.0fs, interpolation=%s",
            self._num_segments,
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
        return self.get_var_itemsize(name) * self._num_segments

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
        """Get a BMI output variable value."""
        if self._mc is None:
            msg = "Model not initialized"
            raise RuntimeError(msg)

        if name == "channel_exit_water_x-section__volume_flow_rate":
            discharge = self._mc._discharge_t
            if discharge is not None:
                dest[:] = discharge.cpu().numpy().astype(np.float32)[: len(dest)]
        elif name == "channel_water__id":
            dest[:] = self._segment_ids[: len(dest)]
        elif name == "channel_water_flow__speed":
            # Velocity is derived from the last route_timestep call
            # _get_trapezoid_velocity stores celerity, not raw velocity
            # We approximate: v ≈ celerity * 3/5
            if self._mc.top_width is not None and self._mc._discharge_t is not None:
                # Re-derive velocity from Manning's equation
                v = self._compute_velocity()
                dest[:] = v[: len(dest)]
            else:
                dest[:] = 0.0
        elif name == "channel_water__mean_depth":
            depth = self._compute_depth()
            dest[:] = depth[: len(dest)]
        else:
            msg = f"Unknown output variable: {name}"
            raise ValueError(msg)
        return dest

    def get_value_ptr(self, name: str) -> np.ndarray:
        """Get a reference to a BMI variable's data."""
        if name == "channel_exit_water_x-section__volume_flow_rate":
            if self._mc is not None and self._mc._discharge_t is not None:
                return self._mc._discharge_t.cpu().numpy().astype(np.float32)
        elif name == "channel_water__id":
            return self._segment_ids
        msg = f"get_value_ptr not supported for: {name}"
        raise NotImplementedError(msg)

    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """Get values at specific indices."""
        full = np.empty(self._num_segments, dtype=np.dtype(self.get_var_type(name)))
        self.get_value(name, full)
        dest[:] = full[inds]
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

    def _compute_velocity(self) -> np.ndarray:
        """Derive flow velocity from current state using Manning's equation."""
        if self._mc is None or self._mc._discharge_t is None:
            return np.zeros(self._num_segments, dtype=np.float32)
        with torch.no_grad():
            n = self._mc.n
            s0 = self._mc.slope
            if n is None or s0 is None:
                return np.zeros(self._num_segments, dtype=np.float32)
            R_approx = self._compute_hydraulic_radius()
            v = (1.0 / n) * torch.pow(R_approx, 2.0 / 3.0) * torch.pow(s0, 0.5)
            v = torch.clamp(v, min=0.0, max=15.0)
        return v.cpu().numpy().astype(np.float32)

    def _compute_depth(self) -> np.ndarray:
        """Derive flow depth from current discharge and channel parameters."""
        if self._mc is None or self._mc._discharge_t is None or self._mc.n is None or self._mc.slope is None:
            return np.zeros(self._num_segments, dtype=np.float32)
        with torch.no_grad():
            q_t = self._mc._discharge_t
            n = self._mc.n
            s0 = self._mc.slope
            q_eps = self._mc.q_spatial + 1e-6 if self._mc.q_spatial is not None else 1e-6
            p = self._mc.p_spatial
            num = q_t * n * (q_eps + 1)
            den = p * torch.pow(s0, 0.5)
            depth = torch.pow(
                torch.div(num, den + 1e-8),
                torch.div(3.0, 5.0 + 3.0 * q_eps),
            )
            depth = torch.clamp(depth, min=self._mc.depth_lb)
        return depth.cpu().numpy().astype(np.float32)

    def _compute_hydraulic_radius(self) -> torch.Tensor:
        """Approximate hydraulic radius from stored geometry."""
        if self._mc is None or self._mc._discharge_t is None:
            return torch.zeros(self._num_segments)
        depth = torch.tensor(self._compute_depth(), dtype=torch.float32, device=self._device)
        if self._mc.top_width is not None and self._mc.side_slope is not None:
            tw = self._mc.top_width
            z = self._mc.side_slope
            bw = torch.clamp(tw - 2 * z * depth, min=self._mc.bottom_width_lb)
            area = (tw + bw) * depth / 2
            wp = bw + 2 * depth * torch.sqrt(1 + z**2)
            return area / wp
        # Fallback: wide channel approximation R ≈ depth
        return depth
