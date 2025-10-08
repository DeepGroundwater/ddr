"""NextGen network topology and geometry provider."""

from pathlib import Path

import geopandas as gpd
import numpy as np
from scipy import sparse


class NextGenNetworkProvider:
    """Provider for NextGen Hydrofabric network.

    Handles:
    - Loading geopackage flowpath attributes
    - Building topology from zarr adjacency matrix
    - Extracting channel geometry
    - Mapping wb-XXX (waterbodies) to cat-XXX (catchments)

    Parameters
    ----------
    gpkg_path : Path
        Path to NextGen geopackage (e.g., conus_nextgen.gpkg)
    adjacency_path : Path
        Path to zarr store with pre-computed adjacency matrix
        (created by engine/adjacency.py)
    gages_adjacency_path : Path, optional
        Path to gauge-specific adjacency matrices
        (created by engine/gages_adjacency.py)

    Examples
    --------
    >>> provider = NextGenNetworkProvider(
    ...     gpkg_path=Path("data/conus_nextgen.gpkg"),
    ...     adjacency_path=Path("data/conus_adjacency.zarr"),
    ... )
    >>>
    >>> # Get full CONUS network
    >>> topology = provider.get_topology()
    >>>
    >>> # Get subset for specific reaches
    >>> subset_topology = provider.get_topology(["wb-1", "wb-2"])
    >>>
    >>> # Get channel geometry
    >>> geometry = provider.get_geometry(["wb-1", "wb-2"])
    >>> print(f"Length: {geometry.length}")
    """

    def __init__(self, gpkg_path: Path, adjacency_path: Path, gages_adjacency_path: Path | None = None):
        self.gpkg_path = gpkg_path
        self.adjacency_path = adjacency_path
        self.gages_adjacency_path = gages_adjacency_path

        self._load_data()

    def _load_data(self):
        """Load NextGen hydrofabric data."""
        # Load flowpath attributes from geopackage
        self.flowpath_attr = gpd.read_file(self.gpkg_path, layer="flowpath-attributes-ml").set_index("id")

        # Remove duplicates (if any)
        self.flowpath_attr = self.flowpath_attr[~self.flowpath_attr.index.duplicated(keep="first")]

        # Load CONUS adjacency matrix
        self.conus_adjacency = read_zarr(self.adjacency_path)
        self.hf_ids = self.conus_adjacency["order"][:]

        # Load gauge adjacency if provided
        if self.gages_adjacency_path:
            self.gages_adjacency = read_zarr(self.gages_adjacency_path)
        else:
            self.gages_adjacency = None

        self._build_topology()

    def _build_topology(self):
        """Build sparse adjacency matrix."""
        rows = self.conus_adjacency["indices_0"][:].tolist()
        cols = self.conus_adjacency["indices_1"][:].tolist()
        shape = tuple(dict(self.conus_adjacency.attrs)["shape"])

        self.full_topology = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=shape).tocsr()

        self.id_to_idx = {f"wb-{_id}": i for i, _id in enumerate(self.hf_ids)}

    def get_topology(self, reach_ids: list[str] | None = None) -> sparse.csr_matrix:
        """Get network topology as sparse adjacency matrix."""
        if reach_ids is None:
            return self.full_topology

        indices = [self.id_to_idx[rid] for rid in reach_ids if rid in self.id_to_idx]
        return self.full_topology[indices, :][:, indices]

    def get_reach_ids(self) -> list[str]:
        """Get all reach IDs in topological order."""
        return [f"wb-{_id}" for _id in self.hf_ids]

    def get_geometry(self, reach_ids: list[str]) -> ChannelGeometry:
        """Get channel geometry from geopackage."""
        subset = self.flowpath_attr.reindex(reach_ids)

        return ChannelGeometry(
            length=subset["Length_m"].fillna(1000.0).values,
            slope=subset["So"].fillna(0.001).values,
            top_width=subset["TopWdth"].fillna(10.0).values,
            side_slope=subset["ChSlp"].fillna(2.0).values,
            x=subset["MusX"].fillna(0.2).values,
        )

    def get_catchment_ids(self, reach_ids: list[str]) -> list[str]:
        """Map waterbody IDs to catchment IDs.

        NextGen convention: wb-XXX → cat-XXX
        """
        return [rid.replace("wb-", "cat-") for rid in reach_ids]

    def find_upstream_reaches(self, outlet_reach_id: str) -> list[str]:
        """Find all reaches upstream of outlet.

        Uses gauge adjacency if available, otherwise does BFS on full topology.
        """
        # Try gauge adjacency first (pre-computed, faster)
        if self.gages_adjacency:
            gauge_id = outlet_reach_id.replace("wb-", "")
            if gauge_id in self.gages_adjacency:
                order = self.gages_adjacency[gauge_id]["order"][:]
                return [f"wb-{_id}" for _id in order]

        # Fallback to BFS on full topology
        outlet_idx = self.id_to_idx.get(outlet_reach_id)
        if outlet_idx is None:
            return []

        visited = set()
        queue = [outlet_idx]
        upstream_indices = []

        while queue:
            idx = queue.pop(0)
            if idx in visited:
                continue
            visited.add(idx)
            upstream_indices.append(idx)

            # Find upstream reaches
            upstream = self.full_topology[idx, :].nonzero()[1]
            queue.extend(upstream)

        return [f"wb-{self.hf_ids[i]}" for i in upstream_indices]
