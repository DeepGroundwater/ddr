---
icon: lucide/binary
---

# Binsparse COO Format

DDR uses a zarr-based storage format for sparse COO (Coordinate) matrices, inspired by the [binsparse specification](https://graphblas.org/binsparse-specification/) and [binsparse-python](https://github.com/ivirshup/binsparse-python). This format efficiently stores river network connectivity for routing computations.

## Format Overview

Each adjacency matrix is stored as a zarr v3 group containing arrays and metadata attributes.

### Arrays

| Array | Type | Description |
|-------|------|-------------|
| `indices_0` | int32 | Row indices (downstream segment indices) |
| `indices_1` | int32 | Column indices (upstream segment indices) |
| `values` | uint8 | Matrix values (1 for connected, 0 otherwise) |
| `order` | int32 | Topological sort order as domain-specific IDs |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `format` | str | Always "COO" |
| `shape` | [int, int] | Matrix dimensions [rows, cols] |
| `geodataset` | str | Geodataset type (e.g., "merit", "lynker") for auto-detection |
| `data_types` | dict | Dtype strings for each array |
| `gage_catchment` | int/str | Origin catchment ID (gauge subsets only) |
| `gage_idx` | int | CONUS matrix index (gauge subsets only) |

## Matrix Structure

The adjacency matrix is **lower triangular**, where `A[i, j] = 1` indicates that flow goes from segment `j` (column) to segment `i` (row). This structure ensures topological ordering: upstream segments always have lower indices than downstream segments.

```
     0  1  2  3  4   (upstream)
   ┌───────────────┐
 0 │ 0             │   Flow direction: column → row
 1 │ 1  0          │   Example: A[1,0]=1 means 0→1
 2 │ 0  1  0       │            A[2,1]=1 means 1→2
 3 │ 0  0  1  0    │            A[4,3]=1 means 3→4
 4 │ 0  0  1  1  0 │            A[4,2]=1 means 2→4
   └───────────────┘
(downstream)
```

## Geodataset Types

Different geodatasets use different ID formats. The `geodataset` attribute stored in zarr metadata enables automatic detection when reading.

### Supported Geodatasets

| Name | ID Format | Example IDs |
|------|-----------|-------------|
| `merit` | Integer COMIDs | `12345`, `12346`, `12347` |
| `lynker` | String wb-* IDs | `"wb-123"`, `"wb-456"` |
| `hydrofabric_v2.2` | Alias for `lynker` | Same as lynker |

### Listing Available Geodatasets

```python
from ddr_engine import list_geodatasets

print(list_geodatasets())  # ['hydrofabric_v2.2', 'lynker', 'merit']
```

### Registering Custom Geodatasets

```python
from ddr_engine import register_converter

class MyConverter:
    def to_zarr(self, ids):
        return np.array(ids, dtype=np.int32)
    def from_zarr(self, order):
        return order.tolist()

register_converter("my_geodataset", MyConverter())
```

## Reading Adjacency Matrices

### Auto-Detection (Recommended)

The simplest way to read a COO matrix - the geodataset type is automatically detected from metadata:

```python
from pathlib import Path
from ddr_engine import coo_from_zarr

# Auto-detects hydrofabric from metadata
coo, ts_order = coo_from_zarr(Path("data/merit_conus_adjacency.zarr"))

# coo: scipy.sparse.coo_matrix
# ts_order: list of domain-specific IDs (int for MERIT, str for Lynker)
```

### Dataset-Specific Functions

For type-hinted return values, use the dataset-specific functions:

```python
from pathlib import Path
from ddr_engine.merit.io import coo_from_zarr

# MERIT - returns COMIDs as integers
coo, ts_order = coo_from_zarr(Path("data/merit_conus_adjacency.zarr"))
# ts_order: list[int]

from ddr_engine.lynker_hydrofabric.io import coo_from_zarr

# Lynker - returns wb-* strings
coo, ts_order = coo_from_zarr(Path("data/hydrofabric_v2.2_conus_adjacency.zarr"))
# ts_order: list[str]
```

### Reading Gauge Subsets

Gauge subsets are stored in a zarr group with one subgroup per gauge:

```python
import zarr

# Open the gauge zarr store
root = zarr.open_group("data/merit_gages_conus_adjacency.zarr", mode="r")

# Each gauge is a subgroup keyed by station ID
gauge_group = root["01570500"]

# Access arrays
row = gauge_group["indices_0"][:]
col = gauge_group["indices_1"][:]
data = gauge_group["values"][:]
order = gauge_group["order"][:]

# Access metadata
shape = tuple(gauge_group.attrs["shape"])
gage_catchment = gauge_group.attrs["gage_catchment"]
gage_idx = gauge_group.attrs["gage_idx"]
```

## Writing Adjacency Matrices

### CONUS Full Network

```python
from pathlib import Path
from scipy import sparse
from ddr_engine import coo_to_zarr

# Create a COO matrix (example)
row = [1, 2, 3, 4, 4]
col = [0, 1, 2, 2, 3]
data = [1, 1, 1, 1, 1]
coo = sparse.coo_matrix((data, (row, col)), shape=(5, 5), dtype="uint8")

# Topological order as COMIDs
ts_order = [12345, 12346, 12347, 12348, 12349]

# Write to zarr - pass geodataset name
coo_to_zarr(coo, ts_order, Path("output/merit_conus_adjacency.zarr"), "merit")
```

### Gauge Subsets

```python
import zarr
from ddr_engine import coo_to_zarr_group

# Create/open the gauge zarr store
store = zarr.storage.LocalStore(root="output/merit_gages_adjacency.zarr")
root = zarr.create_group(store=store)

# Create a subgroup for each gauge
gauge_group = root.create_group("01570500")

# Write the subset COO matrix - pass geodataset name
coo_to_zarr_group(
    coo=subset_coo,
    ts_order=[12345, 12346],  # COMIDs in subset
    origin=12346,  # Gauge catchment COMID
    gauge_root=gauge_group,
    mapping={12345: 0, 12346: 1},  # COMID → CONUS index
    geodataset="merit",
)
```

## File Structure

### CONUS Network

```
merit_conus_adjacency.zarr/
├── zarr.json              # Group metadata
├── indices_0/             # Row indices
│   ├── zarr.json
│   └── c/0
├── indices_1/             # Column indices
│   ├── zarr.json
│   └── c/0
├── values/                # Matrix values
│   ├── zarr.json
│   └── c/0
└── order/                 # Topological order
    ├── zarr.json
    └── c/0
```

### Per-Gauge Subsets

```
merit_gages_conus_adjacency.zarr/
├── zarr.json              # Root group metadata
├── 01570500/              # Gauge station ID
│   ├── zarr.json          # Subgroup metadata with geodataset, gage_catchment, gage_idx
│   ├── indices_0/
│   ├── indices_1/
│   ├── values/
│   └── order/
├── 01563500/
│   └── ...
└── ...
```

## Creating Adjacency Matrices

The engine provides scripts to build adjacency matrices from raw hydrofabric data:

### MERIT Hydro

```bash
uv run python -m ddr_engine.merit /path/to/riv_pfaf_X_MERIT_Hydro.shp \
    --path data/ \
    --gages references/gage_info/dhbv2_gages.csv
```

### Lynker Hydrofabric v2.2

```bash
uv run python -m ddr_engine.lynker_hydrofabric /path/to/conus_nextgen.gpkg \
    --path data/ \
    --gages references/gage_info/dhbv2_gages.csv
```

Both commands create:
- `*_conus_adjacency.zarr`: Full CONUS river network
- `*_gages_conus_adjacency.zarr`: Per-gauge upstream subnetworks

## API Reference

### Primary Functions (Recommended)

::: ddr_engine.core.coo_to_zarr
::: ddr_engine.core.coo_from_zarr
::: ddr_engine.core.coo_to_zarr_group

### Converter Registry

::: ddr_engine.core.get_converter
::: ddr_engine.core.register_converter
::: ddr_engine.core.list_geodatasets

### Generic Functions (Low-Level)

::: ddr_engine.core.coo_to_zarr_generic
::: ddr_engine.core.coo_from_zarr_generic
::: ddr_engine.core.coo_to_zarr_group_generic
