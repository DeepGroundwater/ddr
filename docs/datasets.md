---
icon: lucide/database
---

# Streamflow Datasets & Specifications

DDR is designed to route lateral inflow from a large number of unit-catchments across many timesteps. To accommodate diverse data sources, DDR uses a uniform input specification.

## Overview

DDR requires three main types of input data:

1. **Lateral Inflow** (Q' or Q_l): Runoff predictions from unit catchments
2. **Geospatial Fabric**: River network topology and channel properties
3. **Observations**: Streamflow measurements for training/validation

## Lateral Inflow Specification

### Data Format

Lateral inflow data should be provided as an [Icechunk](https://icechunk.io/) store or zarr array with the following structure:

```python
import xarray as xr

# Expected dimensions and coordinates
ds = xr.Dataset(
    data_vars={
        "Qr": (["time", "divide_id"], qr_data),  # Lateral inflow in m³/s
    },
    coords={
        "time": time_index,           # Daily timestamps
        "divide_id": divide_ids,      # Catchment identifiers
    },
    attrs={
        "units": "m^3/s",
        "source": "your_model_name",
    }
)
```

### Requirements

| Property | Requirement |
|----------|-------------|
| Units | Cubic meters per second (m³/s) |
| Temporal resolution | Daily (interpolated to hourly internally) |
| Spatial coverage | All divide_ids in the routing domain |
| Missing values | Fill with small positive value (e.g., 1e-6) |
| Negative values | Not allowed |

### Unit Conversion

If your data is in mm/day, convert using drainage area:

```python
# Convert mm/day to m³/s
# mm/day * km² * 1000 / 86400 = m³/s
conversion_factor = area_km2 * 1000 / 86400
qr_m3_s = runoff_mm_day * conversion_factor
```

### Supported Data Sources

DDR provides download scripts for several pre-computed lateral inflow products:

| Source | Coverage | Period | Location |
|--------|----------|--------|----------|
| dHBV2.0 (Hydrofabric v2.2) | CONUS | 1980-2020 | `s3://mhpi-spatial/hydrofabric_v2.2_dhbv_retrospective` |
| dHBV2.0 (MERIT) | CONUS | 1980-2020 | [Zenodo](https://zenodo.org/records/15784945) |

## Geospatial Data Requirements

### Hydrofabric v2.2

The NOAA-OWP Hydrofabric v2.2 is the recommended geospatial dataset for CONUS applications.

**Required Layers:**

| Layer | Description |
|-------|-------------|
| `flowpaths` | River reaches with topology (id, toid) |
| `flowpath-attributes-ml` | Channel properties (length, slope, width) |
| `network` | Network connectivity including gauge locations |

**Required Attributes:**

```python
# Flowpath attributes
flowpath_attrs = [
    "id",              # Waterbody identifier (wb-XXXXX)
    "toid",            # Downstream identifier
    "Length_m",        # Channel length in meters
    "So",              # Channel slope (dimensionless)
    "TopWdth",         # Top width in meters
    "ChSlp",           # Channel side slope
    "MusX",            # Muskingum X parameter
]
```

### MERIT Hydro

MERIT Hydro provides global coverage with variable resolution.

**Required Attributes:**

| Attribute | Description |
|-----------|-------------|
| `COMID` | Unique catchment identifier |
| `NextDownID` | Downstream COMID |
| `up1`-`up4` | Upstream COMID connections |
| `lengthkm` | Channel length in kilometers |
| `slope` | Channel slope |
| `unitarea` | Catchment area in km² |

### Connectivity Format

DDR uses sparse COO (Coordinate) matrices to represent river network connectivity:

```python
# Adjacency matrix structure
# Rows: downstream segments
# Columns: upstream segments
# Values: 1 (connected) or 0 (not connected)

# Lower triangular: ensures topological ordering
# A[i, j] = 1 means flow goes from segment j to segment i
```

The matrices are stored in zarr v3 format following the [Binsparse COO specification](engine/binsparse.md). See the binsparse documentation for details on:

- Reading and writing adjacency matrices programmatically
- Order converters for MERIT and Lynker ID formats
- Per-gauge subset structure

The engine scripts automatically create these matrices:

```bash
# Creates:
# - hydrofabric_v2.2_conus_adjacency.zarr (full network)
# - hydrofabric_v2.2_gages_conus_adjacency.zarr (per-gauge subsets)

uv run python engine/scripts/build_hydrofabric_v2.2_matrices.py \
    /path/to/conus_nextgen.gpkg data/
```

## Catchment Attributes

The neural network requires catchment attributes to predict routing parameters.

### Default Attribute Set (Hydrofabric v2.2)

```yaml
input_var_names:
  - aridity               # Aridity index
  - meanelevation         # Mean elevation (m)
  - log_uparea            # Log10 of the upstream area for a catchment (m)
  - meanP                 # Mean Annual Precipitation (mm)
```

These attributes can be either calculated by hand, or provided for you through `s3://mhpi-spatial/hydrofabric_v2.2_attributes/` for the lynker hydrofabric. Catchment attributes are geodataset specific

### Attribute Storage

Attributes should be stored in an Icechunk/zarr store if possible. NetCDF support is used in MERIT:

```python
# Hydrofabric v2.2 format
ds = xr.Dataset(
    data_vars={
        "aridity": (["divide_id"], aridity_values),
        "elev_mean": (["divide_id"], elev_values),
        # ... other attributes
    },
    coords={
        "divide_id": divide_ids,  # Format: "cat-XXXXX"
    }
)
```

### Normalization

DDR automatically computes and caches normalization statistics:

```python
# Statistics stored per attribute
{
    "min": min_value,
    "max": max_value,
    "mean": mean_value,
    "std": std_value,
    "p10": 10th_percentile,
    "p90": 90th_percentile,
}
```

Statistics are cached to `data/statistics/{geodataset}_attribute_statistics_{store_name}.json`.

## Observations

### USGS Streamflow Data

DDR uses USGS streamflow observations for training and validation.

**Data Format:**

```python
ds = xr.Dataset(
    data_vars={
        "streamflow": (["time", "gage_id"], flow_values),  # m³/s
    },
    coords={
        "time": time_index,
        "gage_id": gage_ids,  # 8-digit zero-padded strings
    }
)
```

**Pre-formatted Observations:**

DDR provides access to pre-processed USGS data:

```yaml
observations: "s3://mhpi-spatial/usgs_streamflow_observations/"
```

### Gauge Information CSV

Training requires a gauge information file:

```csv
STAID,STANAME,DRAIN_SQKM,LAT_GAGE,LNG_GAGE
01563500,Susquehanna River at Harrisburg,62419,40.2548,-76.8867
01570500,Susquehanna River at Sunbury,46706,40.8576,-76.7944
```

**Required Columns:**

| Column | Description | Format |
|--------|-------------|--------|
| `STAID` | Station ID | 8-digit, zero-padded |
| `DRAIN_SQKM` | Drainage area | km² (positive float) |
| `LAT_GAGE` | Latitude | Decimal degrees |
| `LNG_GAGE` | Longitude | Decimal degrees |

**Optional Columns:**

| Column | Description |
|--------|-------------|
| `STANAME` | Station name |
| `COMID` | MERIT catchment ID (required for MERIT dataset) |

Pre-prepared gauge lists are available in `references/gage_info/`:

| File | Gages | Source |
|------|-------|--------|
| `camels_670.csv` | 670 | CAMELS / HCDN-2009 ([Newman et al., 2014](https://dx.doi.org/10.5065/D6MW2F4D)) |
| `gages_3000.csv` | 3211 | [Ouyang et al., 2021](https://doi.org/10.1016/j.jhydrol.2021.126455) |
| `GAGES-II.csv` | 8945 | GAGES-II ([Falcone, 2011](https://doi.org/10.3133/70046617)) |

See `references/gage_info/README.md` for derivation details.

## Data Sources

### Pre-computed S3 Data

DDR provides access to pre-computed datasets on AWS S3:

| Dataset | S3 Path | Description |
|---------|---------|-------------|
| HF v2.2 Attributes | `s3://mhpi-spatial/hydrofabric_v2.2_attributes/` | Catchment attributes |
| HF v2.2 Streamflow | `s3://mhpi-spatial/hydrofabric_v2.2_dhbv_retrospective` | dHBV2.0 predictions |
| USGS Observations | `s3://mhpi-spatial/usgs_streamflow_observations/` | Historical streamflow |

Access is anonymous (no AWS credentials required):

```python
from ddr.io.readers import read_ic

# Read from S3
ds = read_ic("s3://mhpi-spatial/hydrofabric_v2.2_attributes/", region="us-east-2")
```

### Local Data

For local data, use file paths:

```yaml
data_sources:
  attributes: "/path/to/local/attributes/"
  streamflow: "/path/to/local/streamflow/"
```

## Preparing Custom Data

### Creating Lateral Inflow Data

If you have your own runoff model, format the output for DDR:

```python
import icechunk as ic
import xarray as xr
from icechunk.xarray import to_icechunk

# Load your model output
qr = load_your_model_output()  # shape: (n_catchments, n_timesteps)

# Create xarray Dataset
ds = xr.Dataset(
    data_vars={
        "Qr": (["divide_id", "time"], qr.astype(np.float32)),
    },
    coords={
        "divide_id": your_divide_ids,
        "time": pd.date_range("1980-01-01", periods=n_timesteps, freq="D"),
    },
    attrs={"units": "m^3/s"},
)

# Save to Icechunk
storage = ic.local_filesystem_storage("./my_streamflow_data")
repo = ic.Repository.create(storage)
session = repo.writable_session("main")
to_icechunk(ds, session)
session.commit("Initial commit")
```

## External Resources

- **Gauge Lists**: [DeepGroundwater/datasets](https://github.com/DeepGroundwater/ddr/references/gage_info/dhbv2_gages.csv)
- **MERIT Hydro**: [University of Tokyo](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/)
