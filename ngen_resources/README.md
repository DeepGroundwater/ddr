To use in the NextGen Framework,
1) Copy the contentx of the DDR repo into `ngen/extern/ddr/ddr/`.
2) The contents of `ngen_resources/data/` should be copied into the ngen repo at `ngen/data/`. This contains data for the model, realizations for NextGen, and other config files enabling the DDR package to be run from within NextGen.

In particular, `ngen_resources/data/` contains:
- "Realization config" files for NextGen in `realization/`,
- Catchment and nexus data GeoJSON files in `spatial/`,
- DDR model config files in `config/`,
- Forcing data for testing AORC on the Juniata River Basin (JRB), etc. in `forcing/`
