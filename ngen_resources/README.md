To use in the NextGen Framework,
1) Copy the content of the DDR repo into `ngen/extern/ddr/ddr/`.
2) The contents of `ngen_resources/data/` should be copied into the ngen repo at `ngen/data/`. This contains data for the model, realizations for NextGen, and other config files enabling the DDR package to be run from within NextGen.

In particular, `ngen_resources/data/` contains:
- DDR model and BMI configuration files in `config/`,
- Pretrained model weights for DDR in `models/`,
- "Realization" configuration files for NextGen in `realization/`,
- CONUS-scale statistics for static river attributes, COO transition matrices for network adjacency, and a subset (Juniata River Basin) of the NextGen hydrofabric v2.2 in `spatial/`,
- AORC forcing data (runoff) for NextGen + DDR forward inference on the Juniata River Basin in `forcing/`.
