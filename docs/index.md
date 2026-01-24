# What is DDR?
Distributed Differentiable Routing (DDR) is a end-to-end differentiable Muskingum Cunge flow router (`src/ddr`) + geospatial scaffolding/wrapper (`/engine`) for building graphs from geospatial fabrics. This work is brought to you by contributers/developers of the $\delta$MC, [$\delta$MC-Juniata-hydroDL2](https://github.com/mhpi/dMC-Juniata-hydroDL2), and [T-Route](https://github.com/NOAA-OWP/t-route). The goal of this project is to provide an open-sourced, easy to follow, routing module that can be applied to a wide variety of geospatial flow networks and input lateral flow datasets.

The goal of this project is to provide an open-sourced, easy to follow, routing module that can be applied to a wide variety of geospatial flow networks and input lateral flow datasets.

## Why DDR?

Traditional river routing models require manual calibration of parameters like Manning's roughness coefficient across thousands of river segments. DDR solves this by using **differentiable programming** - the routing equations are implemented in a way that allows automatic computation of gradients through the entire network, enabling neural networks to learn optimal parameters from observed streamflow data.

### Key Features

- **Differentiable Muskingum-Cunge Routing**: Physics-based routing with learnable parameters
- **Neural Network Parameter Learning**: Uses Kolmogorov-Arnold Networks (KAN) to predict Manning's roughness and channel geometry from catchment attributes
- **GPU Acceleration**: Sparse matrix operations optimized for both CPU and GPU
- **Multiple Geospatial Datasets**: Support for NOAA-OWP Hydrofabric v2.2 and MERIT Hydro
- **Scalable Architecture**: Route across thousands of river segments simultaneously

## How It Works

DDR combines physics-based river routing with machine learning:

```
Catchment Attributes → Neural Network → Physical Parameters → Muskingum-Cunge Routing → Streamflow
        ↑                                                                                  ↓
        └────────────────────── Gradient Backpropagation ──────────────────────────────────┘
```

1. **Input**: Lateral inflow predictions (Q') from unit catchments
2. **Parameter Learning**: A neural network predicts Manning's roughness (n) and other parameters from catchment attributes
3. **Routing**: The Muskingum-Cunge equations route flow downstream through the river network
4. **Training**: Gradients flow backward through the entire system to optimize the neural network

<div class="grid cards" markdown>

- :fontawesome-solid-microchip: [__Model Training__](../usage/train.md) for how to create your own weights/states
- :fontawesome-solid-laptop-code: [__Model Testing__](../usage/test.md) for evaluating your trained weights
- :fontawesome-solid-terminal: [__Routing__](../usage/routing.md) for how to route flow anywhere with trained weights
- :fontawesome-solid-gears: [__Summed Q_Prime__](../usage/summed_q_prime.md) for determining how well your unit catchment predictions are (pre-routing)

</div>

## Supported Geospatial Datasets

DDR currently supports two major hydrographic datasets:

| Dataset | Coverage | Resolution | Status |
|---------|----------|------------|--------|
| [NOAA-OWP Hydrofabric v2.2](https://www.noaa.gov/organization/information-technology/noaa-hydrofabric) | CONUS | ~37 km² catchments | Fully supported |
| [MERIT Hydro](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/) | Global | Variable | Fully supported |

## Architecture Overview

```
ddr/
├── src/ddr/                    # Core routing library
│   ├── routing/                # Muskingum-Cunge implementation
│   │   ├── mmc.py              # Core routing math
│   │   ├── torch_mc.py         # PyTorch nn.Module wrapper
│   │   └── utils.py            # Sparse solver utilities
│   ├── nn/                     # Neural network modules
│   │   └── kan.py              # Kolmogorov-Arnold Network
│   ├── geodatazoo/             # Dataset classes
│   │   ├── lynker_hydrofabric.py
│   │   └── merit.py
│   ├── io/                     # Data I/O utilities
│   └── validation/             # Metrics and plotting
├── engine/                     # Geospatial data preparation
│   └── scripts/                # Matrix building scripts
└── streamflow_datasets/        # Dataset download utilities
```

## Citation

If you use DDR in your research, please cite:

```bibtex
@article{bindas2024improving,
  author = {Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and others},
  title = {Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning},
  journal = {Water Resources Research},
  volume = {60},
  number = {1},
  year = {2024},
  doi = {https://doi.org/10.1029/2023WR035337}
}
```

A newer citation is in the works

## Community

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Contributing**: We welcome contributions! See our contributing guide.

## License

DDR is released under the Apache v2.0 License. See the [LICENSE](https://github.com/DeepGroundwater/ddr/blob/main/LICENSE) file for details.
