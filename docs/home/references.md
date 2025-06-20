# References

## Previous Publications

Below are publications which were directly used to build DDR.

### Song and Bindas et al. (2024)

**Title:** High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning

**Open Sourced Code:** [dHBV2.0](https://github.com/mhpi/dHBV2.0)

**Citation:**
```bibtex
@article{song_high-resolution_2025,
	title = {High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-informed machine learning},
	volume = {61},
	copyright = {© 2025 The Author(s).},
	issn = {1944-7973},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1029/2024WR038928},
	doi = {10.1029/2024WR038928},
	abstract = {The National Water Model (NWM) is a key tool for flood forecasting, planning, and water management. Key challenges facing the NWM include calibration and parameter regionalization when confronted with big data. We present two novel versions of high-resolution (∼37 km2) differentiable models (a type of hybrid model): one with implicit, unit-hydrograph-style routing and another with explicit Muskingum-Cunge routing in the river network. The former predicts streamflow at basin outlets whereas the latter presents a discretized product that seamlessly covers rivers in the conterminous United States (CONUS). Both versions use neural networks to provide a multiscale parameterization and process-based equations to provide a structural backbone, which were trained simultaneously (“end-to-end”) on 2,807 basins across the CONUS and evaluated on 4,997 basins. Both versions show great potential to elevate future NWM performance for extensively calibrated as well as ungauged sites: the median daily Nash-Sutcliffe efficiency of all 4,997 basins is improved to around 0.68 from 0.48 of NWM3.0. As they resolve spatial heterogeneity, both versions greatly improved simulations in the western CONUS and also in the Prairie Pothole Region, a long-standing modeling challenge. The Muskingum-Cunge version further improved performance for basins {\textgreater}10,000 km2. Overall, our results show how neural-network-based parameterizations can improve NWM performance for providing operational flood predictions while maintaining interpretability and multivariate outputs. The modeling system supports the Basic Model Interface (BMI), which allows seamless integration with the next-generation NWM. We also provide a CONUS-scale hydrologic data set for further evaluation and use.},
	language = {en},
	number = {4},
	urldate = {2025-04-11},
	journal = {Water Resources Research},
	author = {Song, Yalan and Bindas, Tadd and Shen, Chaopeng and Ji, Haoyu and Knoben, Wouter J. M. and Lonzarich, Leo and Clark, Martyn P. and Liu, Jiangtao and van Werkhoven, Katie and Lamont, Sam and Denno, Matthew and Pan, Ming and Yang, Yuan and Rapp, Jeremy and Kumar, Mukesh and Rahmani, Farshid and Thébault, Cyril and Adkins, Richard and Halgren, James and Patel, Trupesh and Patel, Arpita and Sawadekar, Kamlesh Arun and Lawson, Kathryn},
	year = {2025},
	note = {\_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1029/2024WR038928
tex.ids= song2024highresolution},
	keywords = {*CIROH Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *CIROH subaward A23-0271-S001 from Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *DoE DE-SC0016605 (HyperFacets), *CA DWR 4600014294 (California Department of Water Resources Atmospheric River Program Phase III), *CIROH subaward A23-0252-S002 from Cooperative Institute for Research to Operations in Hydrology (CIROH) through the NOAA Cooperative Agreement (grant no. NA22NWS4320003), *DoE NERSC award ERCAP0024296},
	pages = {e2024WR038928},
	file = {Full Text PDF:/Users/taddbindas/Zotero/storage/TWJMPMM9/Song et al. - High-resolution national-scale water modeling is enhanced by multiscale differentiable physics-infor.pdf:application/pdf;Full Text PDF:/Users/taddbindas/Zotero/storage/46ZGUIZ5/Song et al. - 2025 - High-Resolution National-Scale Water Modeling Is E.pdf:application/pdf;Full Text PDF:/Users/taddbindas/Zotero/storage/JX2MD99P/Song et al. - 2025 - High-Resolution National-Scale Water Modeling Is Enhanced by Multiscale Differentiable Physics-Informed Machine Learning.pdf:application/pdf},
}

```

### Bindas et al. (2024)

**Title:** Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning

**Open Sourced Code:** [dMC-Juniata-HydroDL2](https://github.com/mhpi/dMC-Juniata-hydroDL2)

**Citation:**
```bibtex
@article{https://doi.org/10.1029/2023WR035337,
author = {Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and Rahmani, Farshid and Feng, Dapeng and Bian, Yuchen and Lawson, Kathryn and Shen, Chaopeng},
title = {Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning},
journal = {Water Resources Research},
volume = {60},
number = {1},
pages = {e2023WR035337},
keywords = {flood, routing, deep learning, physics-informed machine learning, Manning's roughness},
doi = {https://doi.org/10.1029/2023WR035337},
url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2023WR035337},
eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2023WR035337},
note = {e2023WR035337 2023WR035337},
abstract = {Abstract Recently, rainfall-runoff simulations in small headwater basins have been improved by methodological advances such as deep neural networks (NNs) and hybrid physics-NN models—particularly, a genre called differentiable modeling that intermingles NNs with physics to learn relationships between variables. However, hydrologic routing simulations, necessary for simulating floods in stem rivers downstream of large heterogeneous basins, had not yet benefited from these advances and it was unclear if the routing process could be improved via coupled NNs. We present a novel differentiable routing method (δMC-Juniata-hydroDL2) that mimics the classical Muskingum-Cunge routing model over a river network but embeds an NN to infer parameterizations for Manning's roughness (n) and channel geometries from raw reach-scale attributes like catchment areas and sinuosity. The NN was trained solely on downstream hydrographs. Synthetic experiments show that while the channel geometry parameter was unidentifiable, n can be identified with moderate precision. With real-world data, the trained differentiable routing model produced more accurate long-term routing results for both the training gage and untrained inner gages for larger subbasins (>2,000 km2) than either a machine learning model assuming homogeneity, or simply using the sum of runoff from subbasins. The n parameterization trained on short periods gave high performance in other periods, despite significant errors in runoff inputs. The learned n pattern was consistent with literature expectations, demonstrating the framework's potential for knowledge discovery, but the absolute values can vary depending on training periods. The trained n parameterization can be coupled with traditional models to improve national-scale hydrologic flood simulations.},
year = {2024}
}
```

## Related Works

Below are citations of papers which were influential in the creation of DDR.

### David et al. (2011)

**Documentation:** [RAPID River routing](http://rapid-hub.org/docs/RAPID_Parallel_Computing.pdf#page=4.00)
```bibtex
@article{david_river_2011,
	title = {River {Network} {Routing} on the {NHDPlus} {Dataset}},
	volume = {12},
	issn = {1525-7541, 1525-755X},
	url = {https://journals.ametsoc.org/view/journals/hydr/12/5/2011jhm1345_1.xml},
	doi = {10.1175/2011JHM1345.1},
	abstract = {Abstract The mapped rivers and streams of the contiguous United States are available in a geographic information system (GIS) dataset called National Hydrography Dataset Plus (NHDPlus). This hydrographic dataset has about 3 million river and water body reaches along with information on how they are connected into networks. The U.S. Geological Survey (USGS) National Water Information System (NWIS) provides streamflow observations at about 20 thousand gauges located on the NHDPlus river network. A river network model called Routing Application for Parallel Computation of Discharge (RAPID) is developed for the NHDPlus river network whose lateral inflow to the river network is calculated by a land surface model. A matrix-based version of the Muskingum method is developed herein, which RAPID uses to calculate flow and volume of water in all reaches of a river network with many thousands of reaches, including at ungauged locations. Gauges situated across river basins (not only at basin outlets) are used to automatically optimize the Muskingum parameters and to assess river flow computations, hence allowing the diagnosis of runoff computations provided by land surface models. RAPID is applied to the Guadalupe and San Antonio River basins in Texas, where flow wave celerities are estimated at multiple locations using 15-min data and can be reproduced reasonably with RAPID. This river model can be adapted for parallel computing and although the matrix method initially adds a large overhead, river flow results can be obtained faster than with the traditional Muskingum method when using a few processing cores, as demonstrated in a synthetic study using the upper Mississippi River basin.},
	language = {EN},
	number = {5},
	urldate = {2023-08-10},
	journal = {Journal of Hydrometeorology},
	author = {David, Cédric H. and Maidment, David R. and Niu, Guo-Yue and Yang, Zong-Liang and Habets, Florence and Eijkhout, Victor},
	month = oct,
	year = {2011},
	note = {Publisher: American Meteorological Society
Section: Journal of Hydrometeorology},
	pages = {913--934},
}
```
