"""
Forward routing BMI with the NextGen hydrofabric (v2.2) for the Juniata River
Basin.
"""
import sys
import numpy as np
import os
from pathlib import Path
from ddr.bmi import dMCRoutingBMI as Bmi

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '..'))


### Configuration Settings (Single-gauge Run) ###
BMI_CFG_PATH = './ngen_resources/data/ddr/config/bmi_gage-01563500.yaml'

HYDROFABRIC = '/ngen_resources/data/ddr/spatial/jrb_hydrofabric_v2.2.gpkg'
NETWORK = '/ngen_resources/data/ddr/spatial/network.nc'
TRANSITION_MAT = '/ngen_resources/data/ddr/spatial/conus_transition_matrices.nc'
STATISTICS = '/ngen_resources/data/ddr/spatial/attribute_statistics_hydrofabric_v2.2.json'
STREAMFLOW = r'C:/Users/LeoLo/Desktop/routing_data/streamflow_m73'
OBS = r'C:/Users/LeoLo/Desktop/routing_data/gages_9000.zarr'
GAGES = '/ngen_resources/data/ddr/spatial/gages.csv'
### ------------------------------------ ###


pkg_root = Path(__file__).parent.parent
bmi_cfg_path_full = os.path.join(pkg_root, Path(BMI_CFG_PATH))

# Create dMC BMI instance
model = Bmi(config_path=bmi_cfg_path_full)

nan_idx = []

# # 1) Compile forcing data within BMI to do batch run.
# for i in range(0, forc.shape[0]):
#     # Extract forcing/attribute data for the current time step
#     prcp = forc[i, :, 0]
#     temp = forc[i, :, 1]
#     pet = forc[i, :, 2]

#     ## Check if any of the inputs are NaN
#     if np.isnan([prcp, temp, pet]).any():
#         # if model.verbose > 0:
#         print(f"Skipping timestep {i} due to NaN values in inputs.")
#         nan_idx.append(i)
#         continue

#     model.set_value('atmosphere_water__liquid_equivalent_precipitation_rate', prcp)
#     model.set_value('land_surface_air__temperature', temp)
#     model.set_value('land_surface_water__potential_evaporation_volume_flux', pet)


### BMI initialization ###
model.initialize()

# # 2) DO pseudo model forward and return pre-predicted values at each timestep
# for i in range(0, forc.shape[0]):
#     if i in nan_idx:
#         # Skip the update for this timestep
#         continue

#     ### BMI update ###
#     model.update()

#     # Retrieve and scale the runoff output
#     dest_array = np.zeros(1)
#     model.get_value('land_surface_water__runoff_volume_flux', dest_array)
    
#     streamflow_pred[i] = dest_array[0]  # Convert to mm/day -> mm/hr

#  ### BMI finalization ###
# model.finalize()

# print("\n=/= -- Streamflow prediction completed -- =/=")
# print(f"    Basin ID:              {BASIN_ID}")
# print(f"    Total Process Time:    {model.bmi_process_time:.4f} seconds")
# print(f"    Mean streamflow:       {streamflow_pred.mean():.4f} mm/day")
# print(f"    Max streamflow:        {streamflow_pred.max():.4f} mm/day")
# print(f"    Min streamflow:        {streamflow_pred.min():.4f} mm/day")
# print("=/= ------------------------------------- =/=")


# # Calculate NSE for the model predictions
# obs = obs.dropna()
# sim = streamflow_pred.dropna()

# denom = ((obs - obs.mean()) ** 2).sum()
# num = ((sim - obs) ** 2).sum()
# nse = 1 - num / denom
# print(f"NSE: {nse:.2f}")
