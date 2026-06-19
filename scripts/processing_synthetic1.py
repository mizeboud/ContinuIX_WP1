#%% Check and pre-processing of data

# M. Izeboud, June 2026

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Synthetic1/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Synthetic1/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Synthetic1/'


#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed

## all variables are provided; check them and save as geotiff


##################################
'''
