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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Argentiere/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Argentiere/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Argentiere/'


#%% FUnctions

def reproject_match_grid( ref_img_da, img_da , resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan):
    ''' Match xarray grid of different spatial resolutions.'''
    
    # Expected order: ('time', 'y', 'x')
    dims = img_da.dims
    if 'time' in dims:
        ref_img_da = ref_img_da.transpose('time','y','x') # CRS is alreadyy written .rio.write_crs(3031, inplace=True)
        img_da = img_da.transpose('time','y','x')
    
    # -- reproject (even though same crs) and match grid (extent, resolution and projection)
    img_repr_match = img_da.rio.reproject_match(ref_img_da,resampling=resample_method,nodata=nodata_value) # need to specify nodata, otherwise fills with (inf) number 1.79769313e+308

    # advised to update coords to make the coordinates the exact same due to tiny differences in the coordinate values due to floating precision
    img_repr_match = img_repr_match.assign_coords({
        "y": ref_img_da.y,
        "x": ref_img_da.x,
    })
    
    return img_repr_match.transpose(*dims) # transpose dimension order back to original

#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed

## DHDT: 2012-2021 in m/yr. No cleaning needed.

## Velocity: no cleaning needed.

## Thickness: 
profiles.csv is from 2017 and is used to obtain the interpolated H.tiffs (from readme)
But which thickness grid to choose?
- argentiere_h_interpIGM_20170215: is interpolated with thickness inversion 
    from IGM that also uses velocity (same as submitted velocity data)
    --> do not use, circularity of approach?
- argentiere_h_interpSGS_Farinotti2019_20170215: is interpolated using 
    Sequential Gaussian Simulations (SGS) and the Farinotti et al. (2019) bed
    --> can use (?) although Farinotti2019 bed is also obtained from models with thickness inversion..
- argentiere_h_interpSGS_SIA_20170215: is interpolated using SGS
     and a bed estimated using the SIA from the Pléiades surface velocities
     --> do not use, circularity of approach?

## Thickness profiles: 
csv in EPSG4326, convert to shapefile and to EPSG32632


## Bedrock: not provided
Calculate as `argentiere_bedrock = argentiere_DEM_20170215 - argentiere_h_20170215_SGS-Farinotti`.
Before calculating difference, the thickenss grid is upsampled to the DEM grid (using nearest neighbor interpolation so that there's minimal data manipulation)
TO DO: maybe update if we decide to use another thickness grid.

---
TO DO: check grid resolutions and homogenize these

##################################
'''


''' ##################################
Thickness profiles
################################## '''

df_profiles = pd.read_csv(os.path.join(path2data_raw, 'argentiere_h_20170215.csv'),
                        #   skiprows=1, # skip first row with metadata
                          header=0, # use first row as header
                          delimiter=';')
df_profiles.head()
df_profiles['lat'] = df_profiles['latitude'].astype(float)
df_profiles['lon'] = df_profiles['longitude'].astype(float)

## convert to geodataframe, set geometry and CRS
crs_profiles = 'EPSG:4326'
gdf_profiles4326 = gpd.GeoDataFrame(df_profiles,
                                geometry=gpd.points_from_xy(df_profiles['lon'], df_profiles['lat']),
                                crs=crs_profiles)
## reproject to EPSG32632
gdf_profiles = gdf_profiles4326.to_crs('EPSG:32632')

## to save to SHP, column names longer than 10char will be truncated.
# identify columns longer than 10 char 
column_names = gdf_profiles.columns
long_columns = [col for col in column_names if len(col) > 10]
print(f"Columns longer than 10 characters: {long_columns}") 
# ['elevation_date', 'max_elevation_date', 'thickness_uncertainty']

# rename columns to be shorter than 10 char
gdf_profiles = gdf_profiles.rename(columns={
    'elevation_date': 'elev_date',
    # 'max_elevation_date': 'max_elevDate', ## drop this column since there's only one value (15/02/2017)
    'thickness_uncertainty': 'h_uncert',
}).drop(columns=['max_elevation_date']) # drop this column since there's only one value (15/02/2017)

## save as shapefile in cleaned data directory
fname_profiles = 'argentiere_h_profiles_20170215.shp'
if not os.path.exists(os.path.join(path2data_clean, fname_profiles)):
    gdf_profiles.to_file(os.path.join(path2data_clean, fname_profiles))
else:
    print(f"File {fname_profiles} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
Bedrock
################################## '''

da_dem17 = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_DEM_20170215.tif')
                             ).isel(band=0).drop_vars('band')
da_h17 = xr.open_dataarray(os.path.join(path2data_raw, 'argentiere_h_20170215.tif')
                           ).isel(band=0).drop_vars('band')

assert da_dem17.rio.crs == da_h17.rio.crs, "CRS of DEM and thickness grid do not match"
# assert da_dem17.rio.resolution() == da_h17.rio.resolution(), "Resolution of DEM and thickness grid do not match"
print(f"CRS of DEM and thickness grid match: {da_dem17.rio.crs}")


## thickness and DEM have different resolutions. DEM is 4m, thickness is 20m. So need to reproject and match grid of thickness to DEM.
da_h17_4m = reproject_match_grid(da_dem17, da_h17, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
assert da_dem17.rio.resolution() == da_h17_4m.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem17.shape == da_h17_4m.shape, "Shape of DEM and thickness grid do not match"

## calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem17 - da_h17_4m
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem17,  ## whre condition is ture
                               da_bedrock) ## where condition is false
da_bedrock_filled.plot.imshow()

## save to CLEAN directory
fname_bedr = 'argentiere_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")
    


# %%
