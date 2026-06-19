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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Zongo/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Zongo/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Zongo/'


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


## DEMs: as submitted
## DHDT: is in meters --> to m/yr

## Velocity: 
- GRID: submitted millan velocity --> scale
- STAKES: 

## Thickness GRID
submitted millan velocity --> do not use; use GlaTE interpolated thickness (other script)


## Thickness profiles: 
- check csv, save as shp. Repojrect lat/lon to EPSG 32719

## Bedrock: not provided;
- Calculate from DEM - thickness
- Use GlaTE interpolated rather than millan thickness.
  Interpolated thickness is 2012, closest dem is 2013 (not 2006)

---
TO DO: check grid resolutions and homogenize these

##################################
'''
zongo_outline = gpd.read_file(os.path.join(path2data_raw, 'zongo_outline-2006.shp'))

''' ##################################
Thickness profiles
################################## '''

df_profiles = pd.read_csv(os.path.join(path2data_raw, 'zongo_h_profiles_20120809.csv'),
                         delimiter=';',
                         header=0)
df_profiles
df_profiles.head()
df_profiles['lat'] = df_profiles['latitude'].astype(float)
df_profiles['lon'] = df_profiles['longitude'].astype(float)

## convert to geodataframe, set geometry and CRS
crs_profiles = 'EPSG:4326'
gdf_profiles4326 = gpd.GeoDataFrame(df_profiles,
                                geometry=gpd.points_from_xy(df_profiles['lon'], df_profiles['lat']),
                                crs=crs_profiles)
## reproject to EPSG32719
gdf_profiles = gdf_profiles4326.to_crs('EPSG:32719')

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
fname_profiles = 'zongo_h_profiles_20120809.shp'
if not os.path.exists(os.path.join(path2data_clean, fname_profiles)):
    gdf_profiles.to_file(os.path.join(path2data_clean, fname_profiles))
else:
    print(f"File {fname_profiles} already exists in cleaned data directory. Skipping save.")


#%%
''' ##################################
Bedrock
################################## '''

da_dem_2013 = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_DEM_2013.tif')
                    ).isel(band=0).drop_vars('band')
da_h_2012 = xr.open_dataarray(os.path.join(path2data_clean, 'Zongo_h_20120809.tif')
                    ).isel(band=0).drop_vars('band')
print(da_dem_2013.rio.crs)
assert da_dem_2013.rio.crs == da_h_2012.rio.crs, "CRS of DEM and thickness grid do not match"
assert da_dem_2013.rio.resolution() == da_h_2012.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem_2013.shape == da_h_2012.shape, "Shape of DEM and thickness grid do not match"

## calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem_2013 - da_h_2012
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem_2013,  ## whre condition is ture
                               da_bedrock) ## where condition is false
# da_bedrock_filled.plot.imshow()

fig,axs = plt.subplots(1,3, figsize=(15,5))
da_dem_2013.plot.imshow(ax=axs[0], vmin=4500, vmax=6000)
axs[0].set_title('DEM 2013')
da_h_2012.plot.imshow(ax=axs[1], vmin=0, vmax=200, cmap='Blues')
axs[1].set_title('Thickness 2012')  
da_bedrock_filled.plot.imshow(ax=axs[2], vmin=4500, vmax=6000)
axs[2].set_title('Bedrock (DEM - Thickness)')

## save to CLEAN directory
fname_bedr = 'zongo_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
dhdt
################################## '''

da_dhdt = xr.open_dataarray(os.path.join(path2data_raw, 'zongo_dhdt_2006-2013.tif')
                             ).isel(band=0).drop_vars('band')
## has bad NaN values .. mask lim values 
da_dhdt = da_dhdt.where(np.abs(da_dhdt) < 1000) ## set values with abs > 1000 to nan
da_dhdt = da_dhdt / (2013-2006) ## in m/yr

zongo_outline = gpd.read_file(os.path.join(path2data_raw, 'zongo_outline-2006.shp'))

assert da_dhdt.rio.crs == da_h_2012.rio.crs
assert da_dhdt.rio.resolution() == da_h_2012.rio.resolution()

## save to CLEAN directory
fname = 'zongo_dhdt_2006-2013.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")
# %%
