#%% Check and pre-processing of data

# M. Izeboud, June 2026

import numpy as np
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_raw_data/Aletsch/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_cleaned_data/Aletsch/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_experiment_package/Aletsch/'

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

## DHDT: 
check if values are in total range (2017-2023) or per year
--> per year (m/year); so can be put into 'clean' dir.

## Velocity: 
- from m/day to m/year (--> to 'clean' dir) 
- reproject 4326 to 2056 (--> to homogenized dir)

## Thickness: 
check profile data for which year they are --> 2011 and 2009 
thickness .tif is from 2017, so not the same year as profiles. --> specify in filenames.

## Bedrock: infer from thickness and DEM in 2017 --> to 'clean' dir.

---
TO DO: check grid resolutions and homogenize these

##################################
'''

### THICKNESS
gdf_thickness = gpd.read_file(os.path.join(path2data_raw, 'aletsch_h_profiles_2009-2011.shp'))
thickness_profile_dates = gdf_thickness['prf_id']
thickness_profile_dates = [date.split('_')[0] for date in thickness_profile_dates]
gdf_thickness['date'] = thickness_profile_dates

print('CRS thickness shapefile:', gdf_thickness.crs)
file_aletsch_grid = os.path.join(path2data_raw,'aletsch_h_20170901.tif')
da_H17 = xr.open_dataset(file_aletsch_grid, engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'H'})['H']
print('CRS thickness grid:', da_H17.rio.crs)


##  DHDT : calculate DEM and compare to submitted dhdt 
da_dem17 = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_dem_20170901.tif'), engine='rasterio').isel(band=0).drop_vars('band')
da_dem23 = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_dem_20230823.tif'), engine='rasterio').isel(band=0).drop_vars('band')
da_dhdt = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_dhdt_20170901-20230823.tif'), engine='rasterio').isel(band=0).drop_vars('band')
print('CRS DEM 2017, 2023:', da_dem17.rio.crs, da_dem23.rio.crs)
print('CRS DHDT:', da_dhdt.rio.crs)

da_dhdt_calc = (da_dem23 - da_dem17) / (2023-2017)
da_dhdt_diff = da_dhdt - da_dhdt_calc

fig,axs =plt.subplots(1,3, figsize=(15,5))
da_dhdt.plot.imshow(ax=axs[0], vmin=-20, vmax=20, cmap='RdBu_r'); 
axs[0].set_title('Submitted DHDT')
da_dhdt_calc.plot.imshow(ax=axs[1], vmin=-20, vmax=20, cmap='RdBu_r'); axs[1].set_title('Calculated DHDT')
da_dhdt_diff.plot.imshow(ax=axs[2], vmin=-5, vmax=5, cmap='RdBu_r'); axs[2].set_title('Difference')
plt.close() 


''' ##################################
Bedrock
##################################
'''

assert da_dem17.rio.crs == da_H17.rio.crs, "CRS of DEM and thickness grid do not match"
assert da_dem17.rio.resolution() == da_H17.rio.resolution(), "Resolution of DEM and thickness grid do not match"
## thickness tif grid is smaller than DEM, so reproject and match grid
## they already have the same resolution and CRS, so minimal data manipulation occurs.
da_H17_matched = reproject_match_grid(da_dem17, da_H17, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

da_bedrock17 = da_dem17 - da_H17_matched
## fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock17_filled = xr.where(np.isnan(da_bedrock17), 
                               da_dem17,  ## whre condition is ture
                               da_bedrock17) ## where condition is false
da_bedrock17_filled.plot.imshow()

## save to CLEAN directory
fname_bedr = 'aletsch_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock17_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")
    


''' ##################################
Velocity data
##################################
'''

### Velocity read data
da_vx = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_vx_mday_EPSG4326.tif'), engine='rasterio').isel(band=0).drop_vars('band').rename('vx')
da_vy = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_vy_mday_EPSG4326.tif'), engine='rasterio').isel(band=0).drop_vars('band').rename('vy')
# ds_velo = xr.merge([da_vx, da_vy],compat='no_conflicts')
print('CRS velocity grids:', da_vx.rio.crs)
da_vx_stdev = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_vx_stddev_mday_EPSG4326.tif'), engine='rasterio').isel(band=0).drop_vars('band').rename('vx')
da_vy_stdev = xr.open_dataarray(os.path.join(path2data_raw, 'aletsch_vy_stddev_mday_EPSG4326.tif'), engine='rasterio').isel(band=0).drop_vars('band').rename('vy')

## CLEANING: unit conversion from m/day to m/year
da_vx_myear = da_vx * 365.25
da_vy_myear = da_vy * 365.25
da_vx_stdev_myear = da_vx_stdev * 365.25
da_vy_stdev_myear = da_vy_stdev * 365.25

## save to CLEAN directory
fname_vx = 'aletsch_vx_EPSG4326.tif'
fname_vy = 'aletsch_vy_EPSG4326.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
    da_vx_myear.rio.to_raster(os.path.join(path2data_clean, fname_vx))
    da_vy_myear.rio.to_raster(os.path.join(path2data_clean, fname_vy))
else:
    print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")
fname_vx_stdev = 'aletsch_vx_stddev_EPSG4326.tif'
fname_vy_stdev = 'aletsch_vy_stddev_EPSG4326.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_vx_stdev)):
    da_vx_stdev_myear.rio.to_raster(os.path.join(path2data_clean, fname_vx_stdev))
    da_vy_stdev_myear.rio.to_raster(os.path.join(path2data_clean, fname_vy_stdev))

# 2. Reproject the grid definitions using rioxarray
target_crs = "EPSG:2056"  # Swiss coordinate system

# -----------------------------
# Function to rotate vectors
# -----------------------------
def transform_velocity_components_epsg4326_to_epsg2056(da_vx, da_vy, 
                                                       src_epsg="4326", dst_epsg="2056"):
    """
    Convert velocity components from geographic east/north components
    to EPSG:2056 x/y components.

    velocity can be in m/day or m/year, as long as the spatial unit is in meters

    For each velocity pixel:
    1. Take the pixel center coordinate in lon/lat.
    2. Use the velocity vector to define a small displacement over one year:
        point_1 = lon/lat position
        point_2 = point_1 shifted by ve, vn in metres
    3. Transform both points to EPSG:2056.
        vx_2056 = x2 - x1
        vy_2056 = y2 - y1
    """
    from pyproj import CRS, Transformer, Geod

    src_crs = CRS.from_epsg(src_epsg)
    dst_crs = CRS.from_epsg(dst_epsg)

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    geod = Geod(ellps="WGS84")

    # Get lon/lat coordinate grids
    lon = da_vx["x"].values
    lat = da_vx["y"].values
    lon2d, lat2d = np.meshgrid(lon, lat)

    vx = da_vx.values
    vy = da_vy.values

    # Velocity magnitude and azimuth in geographic coordinates
    speed = np.sqrt(vx**2 + vy**2)

    # pyproj.Geod uses azimuth clockwise from north:
    # eastward vx, northward vy -> azimuth = atan2(east, north)
    azimuth = np.degrees(np.arctan2(vx, vy))

    # Create a second point after moving by the velocity distance. 
    # Speed: spatial unit should be in meters (doesnt matter if its m/day or m/year)
    lon_end, lat_end, _ = geod.fwd(lon2d, lat2d, azimuth, speed)

    # Project start and end points to EPSG:2056
    x0, y0 = transformer.transform(lon2d, lat2d)
    x1, y1 = transformer.transform(lon_end, lat_end)

    # Difference gives velocity components in EPSG:2056
    vx_2056 = x1 - x0
    vy_2056 = y1 - y0

    # Preserve NaNs
    mask = np.isnan(vx) | np.isnan(vy)
    vx_2056[mask] = np.nan
    vy_2056[mask] = np.nan

    da_vx_rot = da_vx.copy(data=vx_2056).rename("vx_2056")
    da_vy_rot = da_vy.copy(data=vy_2056).rename("vy_2056")

    da_vx_rot = da_vx_rot.rio.write_crs("EPSG:4326")
    da_vy_rot = da_vy_rot.rio.write_crs("EPSG:4326")

    return da_vx_rot, da_vy_rot


# -----------------------------
# Rotate velocity components
# -----------------------------
# da_vx_rot, da_vy_rot = transform_velocity_components_epsg4326_to_epsg2056(
#     da_vx, da_vy, src_epsg="4326", dst_epsg="2056"
# ) ## the m/day version
da_vx_rot, da_vy_rot = transform_velocity_components_epsg4326_to_epsg2056(
    da_vx_myear, da_vy_myear, src_epsg="4326", dst_epsg="2056"
)

# -----------------------------
# Reproject rasters to EPSG:2056
# -----------------------------
da_vx_2056 = da_vx_rot.rio.reproject("EPSG:2056")
da_vy_2056 = da_vy_rot.rio.reproject("EPSG:2056")

## stdev raster need to be reprojected but not rotated since its scalar fields
da_vx_stdev_2056 = da_vx_stdev_myear.rio.reproject("EPSG:2056")
da_vy_stdev_2056 = da_vy_stdev_myear.rio.reproject("EPSG:2056")

# -----------------------------
# Save output
# -----------------------------

## save to homomgenized directory
fname_vx = 'aletsch_vx_EPSG2056.tif'
fname_vy = 'aletsch_vy_EPSG2056.tif'
if not os.path.exists(os.path.join(path2data_homog, fname_vx)):
    da_vx_2056.rio.to_raster(os.path.join(path2data_homog, fname_vx))
    da_vy_2056.rio.to_raster(os.path.join(path2data_homog, fname_vy))
else:
    print(f"File {fname_vx} already exists in homogenized data directory. Skipping save.")
    
fname_vx_stdev = 'aletsch_vx_stddev_EPSG2056.tif'
fname_vy_stdev = 'aletsch_vy_stddev_EPSG2056.tif'
if not os.path.exists(os.path.join(path2data_homog, fname_vx_stdev)):
    da_vx_stdev_2056.rio.to_raster(os.path.join(path2data_homog, fname_vx_stdev))
    da_vy_stdev_2056.rio.to_raster(os.path.join(path2data_homog, fname_vy_stdev))


fig,axs =plt.subplots(2,2, figsize=(12,10))
da_vx.plot.imshow(ax=axs[0,0], vmin=-1, vmax=1, cmap='RdBu_r');
axs[0,0].set_title('vx 4326 (m/day)')
da_vx_2056.plot.imshow(ax=axs[0,1], vmin=-1*365.25, vmax=1*365.25, cmap='RdBu_r'); 
axs[0,1].set_title('vx 2056 (m/year)'); axs[0,1].set_xlabel('x [m]') #
da_vy.plot.imshow(ax=axs[1,0], vmin=-1, vmax=1, cmap='RdBu_r');
axs[1,0].set_title('vy 4326 (m/day)')
da_vy_2056.plot.imshow(ax=axs[1,1], vmin=-1*365.25, vmax=1*365.25, cmap='RdBu_r'); 
axs[1,1].set_title('vy 2056 (m/year)'); axs[1,1].set_xlabel('x [m]') # x [m]


# %%
