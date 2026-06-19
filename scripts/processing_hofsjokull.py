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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Hofsjokull/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Hofsjokull/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_experiment_package/Hofsjokull/'


#%% FUnctions

def reproject_match_grid( ref_img_da, img_da , 
                         resample_method=rio.enums.Resampling.nearest, 
                         nodata_value=np.nan):
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
## DHDT: calculate from DEMs
- individual dhdt periods --> to cleaned data folder
- average dhdt --> to homogenized data folder (also in target resolution)

## Velocity: gappy data; to interpolate/average

## Thickness GRID: not provided,
calculate from DEM YYYY and BED (which most represents 2013, but assuming constant).

## Thickness profiles: not available

## Bedrock: as submitted.

## Outlines: not avaialble; take RGIv6 outlines and reproject to 3057.

---
TO DO: check grid resolutions and homogenize these

##################################
'''
target_res = 100 # m; target resolution for homogenized data

''' ##################################
Outlines  
################################## '''
gdf_rgiv6 = gpd.read_file('../../ContinuIX_WP1_data/other_data/outline_Hofsjokull_RGIv6_4326.shp')
# gdf_rgiv6.plot()
print(gdf_rgiv6.crs)
## reproject to iceland reprojection system
gdf_hofsj = gdf_rgiv6.to_crs(3057)

## save to cleaned dir
fname = 'hofsjokull_outline_RGIv6.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    gdf_hofsj.to_file(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")

## FOR PLOTS in this script
## merge all geometries into one (dissolve) to have a single outline for clipping
hofsj_union = gdf_hofsj.geometry.union_all()   # GeoPandas/Shapely recent versions
# or (older API): hofsj_union = gdf_hofsj.unary_union
gdf_hofsj_union = gpd.GeoDataFrame(geometry=[hofsj_union], crs=gdf_hofsj.crs)


#%%
''' ##################################
dhdt  
################################## '''

da_dem_2013 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_20131013.tif')
                             ).isel(band=0).drop_vars('band')
da_dem_2020 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_202010.tif')
                             ).isel(band=0).drop_vars('band')
da_dem_2023 = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_dem_202309.tif')
                             ).isel(band=0).drop_vars('band')

assert da_dem_2013.rio.crs == da_dem_2020.rio.crs == da_dem_2023.rio.crs, "CRS should be the same for all DEM files"
assert da_dem_2013.rio.resolution() == da_dem_2020.rio.resolution() == da_dem_2023.rio.resolution(), "Resolution should be the same for all DEM files"
# assert da_dem_2013.shape == da_dem_2020.shape == da_dem_2023.shape, "Shape should be the same for all DEM files"
print('CRS of Hofsjokull DEM files:', da_dem_2013.rio.crs)

## 2023 DEM file has slightly different grid boundaries: reproject match
da_dem_2023 = reproject_match_grid(da_dem_2020, da_dem_2023, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)


## calculate elevation change for different intervals and average

da_dhdt_1320 =(da_dem_2020 - da_dem_2013) / (2020-2013)
da_dhdt_1323 = (da_dem_2023 - da_dem_2013) / (2023-2013)
da_dhdt_2023 = (da_dem_2023 - da_dem_2020) / (2023-2020)
# da_dhdt_all = xr.concat([da_dhdt_1320, da_dhdt_1323, da_dhdt_2023], dim='time').assign_coords(time=['2013-2020', '2013-2023', '2020-2023'])
 
## dhdt 13-23 should be similar to dhdt1320+dhdt2023: check
da_dhdt_1323_b = (da_dhdt_1320*7+da_dhdt_2023*3)/10

## each dhdt period
# fig,axs=plt.subplots(1,3, figsize=(17,5))
# da_dhdt_1323.rename('dhdt (m/yr)').plot.imshow(ax=axs[0], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[0].set_title('2013 - 2023')
# da_dhdt_1320.rename('dhdt (m/yr)').plot.imshow(ax=axs[1], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[1].set_title('2013 - 2020')
# da_dhdt_2023.rename('dhdt (m/yr)').plot.imshow(ax=axs[2], vmin=-4, vmax=4, cmap="RdBu_r")
# axs[2].set_title('2020 - 2023')

## average dhdts comparison
fig,axs=plt.subplots(1,2, figsize=(12,5))
da_dhdt_1323.rename('dhdt (m/yr)').plot.imshow(ax=axs[0], vmin=-4, vmax=4, cmap="RdBu_r")
axs[0].set_title('dH/dt: DEM 2023 - DEM 2013')
da_dhdt_1323_b.rename('dhdt (m/yr)').plot.imshow(ax=axs[1], vmin=-4, vmax=4, cmap="RdBu_r")
axs[1].set_title('dH/dt: (dH/dt 13-20 + dH/dt 20-23)/2')

## save dhdt periods
fname1323 = f'hofsjokull_dhdt_2013-2023.tif'
fname1320 = f'hofsjokull_dhdt_2013-2020.tif'
fname2023 = f'hofsjokull_dhdt_2020-2023.tif'
if not os.path.exists(os.path.join(path2data_clean, fname1323)):
    print(f'Saving dhdt to cleaned data directory...')
    da_dhdt_1320.rio.to_raster(os.path.join(path2data_clean, fname1320))
    da_dhdt_1323.rio.to_raster(os.path.join(path2data_clean, fname1323))
    da_dhdt_2023.rio.to_raster(os.path.join(path2data_clean, fname2023))
else:
    print(f"File {fname1323} already exists in cleaned data directory. Skipping save.")

#%%
## save average dhdt to homogenized dir; at target res


#%% 
''' ##################################
Thickness 
################################## '''

da_bed = xr.open_dataarray(os.path.join(path2data_raw, 'hofsjokull_bedrock_2013.tif')
                             ).isel(band=0).drop_vars('band')

print(da_bed.rio.crs, da_dem_2013.rio.crs) ## both are 3057 but still get assertion error for some reason; manually update crs
da_bed.rio.set_crs(3057,inplace=True)
assert da_bed.rio.crs == da_dem_2013.rio.crs, "CRS should be the same for all DEM files"

## interpolate bed resolution (200m) to DEM resolution (2m)
da_bed_matched = reproject_match_grid(da_dem_2013, da_bed, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
## da_thickness: use 2013
# da_thickness = da_dem_2013 - da_bed_matched
# da_thickness.plot.imshow(cmap='Blues', vmin=0, vmax=500)

### calculate thickness for every year, save file
for da_dem, year in zip([da_dem_2013, da_dem_2020, da_dem_2023], ['2013', '2020', '2023']):
    da_thickness = da_dem - da_bed_matched
    fname = f'hofsjokull_h_{year}.tif'
    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Saving {fname} to cleaned data dir')
        da_thickness.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")





#%%
''' ------------------
### HOFSJOKUL VELOCITIES  
----------------------'''
## 100 m resolution; in METER / DAY; need to convert to m/yr by multiplying by 365.25
data_dir = '../../ContinuIX_WP1_data/'
filelist_velo_hofsj = sorted([f for f in os.listdir(os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/')) if f.endswith('.tif')])
filelist_velo_vz = [file for file in filelist_velo_hofsj if 'vertical' in file] # vertical velocity
years_vz = [2014, 2015, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
## get easting and northign
filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'easting' in file or 'northing' in file] # exclude stdev files
filelist_velo_std = [file for file in filelist_velo_hofsj if 'stddev' in file] # get stdev files for later use
filelist_vx_std = [file for file in filelist_velo_std if 'easting' in file]
filelist_vy_std = [file for file in filelist_velo_std if 'northing' in file]

filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'stddev' not in file] # exclude stdev files 
filelist_vx = [file for file in filelist_velo_hofsj if 'easting' in file]
filelist_vy = [file for file in filelist_velo_hofsj if 'northing' in file]
years_vx = [2014,2016,2017,     2019,2020,2021,2022,2023] ## left year (2014-2015 noted as 2014)
years_vy = [2014,2016,2017,2018,2019,2020,2021,2022,2023]
## drop 2018 from filelist 
filelist_vx = [f for f in filelist_vx if 's2018' not in f]
filelist_vy = [f for f in filelist_vy if 's2018' not in f]
filelist_vx_std = [f for f in filelist_vx_std if 's2018' not in f]
filelist_vy_std = [f for f in filelist_vy_std if 's2018' not in f]
years_vy = [2014,2016,2017,     2019,2020,2021,2022,2023] ## update

assert len(filelist_vx) == len(filelist_vy)
assert len(filelist_vx_std) == len(filelist_vy_std)

## open dataset
ds_hofsj_vx = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vx], 
                                engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                ).assign_coords(time=years_vx).isel(band=0).drop_vars('band').rename({'band_data':'vx'})
ds_hofsj_vy = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vy],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vy).isel(band=0).drop_vars('band').rename({'band_data':'vy'})

ds_hofsj_vx_std = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vx_std],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vx).isel(band=0).drop_vars('band').rename({'band_data':'vx'})
ds_hofsj_vy_std = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vy_std],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vy).isel(band=0).drop_vars('band').rename({'band_data':'vy'})
ds_hofsj_vz = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_velo_vz],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vz).isel(band=0).drop_vars('band').rename({'band_data':'vz'})
assert ds_hofsj_vx.rio.crs == ds_hofsj_vy.rio.crs, f"CRS should match between vx and vy; are {ds_hofsj_vx.rio.crs} and {ds_hofsj_vy.rio.crs}"

print('CRS of Hofsjokull velocity files:', ds_hofsj_vx.rio.crs)


## save each year to RAW datafolder with renaming of the file BEFORE CONVERTING TO m/yr
for year in years_vy:
    da_vx = ds_hofsj_vx['vx'].sel(time=year)
    da_vy = ds_hofsj_vy['vy'].sel(time=year)
    fname_vx = f'hofsjokull_vx_{year}-{year+1}_mday.tif'
    fname_vy = f'hofsjokull_vy_{year}-{year+1}_mday.tif'

    da_vx_std = ds_hofsj_vx_std['vx'].sel(time=year)
    da_vy_std = ds_hofsj_vy_std['vy'].sel(time=year)
    fname_vx_std = f'hofsjokull_vx_{year}-{year+1}_std_mday.tif'
    fname_vy_std = f'hofsjokull_vy_{year}-{year+1}_std_mday.tif'

    print(fname_vx)
    ## VX and VY
    if not os.path.exists(os.path.join(path2data_raw, fname_vx)):
        print(f'Saving {fname_vx} to RAW data dir')
        da_vx.rio.to_raster(os.path.join(path2data_raw, fname_vx))
    else:
        print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")
    if not os.path.exists(os.path.join(path2data_raw, fname_vy)):
        print(f'Saving {fname_vy} to cleaned data dir')
        da_vy.rio.to_raster(os.path.join(path2data_raw, fname_vy))
    else:
        print(f"File {fname_vy} already exists in cleaned data directory. Skipping save.")
    
    ## STDEVS
    if not os.path.exists(os.path.join(path2data_raw, fname_vx_std)):
        print(f'Saving {fname_vx_std} to RAW data dir')
        da_vx_std.rio.to_raster(os.path.join(path2data_raw, fname_vx_std))
    if not os.path.exists(os.path.join(path2data_raw, fname_vy_std)):
        print(f'Saving {fname_vy_std} to RAW data dir')
        da_vy_std.rio.to_raster(os.path.join(path2data_raw, fname_vy_std))

    ## VERTICAL
    if year in years_vz:
        da_vz = ds_hofsj_vz['vz'].sel(time=year)
        fname_vz = f'hofsjokull_v-vertical_{year}-{year+1}_mday.tif'
        if not os.path.exists(os.path.join(path2data_raw, fname_vz)):
            print(f'Saving {fname_vz} to RAW data dir')
            da_vz.rio.to_raster(os.path.join(path2data_raw, fname_vz))

## convert m/day to m/year
ds_hofsj_vx['vx'] = ds_hofsj_vx['vx'] * 365.25
ds_hofsj_vy['vy'] = ds_hofsj_vy['vy'] * 365.25

ds_hofsj_vx_std['vx'] = ds_hofsj_vx_std['vx'] * 365.25
ds_hofsj_vy_std['vy'] = ds_hofsj_vy_std['vy'] * 365.25
ds_hofsj_vz['vz'] = ds_hofsj_vz['vz'] * 365.25

## update attribute 
ds_hofsj_vx['vx'].attrs['units'] = 'm/yr'
ds_hofsj_vy['vy'].attrs['units'] = 'm/yr'
ds_hofsj_vz['vz'].attrs['units'] = 'm/yr'
ds_hofsj_vx_std['vx'].attrs['units'] = 'm/yr'   
ds_hofsj_vy_std['vy'].attrs['units'] = 'm/yr'


#%%
'''## plot velocity components for all years'''
ds_hofsj_vx['vx'].plot.imshow(cmap='PiYG', col='time', col_wrap=2, vmin=-50, vmax=50, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
## get axes
axs = plt.gcf().axes[:-1] # exclude colorbar axis
[gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
[ax.set_axis_off() for ax in axs]

## plot velocity components for all years
ds_hofsj_vy['vy'].plot.imshow(cmap='PiYG', col='time', col_wrap=2, vmin=-50, vmax=50, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
## get axes
axs = plt.gcf().axes[:-1] # exclude colorbar axis
[gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
[ax.set_axis_off() for ax in axs]

'''## calculate velocity magnitude'''
da_hofsj_v = (np.sqrt(ds_hofsj_vx['vx']**2 + ds_hofsj_vy['vy']**2)).rename('velocity')

## plot velocity for all years
da_hofsj_v.plot.imshow(cmap='viridis', col='time', col_wrap=2, vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
## get axes
axs = plt.gcf().axes[:-1] # exclude colorbar axis
[gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
[ax.set_axis_off() for ax in axs]

#%% Select only 2017-18, 2019-20, 2020-21, 2023-24
years_select = [2017,2019,2020,2023]
years_select = [2017,2019,2020] ## update: not 2023, as its outside DEM bounds

da_hofsj_vx_sel = ds_hofsj_vx.sel(time=years_select)['vx']
da_hofsj_vy_sel = ds_hofsj_vy.sel(time=years_select)['vy']
da_hofsj_v_sel = (np.sqrt(da_hofsj_vx_sel**2 + da_hofsj_vy_sel**2)).rename('velocity')

### Save AVERAGE to CLEAN folder
fname_vx = f'hofsjokull_vx_{years_select[0]}-{years_select[-1]}.tif'
fname_vy = f'hofsjokull_vy_{years_select[0]}-{years_select[-1]}.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
    print(f'Saving {fname_vx} to cleaned data dir')
    da_hofsj_vx_sel.mean(dim='time').rio.to_raster(os.path.join(path2data_clean, fname_vx))
    da_hofsj_vy_sel.mean(dim='time').rio.to_raster(os.path.join(path2data_clean, fname_vy))
else:
    print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")


## show average fields (no smoothing)
# fig,axs=plt.subplots(1,3,figsize=(16,5))
fig,axs=plt.subplots(1,3,figsize=(16,4))
da_hofsj_vx_sel.mean(dim='time').plot.imshow(ax=axs[0], vmin=-50, vmax=50, cmap="PiYG")
axs[0].set_title('vx average (m/yr)')
da_hofsj_vy_sel.mean(dim='time').plot.imshow(ax=axs[1], vmin=-50, vmax=50, cmap="PiYG")
axs[1].set_title('vy average (m/yr)')
da_hofsj_v_sel.mean(dim='time').plot.imshow(ax=axs[2], vmin=0, vmax=100, cmap='viridis')
axs[2].set_title('v average (m/yr)')
[ax.set_aspect('equal') for ax in axs]

fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
                f'Hofsjokull_velocity_selectYears_mean.jpg', dpi=300)
    

## save SELECTED files to 'cleaned' datafolder 
for year in years_select:
    da_vx = da_hofsj_vx_sel.sel(time=year)
    da_vy = da_hofsj_vy_sel.sel(time=year)
    fname_vx = f'hofsjokull_vx_{year}-{year+1}.tif'
    fname_vy = f'hofsjokull_vy_{year}-{year+1}.tif'

    print(fname_vx)
    if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
        print(f'Saving {fname_vx} to cleaned data dir')
        da_vx.rio.to_raster(os.path.join(path2data_clean, fname_vx))
    else:
        print(f"File {fname_vx} already exists in cleaned data directory. Skipping save.")

    if not os.path.exists(os.path.join(path2data_clean, fname_vy)):
        print(f'Saving {fname_vy} to cleaned data dir')
        da_vy.rio.to_raster(os.path.join(path2data_clean, fname_vy))
    else:
        print(f"File {fname_vy} already exists in cleaned data directory. Skipping save.")

    ## stdev and vertical velo
    da_vx_std = ds_hofsj_vx_std['vx'].sel(time=year)
    da_vy_std = ds_hofsj_vy_std['vy'].sel(time=year)
    da_vz = ds_hofsj_vz['vz'].sel(time=year)

    fname_vz = f'hofsjokull_vz_{year}-{year+1}.tif'
    fname_vx_std = f'hofsjokull_vx_{year}-{year+1}_std.tif'
    fname_vy_std = f'hofsjokull_vy_{year}-{year+1}_std.tif'

     ## STDEVS
    if not os.path.exists(os.path.join(path2data_clean, fname_vx_std)):
        print(f'Saving {fname_vx_std} to cleaned data dir')
        da_vx_std.rio.to_raster(os.path.join(path2data_clean, fname_vx_std))
    if not os.path.exists(os.path.join(path2data_clean, fname_vy_std)):
        print(f'Saving {fname_vy_std} to cleaned data dir')
        da_vy_std.rio.to_raster(os.path.join(path2data_clean, fname_vy_std))

    ## VERTICAL
    if year in years_vz:
        fname_vz = f'hofsjokull_v-vertical_{year}-{year+1}.tif'
        if not os.path.exists(os.path.join(path2data_clean, fname_vz)):
            print(f'Saving {fname_vz} to cleaned data dir')
            da_vz.rio.to_raster(os.path.join(path2data_clean, fname_vz))




#%% Smoothing velocity fields -- in the end not used

# # ## apply 3x3 rolling median
# # for ksize in [3,5,7,9,11]:
# #     # ksize = 5 ## still pretty spotty
# #     # ksize = 7
# #     # ksize = 9
# #     # ksize = 11 ## maybe too smooth
# #     da_vx_medianfilt = da_hofsj_vx_sel.rolling(y=ksize, x=ksize, center=True).median()
# #     da_vy_medianfilt = da_hofsj_vy_sel.rolling(y=ksize, x=ksize, center=True).median()

# #     da_v = (np.sqrt(da_vx_medianfilt**2 + da_vy_medianfilt**2)).rename('velocity')

# #     ## show average fields (no smoothing)
# #     # fig,axs=plt.subplots(1,3,figsize=(16,5))
# #     fig,axs=plt.subplots(1,3,figsize=(16,4))
# #     da_vx_medianfilt.mean(dim='time').plot.imshow(ax=axs[0], vmin=-50, vmax=50, cmap="PiYG")
# #     da_vy_medianfilt.mean(dim='time').plot.imshow(ax=axs[1], vmin=-50, vmax=50, cmap="PiYG")
# #     da_v.mean(dim='time').plot.imshow(ax=axs[2], vmin=0, vmax=100, cmap='viridis')

# #     axs[0].set_title('vx average (m/yr)')
# #     axs[1].set_title('vy average (m/yr)')
# #     axs[2].set_title('v average (m/yr)')
# #     [ax.set_aspect('equal') for ax in axs]
# #     fig.suptitle(f'Median filter {ksize}x{ksize}px, before average ')

# #     ## save figure
# #     fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
# #                 f'Hofsjokull_velocity_selectYears_medianfilt_{ksize}x{ksize}.jpg', dpi=300)
    
# ## Exponential filter -- better than median filter, less blurring.

# import skimage as sk
# import scipy.signal as signal

# ''' scipy.signal.windows.exponential(
#     M,           : number of points
#     center=None, : Parameter defining the center location of the window function. The default value if not given is center = (M-1) / 2. 
#     tau=1.0,  : Parameter defining the decay. 
#     sym=True : When True (default), generates a symmetric window,
#     )

# '''
# ## define exponential filter
# ksize = 11
# fig1,axs1=plt.subplots(1,5,figsize=(18,4)); c=0;
# for ksize, exp_decay_length in zip([3,5,7,9,11],
#                                    [3,5,7,9,11]
#                                     ):
#     ## decay distance should be smaller or close to ksize, otherwise you get averaging around cntr px
#     # exp_decay_length = 1 ## default
#     # exp_decay_length = 7 ## very reasonable, when ksize=11

#     ## set decay to ~80% of ksize
#     exp_decay_length= int(np.floor(0.8*ksize)) 

#     window_kernel = sk.filters.window(('exponential',None,exp_decay_length), (ksize, ksize))
#     ## normalize weights so that sum(window) is 1
#     window_kernel = window_kernel / np.sum(window_kernel)
#     wmax = 0.9*np.max(window_kernel)

#     # fig,ax=plt.subplots(1,figsize=(6,6))
#     ax=axs1[c]; c+=1;
#     h = ax.imshow(window_kernel, vmin=0, vmax=wmax, cmap='viridis')
#     fig.colorbar(h, ax=ax, location='right', shrink=0.7, )#anchor=(0, 0.3), )
#     ax.set_axis_off(); ax.set_title(f'k={ksize}, decay={exp_decay_length}')

#     ## convolve filter window on dataArray
#     da_list_vx= [];da_list_vy= []
#     for time in da_hofsj_vx_sel.time.values:
#         da_vx_yr = da_hofsj_vx_sel.sel(time=time)
#         da_vy_yr = da_hofsj_vy_sel.sel(time=time)
#         convolved_vx = signal.convolve(da_vx_yr.values, window_kernel,
#                                             mode='same', ## runtimewarning: Use of fft convolution on input with NAN or inf results in NAN or inf output. 
#                                             # method='direct',
#                                             )
#         convolved_vy = signal.convolve(da_vy_yr.values, window_kernel,
#                                             mode='same', ## runtimewarning: Use of fft convolution on input with NAN or inf results in NAN or inf output. 
#                                             # method='direct',
#                                             )
#         da_vx_expfilt_yr = da_vx_yr.copy(data=convolved_vx)
#         da_vy_expfilt_yr = da_vy_yr.copy(data=convolved_vy)
#         da_list_vx.append(da_vx_expfilt_yr)
#         da_list_vy.append(da_vy_expfilt_yr)

#     da_vx_expfilt = xr.concat(da_list_vx, dim='time')
#     da_vy_expfilt = xr.concat(da_list_vy, dim='time')

#     da_v = (np.sqrt(da_vx_expfilt**2 + da_vy_expfilt**2)).rename('velocity')

#     ## show average fields (no smoothing)
#     fig,axs=plt.subplots(1,3,figsize=(16,4))
#     # fig,axs=plt.subplots(1,4,figsize=(21,4))
#     da_vx_expfilt.mean(dim='time').plot.imshow(ax=axs[0], vmin=-50, vmax=50, cmap="PiYG")
#     da_vy_expfilt.mean(dim='time').plot.imshow(ax=axs[1], vmin=-50, vmax=50, cmap="PiYG")
#     da_v.mean(dim='time').plot.imshow(ax=axs[2], vmin=0, vmax=100, cmap='viridis')
#     # ## add kernel
#     # h = axs[3].imshow(window_kernel, vmin=0, vmax=wmax, cmap='viridis')
#     # fig.colorbar(h, ax=axs[3], location='right', shrink=0.7, )#anchor=(0, 0.3), )
#     # axs[3].set_axis_off(); axs[3].set_title(f'exp.filter: k={ksize}, decay={exp_decay_length}')

#     axs[0].set_title('vx average (m/yr)')
#     axs[1].set_title('vy average (m/yr)')
#     axs[2].set_title('v average (m/yr)')
#     [ax.set_aspect('equal') for ax in axs]

#     fig.suptitle(f'Exponential filter {ksize}x{ksize}px, before average ')
#     # fig.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
#     #                 f'Hofsjokull_velocity_selectYears_expfilt_{ksize}x{ksize}.jpg', dpi=300)
#     # raise RuntimeError()
# # fig1.savefig('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
# #                     f'Hofsjokull_velocity_expfilt_wieghts.jpg', dpi=300)

# #  JPG to GIF for smoothing
# import glob
# jpg_list = sorted(glob.glob('/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
#                      'Hofsjokull_velocity_selectYears_expfilt_*.jpg'))
# ## move 11x11 to end of list instead of second index
# jpg_list = jpg_list[1:]+[jpg_list[0]]
# jpg_list.insert(0,'/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/' + \
#                 f'Hofsjokull_velocity_selectYears_mean.jpg')

# ## open images and create gif

# def create_gif(files, 
#                output_path, 
#                delay=25,
#                pause_first_frame=False):
#     " delay is in 100ths of a second. So delay of 100 means 1 second"
#     " delay of 25 means 4 frames per second "
#     import subprocess
    
#     # Build magick command: base command is  : magick -delay 25 -dispose previous "${files_f081[@]}" 
#     # And to do with a pause at the end,  do : magick -delay 25 -dispose previous "${files_f081[@]}" -delay 100 "${files_f081[-1]}" out_F081.gif

#     magick_path = "/opt/homebrew/bin/magick"
#     cmd = [magick_path, "-dispose", "previous"] # main gif setup, use -dispose so that each frame replaces the previous rather than stacking
#     # cmd = ["magick", "-dispose", "previous"] # main gif setup, use -dispose so that each frame replaces the previous rather than stacking
    
#     delay_last = str(4*delay)
#     delay = str(delay)
#     if pause_first_frame:
#         delay_first = delay_last
    
#     if pause_first_frame and len(files) > 0:
#         # Add first frame with longer delay (pause)
#         cmd.extend(["-delay", delay_first, str(files[0])])  # 1 second pause on first frame
#         # Add remaining frames with normal delay
#         cmd.extend(["-delay", delay])
#         cmd.extend([str(f) for f in files[1:]])  # skip first frame since we already added it
#     else:
#         # Normal delay for all frames
#         cmd.extend(["-delay", delay])
#         cmd.extend([str(f) for f in files])
    
#     # Pause at end
#     cmd.extend(["-delay", delay_last, str(files[-1]), output_path]) # pause at end
    
#     # print(f".. creating {output_name}")
#     print(f"Running command: {' '.join(cmd)}")
#     subprocess.run(cmd)

# gif_name = 'Hofsjokull_velocity_expfilt_ksizes.gif'
# outpath = '/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/WP1_data/figures/'
# gif_file = os.path.join(outpath, gif_name)
# # if not os.path.exists( gif_file ):
# # create_gif(jpg_list, gif_file, delay=100, 
# #            pause_first_frame=False) # make gif; pauses at last frame.

