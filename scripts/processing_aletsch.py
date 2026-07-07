#%% Check and pre-processing of data

# M. Izeboud, June 2026

import numpy as np
import geopandas as gpd
import xarray as xr
import os
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray #  activates .rio accessor of xarray
import warnings

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Aletsch/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Aletsch/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Aletsch/'

import datafunctions as datafuncs


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

#%%

''' ##################################
Elevation bins
--> 50 m
--> based on earliest DEM if multiple available (assuming glacier is retreating, so earliest DEM has highest elevations)
##################################
'''


## multiple DEMs: use first to do the binning, but use the min-max range of both to define bin range
hmin = np.min([da_dem17.min().item(), da_dem23.min().item()]) 
hmax = np.max([da_dem17.max().item(), da_dem23.max().item()]) 
da_elev_bins, elev_bin_edges = datafuncs.dicretize_elevation_bins(da_dem17, 
                                                     hmin=hmin, hmax=hmax,
                                                     binstep=50)


## save to CLEAN directory
fname = 'aletsch_elev-bins.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    da_elev_bins.rename('elevation_bins').rio.to_raster(os.path.join(path2data_clean, fname))


#%%
''' ##################################
Bedrock
##################################
'''

assert da_dem17.rio.crs == da_H17.rio.crs, "CRS of DEM and thickness grid do not match"
assert da_dem17.rio.resolution() == da_H17.rio.resolution(), "Resolution of DEM and thickness grid do not match"
## thickness tif grid is smaller than DEM, so reproject and match grid
## they already have the same resolution and CRS, so minimal data manipulation occurs.
da_H17_matched = datafuncs.reproject_match_grid(da_dem17, da_H17, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

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
    
#%%
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


### Save cleaned velocity data to CLEAN directory

## write attributes / clear existing
attrs_velo_4326 = {
              'units':'m/year',
              'crs':'EPSG:4326',
              'timestamp':'2011-2019',
              'description':'median velocity over 2011-2019.'
             }

da_vx_myear.attrs = attrs_velo_4326
da_vx_myear.attrs['long_name'] = 'surface ice velocity (x-component)'
da_vy_myear.attrs = attrs_velo_4326
da_vy_myear.attrs['long_name'] = 'surface ice velocity (y-component)'
da_vx_stdev_myear.attrs = attrs_velo_4326
da_vx_stdev_myear.attrs['long_name'] = 'surface ice velocity (x-component) standard deviation'
da_vx_stdev_myear.attrs['description'] = 'standard deviation of ice velocity (2011-2019)'
da_vy_stdev_myear.attrs = attrs_velo_4326
da_vy_stdev_myear.attrs['long_name'] = 'surface ice velocity (y-component) standard deviation'
da_vy_stdev_myear.attrs['description'] = 'standard deviation of ice velocity (2011-2019)'


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*angle from rectified to skew grid parameter lost in conversion to CF.*",
        category=UserWarning,
        module="pyproj.*",
    )


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
# Rotate velocity components
# -----------------------------
# da_vx_rot, da_vy_rot = transform_velocity_components_epsg4326_to_epsg2056(
#     da_vx, da_vy, src_epsg="4326", dst_epsg="2056"
# ) ## the m/day version
da_vx_rot, da_vy_rot = datafuncs.transform_velocity_components_epsg4326_to_epsg2056(
    da_vx_myear, da_vy_myear, src_epsg="4326", dst_epsg="2056"
)

# -----------------------------
# Reproject underlying rasters to EPSG:2056
# -----------------------------
da_vx_2056 = da_vx_rot.rio.reproject("EPSG:2056")
da_vy_2056 = da_vy_rot.rio.reproject("EPSG:2056")

## stdev raster need to be reprojected but not rotated since its scalar fields
da_vx_stdev_2056 = da_vx_stdev_myear.rio.reproject("EPSG:2056")
da_vy_stdev_2056 = da_vy_stdev_myear.rio.reproject("EPSG:2056")


## resample reprojected velocity to 10 m

da_vx_10m = datafuncs.reproject_match_grid(da_dem17, da_vx_2056, 
                                 resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_vy_10m = datafuncs.reproject_match_grid(da_dem17, da_vy_2056, 
                                 resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)


# -----------------------------
# Reprojected velocities are 64.17 m resoltuion; 
# reproject to a regular grid of 60 m.
# -----------------------------
da_velo_60m = datafuncs.create_regular_dummy_grid(da_vx_2056, grid_res=60, crs="EPSG:2056", unit='m')
da_vx_2056 = datafuncs.reproject_match_grid(da_velo_60m, da_vx_2056, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_vy_2056 = datafuncs.reproject_match_grid(da_velo_60m, da_vy_2056, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_vx_stdev_2056 = datafuncs.reproject_match_grid(da_velo_60m, da_vx_stdev_2056, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_vy_stdev_2056 = datafuncs.reproject_match_grid(da_velo_60m, da_vy_stdev_2056, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)


# -----------------------------
# Save output
# -----------------------------

## write attributes / clear existing
attrs_velo = {
              'units':'m/year',
              'crs':'EPSG:2056',
              'timestamp':'2011-2019',
              'description':'median velocity over 2011-2019.'
             }

da_vx_2056.attrs = attrs_velo 
da_vx_2056.attrs['long_name'] = 'surface ice velocity (x-component)'
da_vy_2056.attrs = attrs_velo
da_vy_2056.attrs['long_name'] = 'surface ice velocity (y-component)'
da_vx_stdev_2056.attrs = attrs_velo
da_vx_stdev_2056.attrs['long_name'] = 'surface ice velocity (x-component) standard deviation'
da_vx_stdev_2056.attrs['description'] = 'standard deviation of ice velocity (2011-2019)'
da_vy_stdev_2056.attrs = attrs_velo
da_vy_stdev_2056.attrs['long_name'] = 'surface ice velocity (y-component) standard deviation'
da_vy_stdev_2056.attrs['description'] = 'standard deviation of ice velocity (2011-2019)'


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*angle from rectified to skew grid parameter lost in conversion to CF.*",
        category=UserWarning,
        module="pyproj.*",
    )

    ## save to CLEAN directory
    fname_vx = 'aletsch_vx_EPSG2056.tif'
    fname_vy = 'aletsch_vy_EPSG2056.tif'
    if not os.path.exists(os.path.join(path2data_clean, fname_vx)):
        da_vx_2056.rio.to_raster(os.path.join(path2data_clean, fname_vx))
        da_vy_2056.rio.to_raster(os.path.join(path2data_clean, fname_vy))
    else:
        print(f"File {fname_vx} already exists in homogenized data directory. Skipping save.")
        
    fname_vx_stdev = 'aletsch_vx_stddev_EPSG2056.tif'
    fname_vy_stdev = 'aletsch_vy_stddev_EPSG2056.tif'
    if not os.path.exists(os.path.join(path2data_clean, fname_vx_stdev)):
        da_vx_stdev_2056.rio.to_raster(os.path.join(path2data_clean, fname_vx_stdev))
        da_vy_stdev_2056.rio.to_raster(os.path.join(path2data_clean, fname_vy_stdev))


fig,axs =plt.subplots(2,2, figsize=(12,10))
da_vx.plot.imshow(ax=axs[0,0], vmin=-1, vmax=1, cmap='RdBu_r');
axs[0,0].set_title('vx 4326 (m/day)')
da_vx_2056.plot.imshow(ax=axs[0,1], vmin=-1*365.25, vmax=1*365.25, cmap='RdBu_r'); 
axs[0,1].set_title('vx 2056 (m/year)'); axs[0,1].set_xlabel('x [m]') #
da_vy.plot.imshow(ax=axs[1,0], vmin=-1, vmax=1, cmap='RdBu_r');
axs[1,0].set_title('vy 4326 (m/day)')
da_vy_2056.plot.imshow(ax=axs[1,1], vmin=-1*365.25, vmax=1*365.25, cmap='RdBu_r'); 
axs[1,1].set_title('vy 2056 (m/year)'); axs[1,1].set_xlabel('x [m]') # x [m]


#%% 

''' ##################################
ASSEMBLE NETCDF
- DEM
- thickness
- vx
- vy
- dhdt
- bedrock
- outline: as mask
Fill NaN values with another nodata value 
--> can be 0 for all variables except DEM and bedrock, so need to make sure these don't have missing values
##################################
'''
gdf_outline1 = gpd.read_file(os.path.join(path2data_clean,'aletsch_outline_20170901.shp'))
gdf_outline2 = gpd.read_file(os.path.join(path2data_clean,'aletsch_outline_20230823.shp'))

## burn outline into raster mask
da_dummy = da_dem17.copy(data=np.ones_like(da_dem17.values))
da_outline_mask1 = (da_dummy*2017).rio.clip(gdf_outline1.geometry, gdf_outline1.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
da_outline_mask2 = (da_dummy*2023).rio.clip(gdf_outline2.geometry, gdf_outline2.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)
## fill mask NaN with 0
da_outline_mask = xr.concat([da_outline_mask1, da_outline_mask2], dim='time'
                            )#.fillna(0) # fill NaN values with 0 (outside outline)
da_outline_mask = (da_outline_mask.copy()
                   .max(dim='time') 
                   .rename('icemask')
                   .assign_attrs({'long_name':'Glacier Outline Mask',
                                  'units':'year',
                                  'crs':'EPSG:2056',
                                  'timestamp':'20170901 and 20230823',
                                  'description': 'Value is max year of valid glaciated pixel; 0 for non-glaciated pixels.',
                                #   'nodata': 0
                                  })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs('EPSG:2056')
)


## set attributes

# DEM and Bedrock: should not have NaN values 
if da_dem17.isnull().any():
    raise ValueError("DEM has NaN values.")
else: da_dem_10m = (da_dem17.copy()
                .rename('DEM') 
                .assign_attrs({'long_name':'Elevation',
                                'units':'m',
                                'crs':'EPSG:2056',
                                'timestamp':'20170901',
                                'description':'Elevation data.'
                                })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs('EPSG:2056')
    )

if da_bedrock17_filled.isnull().any():
    raise ValueError("Bedrock has NaN values, cannot assign nodata value of 0.")
else: da_bedrock_10m = (da_bedrock17_filled.copy()
                    .rename('bedrock')
                    .assign_attrs({'long_name':'Bedrock Elevation',
                                   'units':'m',
                                   'crs':'EPSG:2056',
                                   'timestamp':'20170901',
                                   'description':'bedrock calculated from DEM and H. Where H=0, bedrock=DEM.'
                                   })
                    # .drop_vars('spatial_ref')
                    .rio.write_crs('EPSG:2056')
    )

## thickness, dhdt, velo: can fill NaN with 0
da_thickness_10m = (da_H17_matched.fillna(0).copy()
                    .rename('thickness')
                    .assign_attrs({'long_name':'Ice Thickness',
                                   'units':'m',
                                   'crs':'EPSG:2056',
                                   'timestamp':'20170901',
                                   'description':'ice thickness interpolated from GPR. Missing/NaN values were filled with 0.',
                                   'nodata': 0})
                # .drop_vars('spatial_ref')
                .rio.write_crs('EPSG:2056')
                    )
da_dhdt_10m = (da_dhdt.fillna(0).copy()
               .rename('dhdt')
               .assign_attrs({'long_name':'Surface Elevation Change',
                              'units':'m/year',
                              'crs':'EPSG:2056',
                              'timestamp':'20170901-20230823',
                              'description':'annual elevation change. Missing/NaN values were filled with 0.',
                              'nodata': 0})
                # .drop_vars('spatial_ref')
                .rio.write_crs('EPSG:2056')
               )

da_vx_10m = (da_vx_10m.fillna(0).copy()
             .rename('vx')
             .assign_attrs({'long_name': 'surface ice velocity (x-component)',
                            'units':'m/year',
                            'crs':'EPSG:2056',
                            'timestamp':'2011-2019',
                            'description':'median velocity over 2011-2019. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                # .drop_vars('spatial_ref')
                .rio.write_crs('EPSG:2056')
)

da_vy_10m = (da_vy_10m.fillna(0).copy()
             .rename('vy')
             .assign_attrs({'long_name': 'surface ice velocity (y-component)',
                            'units':'m/year',
                            'crs':'EPSG:2056',
                            'timestamp':'2011-2019',
                            'description':'median velocity over 2011-2019. Missing/NaN values were filled with 0.',
                            'nodata': 0
                            })
                .drop_vars('spatial_ref')
                .rio.write_crs('EPSG:2056')
)

da_elev_bins_10m = (da_elev_bins.copy()
                   .rename('elevation_bins')
                   .assign_attrs({'long_name':'Elevation Bins',
                                  'units':'m',
                                  'crs':'EPSG:2056',
                                  'timestamp':'20170901',
                                  'description': f'Discretized elevation values into bins of 50 m. Using lowest (left-edge) value for each bin.'
                                  })
                    .rio.write_crs('EPSG:2056')
)


da_var_list = [ da_bedrock_10m,
                da_dem_10m, 
                da_elev_bins_10m,
                da_thickness_10m,
                da_dhdt_10m,
                da_vx_10m,
                da_vy_10m,
                da_outline_mask,
                ]
assert all(da.rio.crs == da_dem_10m.rio.crs for da in da_var_list), "Not all variables have the same CRS"
assert all(da.rio.resolution() == da_dem_10m.rio.resolution() for da in da_var_list), "Not all variables have the same resolution"
assert all(da.shape == da_dem_10m.shape for da in da_var_list), "Not all variables have the same shape"

ds_aletsch_10m = (xr.combine_by_coords(da_var_list, 
                                       compat='no_conflicts')
                    .assign_attrs({'title':'homogenized glacier observation data',
                               'grid_resolution':'10 m',
                               'description':'see attributes of each variable',
                               'timestamp':'2017-2023',
                               'nodata': 0,
                    })
                    .rio.set_spatial_dims(x_dim="x", y_dim="y") # Make sure spatial dims are known
                    # Write CRS and CF grid mapping to the whole dataset
                    .rio.write_crs("EPSG:2056")
                    .rio.write_grid_mapping("spatial_ref")
                    .rio.write_transform()
)

# Force each real data variable to point to spatial_ref
for var in ds_aletsch_10m.data_vars:
    if var != "spatial_ref":
        ds_aletsch_10m[var].attrs["grid_mapping"] = "spatial_ref"

'''# check values by plotting
'''
# for var in ds_aletsch_10m.data_vars:
#     da_plot = ds_aletsch_10m[var]
#     # print(da_plot)
#     fig,ax=plt.subplots(figsize=(6,5))
    
#     da_plot.plot.imshow(ax=ax)
# ds_aletsch_10m
# Set the no-data value in the encoding dictionary

'''## save to netcdf'''

## encoding settings for compression and data type; same for all variables
comp = {"zlib": True, 
        "complevel": 5,  ## level of compression; higher number = more compression but slower read/write
        "dtype": "float32", ## 7 digits of precision 
        # "_FillValue": np.float32(-999),
        }
encoding = {var: comp for var in ds_aletsch_10m.data_vars if var != "spatial_ref"}  # Exclude spatial_ref from encoding
encoding["spatial_ref"] = {}  # No compression for spatial_ref

fname_nc = 'aletsch_glacier_observations.nc'

try:
    print('--> saving homogenized data to netcdf; overwriting if file exists')
    ds_aletsch_10m.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding ## don't use encoding; although it compresses data size, it loses CRS info 
    )
    ds_aletsch_10m.close()

except PermissionError:
    print('--> CHECK INPUT WINDOW')
    answ = input(f"PermissionError to write {fname_nc}. Input Y to overwrite")
    if answ == 'Y' or answ == 'y':
        print('..removing existing and re-saving file')
        os.remove(os.path.join(path2data_homog, fname_nc))
        ds_aletsch_10m.to_netcdf(os.path.join(path2data_homog, fname_nc), 
                            mode='w', format='NETCDF4', 
                            engine='netcdf4',
                            encoding=encoding ## don't use encoding; although it compresses data size, it loses CRS info 
        )
        ds_aletsch_10m.close()
    else: print('..aborted saving file')

#%% check values by loading saved data & plotting

with xr.open_dataset(
        os.path.join(path2data_homog, fname_nc),
        decode_coords="all" # decode_coords="all" is important when reopening NetCDFs with rioxarray-style CRS metadata; otherwise the CRS may appear to be missing.
    ) as ds_glacier_loaded:
    
    print('CRS:', ds_glacier_loaded.rio.crs)
    print('spatial_ref attrs:', ds_glacier_loaded["spatial_ref"].attrs)
    assert ds_glacier_loaded.rio.crs is not None, "CRS is missing in the loaded dataset"

## check values by plotting
fig,axs=plt.subplots(2,4, figsize=(16,8))
row,col = 0,0
for var, cmap in zip(ds_glacier_loaded.data_vars, ['cividis','cividis','cividis','Blues',
                                                   'RdBu','PiYG','PiYG','viridis']):
    if var == 'spatial_ref':
        continue  # Skip plotting the spatial_ref variable
    if var == 'dhdt':
        vmin,vmax = -5,5;
    elif var == 'vx' or var == 'vy':
        vmin,vmax = -100,100;
    else:
        vmin,vmax=None,None
    da_plot = ds_glacier_loaded[var]
    # print(da_plot)
    # fig,ax=plt.subplots(figsize=(6,5))
    ax=axs[row,col]
    da_plot.plot.imshow(ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar_kwargs={'shrink': 0.7})
    ax.set_title(var)
    col+=1
    if col >= 4:
        col = 0
        row += 1
[ax.set_aspect('equal') for ax in axs.flatten()];
[ax.set_axis_off() for ax in axs.flatten()];

fig.savefig(os.path.join(path2data_homog, 'aletsch_netcdf_vars.png'), dpi=300)


# ds_aletsch_10m
# %% Also check all CLEANED files 

files_cleaned = [f for f in os.listdir(path2data_clean) if f.endswith('.tif')]


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=".*angle from rectified to skew grid parameter lost in conversion to CF.*",
        category=UserWarning,
        module="pyproj.*",
    )

    for filename in files_cleaned:
        file = os.path.join(path2data_clean, filename)
        with xr.open_dataarray(file, engine='rasterio') as da:
            print('------')
            print(f"File: {filename}")
            print(f"  CRS: {da.rio.crs}")
            print(f"  Resolution: {da.rio.resolution()}")
            display(da.attrs)
            # print(f"  Shape: {da.shape}")
            # print(f"  Min/Max: {np.nanmin(da.values)}/{np.nanmax(da.values)}")
            # print(f"  NaN count: {np.isnan(da.values).sum()}")
        # input('Press Enter to continue to the next file...')

# #%% Load synthetic Aletsch case

# path2synth = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Synthetic2/'

# ds_synth_aletsch = xr.open_dataset(
#     os.path.join(path2synth, 'aletsch_ss_100year.nc'), 
#     decode_coords="all",
#     # engine='rasterio'
#     )
# print(f"  CRS: {ds_synth_aletsch.rio.crs}")
# print(f"  Resolution: {ds_synth_aletsch.rio.resolution()}")

# # print(f"  Shape: {da.shape}")
# # print(f"  Min/Max: {np.nanmin(da.values)}/{np.nanmax(da.values)}")
# # print(f"  NaN count: {np.isnan(da.values).sum()}")


# %%
