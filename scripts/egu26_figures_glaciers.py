#%% Visualize datafields of a few glaciers for EGU26 poster
# little data processing, just mapping and plotting

import os
from shapely import LineString
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import geopandas as gpd
import rasterio as rio
import geopandas as gdp

# os.chdir('/Users/mizeboud/Documents/Documents_mizeboud/PostDoc/2D-SMB/code/SMB-from-remote-sensing/scripts/')
# import myFunctions as myf

target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution) --> where I have all my input/bruteForceOutput in
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes
swiss_crs_morteratsch = 'EPSG:2056' ##  CH1903+ / LV95 

# data_dir = '/Users/mizeboud/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data_iCloud/SMB2D/'
data_dir = '/Users/mizeboud/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/ContinuIX/ContinuIX_WP1_data/'
homedir = '/Users/mizeboud/Documents/Documents_mizeboud/Projects/ContinuIX/'

my_palette = ['#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a

## import scientificcolormaps 
from cmcrameri import cm
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 100, 100)[None, :]
# plt.imshow(x, aspect='auto', cmap=cm.tokyo) #


#%%

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



#%% Hofsjokull: outline -- RGI v6
crs_hofsj = 'EPSG:3057'

outline_hofsj = os.path.join(data_dir,'other_data/outline_Hofsjokull_RGIv6_4326.shp')
gdf_hofsj = gpd.read_file(outline_hofsj).to_crs(crs_hofsj)

## merge all geometries into one (dissolve) to have a single outline for clipping
# gdf_hofsj_outline = gdp.unary_union(gdf_hofsj) # dissolve by RGIId to merge all geometries into one
# gdf_hofsj_outline = gdf_hofsj.geometry.union()# Single merged geometry (Polygon or MultiPolygon)
hofsj_union = gdf_hofsj.geometry.union_all()   # GeoPandas/Shapely recent versions
# or (older API): hofsj_union = gdf_hofsj.unary_union
gdf_hofsj_union = gpd.GeoDataFrame(geometry=[hofsj_union], crs=gdf_hofsj.crs)
gdf_hofsj_union.boundary.plot()


#%% Hofsjokull : DEM AND BEDROCK
#  Icelandic reference system ISN93 (EPSG:3057) 
crs_hofsj = 'EPSG:3057'
DEM_files = ['Hofsjokull_20131013_zmae.tif','hofs_oct2020_jitcor_mosaicblend.tif','hofs_sep2023_jitcor_mosaicblend.tif' ]
path2dem = os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_dem/')
# da_hofsj_dem = xr.open_mfdataset([os.path.join(path2dem, f) for f in DEM_files], engine='rasterio', join='outer',
#                                   combine='nested', concat_dim='file'
#                                   ).isel(band=0).drop_vars('band').rename({'band_data':'DEM'})

## at 2m spatial resolution
da_hofsj_2013 = xr.open_dataset(os.path.join(path2dem, DEM_files[0]), engine='rasterio').isel(band=0).drop_vars('band').rename({'band_data':'DEM'})
da_hofsj_2020 = xr.open_dataset(os.path.join(path2dem, DEM_files[1]), engine='rasterio').isel(band=0).drop_vars('band').rename({'band_data':'DEM'})
da_hofsj_2023 = xr.open_dataset(os.path.join(path2dem, DEM_files[2]), engine='rasterio').isel(band=0).drop_vars('band').rename({'band_data':'DEM'})

assert da_hofsj_2020.rio.crs is not None, "CRS should not be set yet"
assert da_hofsj_2020.rio.crs == da_hofsj_2023.rio.crs == da_hofsj_2013.rio.crs, "CRS should be the same for all DEM files"
print('CRS of Hofsjokull DEM files:', da_hofsj_2020.rio.crs)

## downsample to 50 m resolution for faster plotting
ds_hofsj_dem = xr.concat([da_hofsj_2013, da_hofsj_2020, da_hofsj_2023], dim='time', join='outer'
                         ).assign_coords(time=[2013, 2020, 2023])
ds_hofsj_dem_50m = ds_hofsj_dem.rio.reproject(crs_hofsj, resolution=50, 
                                              resampling=rio.enums.Resampling.nearest, 
                                            #   resampling=rio.enums.Resampling.bilinear, 
                                              nodata=np.nan)

# %%
ds_hofsj_dem_50m['DEM'].plot.imshow(cmap='terrain', col='time', col_wrap=3)
## add outlien to each of the subplots
fig, axs = plt.subplots(figsize=(15,9), ncols=3, sharex=True, sharey=True)
for i, ax in enumerate(axs):
    ds_hofsj_dem_50m['DEM'].isel(time=i
                  ).plot.imshow(ax=ax, cmap='terrain', vmin=700, vmax=1800,
                            #    add_colorbar=False if i<2 else True, # only add colorbar to the last subplot
                            cbar_kwargs={'fraction':0.04, 'label':'DEM (m)'} )#if i == 2 else None) # only add colorbar to the last subplot
    
    gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
    gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
    ax.set_aspect('equal') #
    ax.set_axis_off()
    ax.set_title(f"DEM {ds_hofsj_dem_50m.time.values[i]}")
plt.tight_layout()

#%%
da_hofsj_bedrock = xr.open_dataset(os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_bedrock/Ho-botn-land-1983-gert2026-200x200.tif'), engine='rasterio'
                                   ).isel(band=0).drop_vars('band').rename({'band_data':'bedrock'})['bedrock']
# assert da_hofsj_bedrock.rio.crs == da_hofsj_2013.rio.crs, f"CRS should match between bedrock and DEM; are {da_hofsj_bedrock.rio.crs} and {da_hofsj_2013.rio.crs}"
print(da_hofsj_bedrock.rio.crs, da_hofsj_bedrock.rio.resolution())
da_hofsj_bedrock.plot.imshow(cmap='terrain')

da_hofsj_bedrock_50m = reproject_match_grid(ds_hofsj_dem_50m['DEM'], da_hofsj_bedrock, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

#%% Hofsjokull THICKNESS

da_hofsj_H = ds_hofsj_dem_50m['DEM'] - da_hofsj_bedrock_50m
fig, axs = plt.subplots(figsize=(15,9), ncols=3, sharex=True, sharey=True)
for i, ax in enumerate(axs):
    da_hofsj_H.isel(time=i
              ).plot.imshow(ax=ax, cmap='Blues', vmin=0,vmax=500,
                    #    add_colorbar=False if i<2 else True, # only add colorbar to the last subplot
                    cbar_kwargs={'fraction':0.04, 'label':'DEM (m)'} )#if i == 2 else None) # only add colorbar to the last subplot
    
    gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
    gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
    ax.set_aspect('equal') #
    ax.set_axis_off()
    ax.set_title(f"Thickness {da_hofsj_H.time.values[i]}")
plt.tight_layout()

#%%
## average
da_hofsj_Havg = da_hofsj_H.mean(dim='time')
## clip to outline
da_hofsj_Havg = da_hofsj_Havg.rio.clip(gdf_hofsj.geometry, gdf_hofsj.crs, drop=False) # drop false to keep the same grid and have nans outside the outline


fig,ax=plt.subplots(figsize=(15,10))

da_hofsj_Havg.plot.imshow(ax=ax, cmap='Blues', vmin=0, vmax=500, cbar_kwargs={'fraction':0.02, 'label':'Ice thickness (m)'})
gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

## save to tiff
# filename = os.path.join(data_dir,'Output/06_Hofsjokull/Hofsjokull_thickness_avg_50m_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_hofsj_Havg.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Hofsjokull_thickness_avg_50m_clipped.pdf'), bbox_inches='tight')

#%% Hofsjokull ELEVATION CHANGE from DEM change
ds_hofsj_dem_50m = ds_hofsj_dem_50m.rio.clip(gdf_hofsj_union.geometry, gdf_hofsj_union.crs, drop=False) # clip to union of all outlines to have the same grid for all DEMs and avoid nans outside the outline
da_hofsj_dhdt_1320 =(ds_hofsj_dem_50m['DEM'].sel(time=2020) - ds_hofsj_dem_50m['DEM'].sel(time=2013)) / (2020-2013)
da_hofsj_dhdt_1323 = (ds_hofsj_dem_50m['DEM'].sel(time=2023) - ds_hofsj_dem_50m['DEM'].sel(time=2013)) / (2023-2013)
da_hofsj_dhdt_2023 = (ds_hofsj_dem_50m['DEM'].sel(time=2023) - ds_hofsj_dem_50m['DEM'].sel(time=2020)) / (2023-2020)
da_hofsj_dhdt = xr.concat([da_hofsj_dhdt_1320, da_hofsj_dhdt_1323, da_hofsj_dhdt_2023], dim='time').assign_coords(time=['2013-2020', '2013-2023', '2020-2023'])

# da_hofsj_dhdt.plot.imshow(cmap='RdBu', col='time', col_wrap=3, vmin=-8, vmax=8, cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})

da_plot = da_hofsj_dhdt.copy()
fig, axs = plt.subplots(figsize=(15,9), ncols=3, sharex=True, sharey=True)
for i, ax in enumerate(axs):
    da_plot.isel(time=i
              ).plot.imshow(ax=ax, cmap='RdBu', vmin=-4, vmax=4,
                cbar_kwargs={'fraction':0.04, 'label':'Elevation change (m/yr)'} )#if i == 2 else None) # only add colorbar to the last subplot
    gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
    gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
    ax.set_aspect('equal') #
    ax.set_axis_off()
    ax.set_title(f"dH/dt {da_plot.time.values[i]}")
plt.tight_layout()

da_hofsj_dhdt_avg = da_hofsj_dhdt.mean(dim='time')

fig,ax=plt.subplots(figsize=(15,10))

da_hofsj_dhdt_1323.plot.imshow(ax=ax, cmap='RdBu', vmin=-5, vmax=5, 
                               cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})
gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# ## save to tiff
# filename = os.path.join(data_dir,'Output/06_Hofsjokull/Hofsjokull_dhdt_2013-2023_50m_clipped.tif')
# # if not os.path.exists(os.path.dirname(filename)):
# da_hofsj_dhdt_1323.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Hofsjokull_dhdt_2013-2023_50m_clipped.pdf'), bbox_inches='tight')


#%% Hofsjokull VELOCITIES
## 100 m resolution; in METER / DAY; need to convert to m/yr by multiplying by 365.25

filelist_velo_hofsj = sorted([f for f in os.listdir(os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/')) if f.endswith('.tif')])
## get easting and northign
filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'easting' in file or 'northing' in file] # exclude stdev files
filelist_velo_hofsj = [file for file in filelist_velo_hofsj if 'stddev' not in file] # exclude stdev files 
filelist_vx = [file for file in filelist_velo_hofsj if 'easting' in file]
filelist_vy = [file for file in filelist_velo_hofsj if 'northing' in file]
years_vx = [2014,2016,2017,2019,2020,2021,2022,2023]
years_vy = [2014,2016,2017,2018,2019,2020,2021,2022,2023]

## open dataset
ds_hofsj_vx = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vx], 
                                engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                ).assign_coords(time=years_vx).isel(band=0).drop_vars('band').rename({'band_data':'vx'})
ds_hofsj_vy = xr.open_mfdataset([os.path.join(data_dir,'06_Hofsjokull/Hofsjokull_velocities/', f) for f in filelist_vy],
                                 engine='rasterio', join='outer', combine='nested', concat_dim='time'
                                 ).assign_coords(time=years_vy).isel(band=0).drop_vars('band').rename({'band_data':'vy'})

assert ds_hofsj_vx.rio.crs == ds_hofsj_vy.rio.crs, f"CRS should match between vx and vy; are {ds_hofsj_vx.rio.crs} and {ds_hofsj_vy.rio.crs}"
print('CRS of Hofsjokull velocity files:', ds_hofsj_vx.rio.crs)

## calculate velocity magnitude
da_hofsj_v = (np.sqrt(ds_hofsj_vx['vx']**2 + ds_hofsj_vy['vy']**2)*365.25).rename('velocity')

## plot
da_hofsj_v.plot.imshow(cmap='viridis', col='time', col_wrap=4, vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
## get axes
axs = plt.gcf().axes[:-1] # exclude colorbar axis
[gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]
[ax.set_axis_off() for ax in axs]

# ds_hofsj_vx['vx'].plot.imshow(cmap='viridis', col='time', col_wrap=4, vmin=-100, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Easting velocity (m/yr)'})
#%%
## smooth data with spatial rolling window to reduce noise
da_hofsj_v_smooth_median = da_hofsj_v.rolling(x=10, y=10, center=True).median()

## plot
da_hofsj_H_100m = reproject_match_grid(da_hofsj_v, da_hofsj_Havg, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
# da_thickness = da_hofsj_H_100m.copy(data = np.ones_like(da_hofsj_H_100m) * 100 )# uniform scaling field for exponential filter, fake thickness 
da_thickness = da_hofsj_H_100m.copy()
Nlength = 1
da_hofsj_v_smooth_exp = myf.exp_smooth_numba(da_hofsj_v.isel(time=-1), da_thickness, Nlength, 
                     kernel_scale=None, min_ksize=2, max_ksize=10,
                     verbose=False)

fig,axs=plt.subplots(1,3, figsize=(15,5))
ax=axs[0]; 
da_hofsj_v.isel(time=-1).plot.imshow(ax=ax, cmap='viridis', vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
ax.set_title(f'raw velocity ({da_hofsj_v.isel(time=-1).time.values})')
ax=axs[1];
da_hofsj_v_smooth_exp.plot.imshow(ax=ax, cmap='viridis', vmin=0, vmax=100, 
                               cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
ax.set_title(f'exponential filter ({da_hofsj_v.isel(time=-1).time.values})')
ax=axs[2]; 
da_hofsj_v_smooth_median.isel(time=-1).plot.imshow(ax=ax,cmap='viridis', vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Smoothed velocity (m/yr)'})
ax.set_title(f'median filter ({da_hofsj_v.isel(time=-1).time.values})')
[ax.set_aspect('equal') for ax in axs]
[ax.set_axis_off() for ax in axs]
[gdf_hofsj_union.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1) for ax in axs]

#%% Hofsjokul -- plot velocity after exponential smoothing AND average

da_list = []
for y in da_hofsj_v.time.values:
    da_hofsj_v_smooth_exp = myf.exp_smooth_numba(da_hofsj_v.sel(time=y), da_hofsj_H_100m, Nlength, 
                     kernel_scale=None, min_ksize=2, max_ksize=10,
                     verbose=False)
    da_list.append(da_hofsj_v_smooth_exp)
da_hofsj_v_smoothed = xr.concat(da_list, dim='time').assign_coords(time=da_hofsj_v.time.values)

# da_hofsj_v_smoothed.plot.imshow(cmap='viridis', vmin=0, vmax=100, col='time', col_wrap=4, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})

## temporal average
da_hofsj_v_smoothed_avg = da_hofsj_v_smoothed.mean(dim='time', skipna=True)
## select only few years that seem similar
da_hofsj_v_smoothed_avg_1723 = da_hofsj_v_smoothed.sel(time=[2017,2019,2020,2023]).mean(dim='time', skipna=True)

# fig,ax=plt.subplots(figsize=(10,8))
# da_hofsj_v_smoothed_avg.plot.imshow(ax=ax,
#                    cmap='viridis', 
#                 #    cmap=cm.tokyo,
#                    vmin=0, vmax=100, 
#                    cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
# ax.set_title('average velocity (after exp.smooth)')
# ax.set_axis_off()

## save the velocity field with the selected years

fig,ax=plt.subplots(figsize=(15,10))
da_hofsj_v_smoothed_avg_1723.plot.imshow(ax=ax, 
                    #  cmap=cm.tokyo, 
                     cmap='turbo',
                     vmin=0, vmax=200, 
                    cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})
gdf_hofsj.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1)
gdf_hofsj_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# # # ## save to tiff
# filename = os.path.join(data_dir,'Output/06_Hofsjokull/Hofsjokull_v_smoothed_avg_1723_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_hofsj_v_smoothed_avg_1723.rio.to_raster(filename)
# ## save as img (pdf) for poster
fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Hofsjokull_v_smoothed_avg_1723_clipped.pdf'), bbox_inches='tight')


#%% 
'''# ########################################
# GEPATSCH
Outlines EPSG 25832
- 1 thickness file for 2006, 5m resolution
- 2 dh files for 2006-2017 and 2006-2018, 1m resolution
- bedrock file but not needed
- outlines: 6 years

#########################################
'''
import pandas as pd

files_gepatsch = sorted(os.listdir(os.path.join(data_dir,'11_Gepatschferner/')))
f_outlines_gep = sorted([f for f in files_gepatsch if 'outline' in f])

gdf_list = []
for f in f_outlines_gep:
    tmp = gpd.read_file(os.path.join(data_dir,'11_Gepatschferner/', f))
    tmp['year'] = f.split('.')[0].split('_')[-1] # extract year from filename, assuming format like 'outline_2013.shp'
    gdf_list.append (tmp)
gdf_gepatsch = gpd.GeoDataFrame( pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)

da_gepatsch_H = xr.open_dataset(os.path.join(data_dir,'11_Gepatschferner/gepatsch_h_2006.tif'), engine='rasterio',
                                   ).isel(band=0).drop_vars('band').rename({'band_data':'H'})['H']
print('CRS of Gepatsch thickness file:', da_gepatsch_H.rio.crs)
assert da_gepatsch_H.rio.crs == gdf_gepatsch.crs, f"CRS should match between thickness and outlines; are {da_gepatsch_H.rio.crs} and {gdf_gepatsch.crs}"

da_gepatsch_dhdt_0617 = xr.open_dataset(os.path.join(data_dir,'11_Gepatschferner/gepatsch_dh_2006-2017.tif'), engine='rasterio',
                                   ).isel(band=0).drop_vars('band').rename({'band_data':'dhdt_2006_2017'})['dhdt_2006_2017']
da_gepatsch_dhdt_0618 = xr.open_dataset(os.path.join(data_dir,'11_Gepatschferner/gepatsch_dh_2006-2018.tif'), engine='rasterio',
                                   ).isel(band=0).drop_vars('band').rename({'band_data':'dhdt_2006_2018'})['dhdt_2006_2018']
print('CRS of Gepatsch dhdt file:', da_gepatsch_dhdt_0617.rio.crs)
assert da_gepatsch_dhdt_0617.rio.crs == gdf_gepatsch.crs, f"CRS should match between dhdt and outlines; are {da_gepatsch_dhdt_0617.rio.crs} and {gdf_gepatsch.crs}"
## downsample to 5m resolution of H
da_gepatsch_dhdt_0617_5m = reproject_match_grid(da_gepatsch_H, da_gepatsch_dhdt_0617, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
da_gepatsch_dhdt_0618_5m = reproject_match_grid(da_gepatsch_H, da_gepatsch_dhdt_0618, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
## scale dhdt to annual values
da_gepatsch_dhdt_0617_5m = da_gepatsch_dhdt_0617_5m / (2017-2006)
da_gepatsch_dhdt_0618_5m = da_gepatsch_dhdt_0618_5m / (2018-2006)

## merge to one dataset since they are mosaic
da_gepatsch_dhdt_5m = da_gepatsch_dhdt_0617_5m.combine_first( da_gepatsch_dhdt_0618_5m)

## clip to largest outline of dhdt (2006)
da_gepatsch_dhdt_5m = da_gepatsch_dhdt_5m.rio.clip(gdf_gepatsch.loc[gdf_gepatsch['year']=='2006'].geometry, gdf_gepatsch.crs, drop=False)
## mask 0 values 
da_gepatsch_dhdt_5m = da_gepatsch_dhdt_5m.where(da_gepatsch_dhdt_5m != 0)

fig,ax=plt.subplots(figsize=(15,10))

# da_gepatsch_dhdt_0617_5m.plot.imshow(ax=ax, cmap='RdBu', vmin=-20, vmax=20, add_colorbar=False)
da_gepatsch_dhdt_5m.plot.imshow(ax=ax, cmap='RdBu', vmin=-5, vmax=5, 
                               cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})

highlight_year = '2017'
cmap_outlines = cm.glasgow_r
## discretize the colormap to have a different color for each year
y_range = 2023-1850
cmap_outlines_discrete = cmap_outlines(np.linspace(0, 1, 20)[::3]) # take only the last 6 colors for the 6 years
for idx, row in gdf_gepatsch.iterrows():
    row_gdf = gpd.GeoDataFrame([row], crs=gdf_gepatsch.crs)
    is_highlight = row['year'] == highlight_year
    row_gdf.boundary.plot(
        ax=ax,
        linestyle='-' if is_highlight else '--',
        # color=cmap_outlines_discrete[idx] if not is_highlight else 'black',
        color=cmap_outlines_discrete[idx] if int(row['year']) < int(highlight_year) else 'black',
        linewidth=1.5 if is_highlight else 1,
        label=row['year'] 
    )

    label_point = row.geometry.representative_point()
    # ax.text(label_point.x, label_point.y, row['year'], fontsize=9, ha='center', va='center')
ax.legend()
# gdf_gepatsch_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# # ## save to tiff
# filename = os.path.join(data_dir,'Output/11_Gepatschferner/Gepatschferner_dhdt_2006-2018_5m_clipped.tif')
# # if not os.path.exists(os.path.dirname(filename)):
# da_gepatsch_dhdt_5m.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Gepatschferner_dhdt_2006-2018_5m_clipped.pdf'), bbox_inches='tight')

#%% GEpatsch THIKNESS


fig,ax=plt.subplots(figsize=(15,10))

da_gepatsch_H.plot.imshow(ax=ax, cmap='Blues', vmin=0, vmax=200, add_colorbar=False)
da_gepatsch_H.plot.imshow(ax=ax, cmap='Blues', vmin=0, vmax=200, 
                               cbar_kwargs={'fraction':0.02, 'label':'Thickness (m)'})

highlight_year = '2006'
cmap_outlines = cm.glasgow_r
## discretize the colormap to have a different color for each year
y_range = 2023-1850
cmap_outlines_discrete = cmap_outlines(np.linspace(0, 1, 20)[::3]) # take only the last 6 colors for the 6 years
for idx, row in gdf_gepatsch.iterrows():
    row_gdf = gpd.GeoDataFrame([row], crs=gdf_gepatsch.crs)
    is_highlight = row['year'] == highlight_year
    row_gdf.boundary.plot(
        ax=ax,
        linestyle='-' if is_highlight else '--',
        color=cmap_outlines_discrete[idx] if not is_highlight else 'black',
        # color=cmap_outlines_discrete[idx] if int(row['year']) < int(highlight_year) else 'black',
        linewidth=1.5 if is_highlight else 1,
        label=row['year'] 
    )

    label_point = row.geometry.representative_point()
    # ax.text(label_point.x, label_point.y, row['year'], fontsize=9, ha='center', va='center')
ax.legend()
# gdf_gepatsch_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# # ## save to tiff
# filename = os.path.join(data_dir,'Output/11_Gepatschferner/Gepatschferner_H_2006-2018_5m_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_gepatsch_H.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Gepatschferner_H_2006-2018_5m_clipped.pdf'), bbox_inches='tight')

#%% velocity from Millan

da_millan_v = xr.open_dataset(
    os.path.join('/Users/mizeboud/Documents/Data_iCloud/',
                 'SMB2D/GlobalGlacierVelocity_Millan2022/RGI-11/',
                 'V_RGI-11_2021July01.tif'), engine='rasterio'
            ).isel(band=0).drop_vars('band').rename({'band_data':'velocity'})['velocity']
print('CRS of Millan velocity file:', da_millan_v.rio.crs)
# da_millan_v.plot.imshow(cmap='viridis', vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})


da_gepatsch_v = reproject_match_grid(da_gepatsch_H, da_millan_v, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
## clip to 1997 outline
da_gepatsch_v = da_gepatsch_v.rio.clip(gdf_gepatsch.loc[gdf_gepatsch['year']=='1997'].geometry, gdf_gepatsch.crs, drop=False)   
# da_gepatsch_v.plot.imshow(cmap='viridis', vmin=0, vmax=100, cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})

#%% PLOT VELOCITY GEPTATSCH

fig,ax=plt.subplots(figsize=(15,10))

da_gepatsch_v.plot.imshow(ax=ax, 
                          cmap='turbo', 
                vmin=0, vmax=200,  # 100
               cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})

highlight_year = None#'2017'
cmap_outlines = cm.glasgow_r
## discretize the colormap to have a different color for each year
y_range = 2023-1850
cmap_outlines_discrete = cmap_outlines(np.linspace(0, 1, 20)[::3]) # take only the last 6 colors for the 6 years
for idx, row in gdf_gepatsch.iterrows():
    row_gdf = gpd.GeoDataFrame([row], crs=gdf_gepatsch.crs)
    is_highlight = row['year'] == highlight_year
    row_gdf.boundary.plot(
        ax=ax,
        linestyle='-' if is_highlight else '--',
        color=cmap_outlines_discrete[idx] if not is_highlight else 'black',
        # color=cmap_outlines_discrete[idx] if int(row['year']) < int(highlight_year) else 'black',
        linewidth=1.5 if is_highlight else 1,
        label=row['year'] 
    )

    label_point = row.geometry.representative_point()
    # ax.text(label_point.x, label_point.y, row['year'], fontsize=9, ha='center', va='center')
ax.legend()
# gdf_gepatsch_union.boundary.plot(ax=ax, color='black', linewidth=2)
## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    # fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# # # ## save to tiff
# filename = os.path.join(data_dir,'Output/11_Gepatschferner/Gepatschferner_velo-millan_5m_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_gepatsch_v.rio.to_raster(filename)
# ## save as img (pdf) for poster
fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Gepatschferner_velo-millan_5m_clipped.pdf'), bbox_inches='tight')

#%% ALETSCH

'''# ########################################
# ALETSCH
Outlines EPSG 
- thickness file only lines; multidate
- 1x thickness file GRD, 2017; 10m res
- 2x DEM 2017 and 2023; 
- 1x dhdt for 17-23: EPSG:2056; 10m res
- velo: median from 2011-2019 : EPSG : 4326; m/day so convert to m/yr
- outlines: RGI
- bedrock: none
## most of CSV data in 'old LV95' crs, which I think is 21781
# ds_aletsch.rio.write_crs('EPSG:21781', inplace=True)

#########################################
'''
import geopandas as gpd
swiss_2056_crs = 'EPSG:2056' # Swiss national grid, in meters; used for DEM and dhdt

# load RGI outlines
glacier_rgiid = 'RGI60-11.01450' # Aletsch ['RGI60-11.01450']
rgi_swiss_file = os.path.join('/Users/mizeboud/Documents/Data_iCloud/',
                 'SMB2D/', 'RGI/11_rgi60_Swiss/11_rgi60_Swiss_simplified.shp')
gl_outline_swiss = gpd.read_file(rgi_swiss_file)

''' ALL SWISS glaciers; use RGI outliens; only larger than >2km '''
gdf_swiss_rgi = gl_outline_swiss.loc[gl_outline_swiss['Area']>2] # 117 glaciers 
# print(len(gdf_swiss_rgi))

gdf_aletsch = gdf_swiss_rgi.loc[gdf_swiss_rgi['RGIId']==glacier_rgiid].copy()
gdf_aletsch.to_crs(swiss_2056_crs, inplace=True)

path2aletsch = '11_Aletsch/'

da_aletsch_velo = xr.open_dataset(os.path.join(data_dir, '11_Aletsch/',
                'aletsch_v_2011-2019/swissALTI3D-orthorectification/median/',
                 'vel_abs-median.tiff'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'velocity'})['velocity']
# reporject to swiss 2056 
da_aletsch_velo = da_aletsch_velo.rio.reproject(swiss_2056_crs, resampling=rio.enums.Resampling.bilinear, nodata=np.nan)

da_aletsch_dhdt = xr.open_dataset(os.path.join(data_dir, '11_Aletsch/',
                'aletsch_dhdt_20170901-20230823.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'dhdt'})['dhdt']   

da_aletsch_velo = reproject_match_grid(da_aletsch_dhdt, da_aletsch_velo, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
## convert to m/yr
da_aletsch_velo = da_aletsch_velo * 365.25

## clip to outline
da_aletsch_velo = da_aletsch_velo.rio.clip(gdf_aletsch.geometry, gdf_aletsch.crs, drop=True)
da_aletsch_dhdt = da_aletsch_dhdt.rio.clip(gdf_aletsch.geometry, gdf_aletsch.crs, drop=True)

#%%
## downsample to 50m for plot pdf
## create dummy grid with 50m resolution and same extent as da_aletsch_dhdt
xvals = np.arange(da_aletsch_dhdt.rio.bounds()[0], da_aletsch_dhdt.rio.bounds()[2], 50)
yvals = np.arange(da_aletsch_dhdt.rio.bounds()[1], da_aletsch_dhdt.rio.bounds()[3], 50)
da_aletsch_50m_dummy = xr.DataArray(
    np.zeros((len(yvals), len(xvals))), 
    coords=[yvals, xvals], 
    dims=['y', 'x'], 
    name='dummy')
da_aletsch_50m_dummy.rio.write_crs(swiss_2056_crs, inplace=True)

da_aletsch_velo_50m = reproject_match_grid(da_aletsch_50m_dummy, da_aletsch_velo, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
da_aletsch_dhdt_50m = reproject_match_grid(da_aletsch_50m_dummy, da_aletsch_dhdt, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

#%% ALETSCH VELOCITY

fig,ax=plt.subplots(figsize=(15,10))
da_aletsch_velo_50m.plot.imshow(ax=ax, 
                     cmap='turbo', 
                     vmin=0, vmax=200,
                    cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})  
gdf_aletsch.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1)

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') ; ax.add_artist(scalebar)
ax.set_axis_off(); ax.set_title('')

# # # ## save to tiff; 
# filename = os.path.join(data_dir,'Output/09_Aletsch/Aletsch_velo_median-2011-2019_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_aletsch_velo.rio.to_raster(filename)
## save as img (pdf) for poster
fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Aletsch_velo_median-2011-2019_clipped.pdf'), bbox_inches='tight')


#%% ALETSCH dhdt

fig,ax=plt.subplots(figsize=(15,10))
da_aletsch_dhdt_50m.plot.imshow(ax=ax, 
                     cmap='RdBu', 
                     vmin=-8, vmax=8,
                    cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})  

gdf_aletsch.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1)

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') ; 
ax.add_artist(scalebar)
ax.set_axis_off(); 
ax.set_title('')

# # # ## save to tiff; 
# filename = os.path.join(data_dir,'Output/09_Aletsch/Aletsch_dhdt_2017-2023_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_aletsch_dhdt.rio.to_raster(filename)
## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Aletsch_dhdt_2017-2023_clipped.pdf'), bbox_inches='tight')


#%% ALETSCH THICKNESS
gdf_aletsch_thickness = gpd.read_file(os.path.join(data_dir, '11_Aletsch/',
                'aletsch_h_multidate.shp'))
print('CRS of Aletsch thickness shapefile:', gdf_aletsch_thickness.crs)

## open .grid file
file_aletsch_grid = os.path.join(data_dir, '11_Aletsch/',
                'aletsch_h_20170901.tif')
da_aletsch_H = xr.open_dataset(file_aletsch_grid, engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'H'})['H']
print('CRS of Aletsch thickness file:', da_aletsch_H.rio.crs)
## to same grid as dhdt and velo
da_aletsch_H = reproject_match_grid(da_aletsch_dhdt, da_aletsch_H, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)
da_aletsch_H_50m = reproject_match_grid(da_aletsch_50m_dummy, da_aletsch_H, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

from shapely.geometry import LineString 
import pandas as pd

## make gdf of profiles a line instead of points, grouping by prf_id, getting x and y from POINT in geometry
## add column with x annd y coordinates from geometry points
gdf_aletsch_thickness['x'] = gdf_aletsch_thickness.geometry.x
gdf_aletsch_thickness['y'] = gdf_aletsch_thickness.geometry.y
gdf_aletsch_H_profiles = gdf_aletsch_thickness.groupby('prf_id')

fig,ax=plt.subplots(figsize=(15,10))
da_aletsch_H_50m.plot.imshow(ax=ax, 
                     cmap='Blues',
                     vmin=0, vmax=500, 
                    cbar_kwargs={'fraction':0.02, 'label':'Thickness (m)'})
## add profile lines
gdf_aletsch_thickness.plot(ax=ax,column='thk', 
                           color='black', markersize=1,alpha=0.5,
                        #    cmap='Blues', vmin=0, vmax=500, 
                           edgecolor=None, #linewidth=0.5,
                           legend=True, legend_kwds={'label':'Thickness (m)'}  )
gdf_aletsch.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1)

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')

# # # ## save to tiff; is al
# filename = os.path.join(data_dir,'Output/09_Aletsch/Aletsch_H-grid_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_aletsch_H.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Aletsch_H-grid_H-profiles_clipped.pdf'), bbox_inches='tight')
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Aletsch_H-grid_clipped.pdf'), bbox_inches='tight')

#%% ZONGO

'''# ########################################
# ZONGO
- Outlines EPSG : 2006 & 2013
- thickness file: millan 3D
- thickness file: field 2012 Zongo_h-InSitu_20120809.csv
- DEM : 2x, 2006 and 2013
- 1x dhdt for 2006-2013: EPSG:32719; 25m res
- velo: 2017-18, 50m res -- also van millan
- outlines: 
- bedrock: 

#########################################
'''
import pandas as pd 

gdf_2006 = gpd.read_file(os.path.join(data_dir,'16_Zongo/Zongo_Outline-2006.shp'))
gdf_2013 = gpd.read_file(os.path.join(data_dir,'16_Zongo/Zongo_Outline-2013.shp'))
print('CRS of Zongo outline files:', gdf_2006.crs, gdf_2013.crs)
assert gdf_2006.crs == gdf_2013.crs, f"CRS should match between 2006 and 2013 outlines; are {gdf_2006.crs} and {gdf_2013.crs}"
gdf_zongo_outline = gpd.GeoDataFrame( pd.concat([gdf_2006, gdf_2013], ignore_index=True), crs=gdf_2006.crs)

da_zongo_dhdt = xr.open_dataset(os.path.join(data_dir,'16_Zongo/Zongo_dhdt_2006-13.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'dhdt'})['dhdt']   
print('CRS of Zongo dhdt file:', da_zongo_dhdt.rio.crs, da_zongo_dhdt.rio.resolution())
da_zongo_dem2006 = xr.open_dataset(os.path.join(data_dir,'16_Zongo/Zongo_DEM_2006.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'dem'})['dem']   
da_zongo_dem2013 = xr.open_dataset(os.path.join(data_dir,'16_Zongo/Zongo_DEM_2013.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'dem'})['dem']   



da_zongo_vx = xr.open_dataset(os.path.join(data_dir, '16_Zongo/Zongo_velx_2017-18.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'vx'})['vx']   
print('CRS of Zongo vx file:', da_zongo_vx.rio.crs, da_zongo_vx.rio.resolution())
da_zongo_v = xr.open_dataset(os.path.join(data_dir, '16_Zongo/Zongo_vel_2017-18.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'v'})['v']   
da_zongo_vy = xr.open_dataset(os.path.join(data_dir, '16_Zongo/Zongo_vely_2017-18.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'vy'})['vy']   
print('CRS of Zongo vy file:', da_zongo_vy.rio.crs, da_zongo_vy.rio.resolution())

da_zongo_H_millan = xr.open_dataset(os.path.join(data_dir, '16_Zongo/Zongo_h_Millan.tif'), engine='rasterio'
                ).isel(band=0).drop_vars('band').rename({'band_data':'H'})['H']   
print('CRS of Zongo thickness file:', da_zongo_H_millan.rio.crs, da_zongo_H_millan.rio.resolution() )
df_zongo_H = pd.read_csv(os.path.join(data_dir, '16_Zongo/Zongo_h-InSitu_20120809.csv'),
                          delimiter=';')
gdf_zongo_H = gpd.GeoDataFrame(df_zongo_H, geometry=gpd.points_from_xy(df_zongo_H['longitude'], df_zongo_H['latitude']), crs='EPSG:4326')
gdf_zongo_H.to_crs(da_zongo_dhdt.rio.crs, inplace=True)

## verything to 50m resolution
# da_zongo_dhdt = reproject_match_grid(da_zongo_vx, da_zongo_dhdt, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan)

## dhdt zongo: to m/yr
dt_zongo = (2013-2006) # years between the two DEMs
da_zongo_dhdt = da_zongo_dhdt / dt_zongo # dhdt from total to m/yr


## clip to 2006 outline
da_zongo_dhdt = da_zongo_dhdt.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=True)   
da_zongo_vx = da_zongo_vx.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=True)   
da_zongo_vy = da_zongo_vy.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=True)   
da_zongo_H_millan = da_zongo_H_millan.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=True)   
da_zongo_v = da_zongo_v.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=True)    
#%% THICKNESS
plt.rcParams.update({'font.size': 12})

fig,ax=plt.subplots(figsize=(15,10))
da_zongo_H_millan.plot.imshow(ax=ax, 
                     cmap='Blues',
                     vmin=0, vmax=200, 
                    cbar_kwargs={'fraction':0.02, 'label':'Thickness (m)'})
## add profile lines
gdf_zongo_H.plot(ax=ax,column='thickness', 
                        markersize=80,
                        #    color='black', markersize=1,alpha=0.5,
                           cmap='Blues', vmin=0, vmax=200, 
                           edgecolor='black', linewidth=0.1,
                           legend=False, legend_kwds={'label':'Thickness (m)'}  )
gdf_2006.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1, label='2006')
gdf_2013.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1, label='2013')

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')
ax.legend()

# # # ## save to tiff; is al
# filename = os.path.join(data_dir,'Output/16_Zongo/Zongo_H-millan_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_zongo_H_millan.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Zongo_H_clipped.pdf'), bbox_inches='tight')

#%% ZONGO DHDT

fig,ax=plt.subplots(figsize=(15,10))
da_zongo_v.plot.imshow(ax=ax, 
                    cmap='turbo',
                     vmin=0, vmax=200, 
                    cbar_kwargs={'fraction':0.02, 'label':'Velocity (m/yr)'})

gdf_2006.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1.2, label='2006')
gdf_2013.boundary.plot(ax=ax,linestyle='--', color='white', linewidth=1.2, label='2013')

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') #
ax.add_artist(scalebar)
ax.set_axis_off()
ax.set_title('')
ax.legend()

# # # ## save to tiff; is al
# filename = os.path.join(data_dir,'Output/16_Zongo/Zongo_v-millan_clipped.tif')
# if not os.path.exists(os.path.dirname(filename)):
#     da_zongo_v.rio.to_raster(filename)
# ## save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Zongo_v-millan_clipped.pdf'), bbox_inches='tight')

#%% Zongo dhdt

# dhdt_dem = da_zongo_dem2013 - da_zongo_dem2006
# dhdt_dem.plot.imshow(cmap='RdBu', vmin=-20, vmax=20, cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})

fig,ax=plt.subplots(figsize=(15,10))
da_zongo_dhdt.plot.imshow(ax=ax, 
                     cmap='RdBu', 
                     vmin=-5, vmax=5,
                    cbar_kwargs={'fraction':0.02, 'label':'Elevation change (m/yr)'})  

gdf_2013.boundary.plot(ax=ax,linestyle='--', color='black', linewidth=1.2, label='2013')
gdf_2006.boundary.plot(ax=ax,linestyle='-', color='black', linewidth=1.2, label='2006')

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',scale_loc='top',box_alpha=0.5,# fontsize=14
                    )
ax.set_aspect('equal') ; 
ax.add_artist(scalebar)
ax.set_axis_off(); 
ax.set_title('');
ax.legend()

# ## save to tiff; 
# filename = os.path.join(data_dir,'Output/16_Zongo/Zongo_dhdt_2006-2013_25m_clipped.tif')
# # if not os.path.exists(os.path.dirname(filename)):
# da_zongo_dhdt.rio.to_raster(filename)
# # save as img (pdf) for poster
# fig.savefig(os.path.join(homedir,'26-EGUposter/figures/Zongo_dhdt_2006-2013_25m_clipped.pdf'), bbox_inches='tight')

# %%
