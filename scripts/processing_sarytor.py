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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_raw_data/SaryTor/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_cleaned_data/SaryTor/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_experiment_package/SaryTor/'


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

def create_regular_dummy_grid(ds, grid_res, crs=None, unit='m'):
    ''' Create a regular dummy grid with specified resolution and CRS, based on the extent of the input dataset/dataArray. 
    This can be used as a reference grid for reprojecting/matching other datasets. 
    Parameters:
    - ds: xarray Dataset or DataArray to define the extent of the grid
    - grid_res: desired grid resolution (in the same units as the CRS, e.g., meters)
    - crs: desired coordinate reference system (default: same as input dataset)
    - unit: unit of the grid resolution (default: 'm' for meters)
    '''
    if not crs:
        crs = ds.rio.crs
    x0 = ds.x.min().item() ; x1 = ds.x.max().item() ; y0 = ds.y.min().item() ; y1 = ds.y.max().item()
    # x0 = np.floor(x0/grid_res)*grid_res; x1 = np.floor(x1/grid_res)*grid_res; 
    # y0 = np.floor(y0/grid_res)*grid_res; y1 = np.floor(y1/grid_res)*grid_res
    x_seq = np.arange(x0, x1+grid_res, step=grid_res )
    y_seq = np.arange(y0, y1+grid_res, step=grid_res )

    ## get floating point presicion of grid_res, and apply that to x0 (e.g. if x0 is at 0.0730001 and grid_res is 0.08, then start xgrid at 287.00 instead of 287.0000000001)
    decimal_places = int(-np.floor(np.log10(grid_res)))
    ## round the sequences to avoid floating point precision issues (e.g. if x_seq is [287.00, 287.08, 287.16, ...] but due to floating point precision it is actually [287.0000000001, 287.0800000001, 287.1600000001, ...], then round to 2 decimal places to get rid of the tiny differences)
    x_seq = np.round(x_seq, decimal_places)
    y_seq = np.round(y_seq, decimal_places)

    grid_dummy = xr.DataArray(
        data=np.ones( (len(y_seq), len(x_seq)) ),
        dims=["y", "x"],
        coords=dict(
            y=y_seq,
            x=x_seq,
        ),
        attrs=dict(
            description=f"regular grid at {grid_res} {unit} resolution",
            unit=unit,
        ),
    ).rio.write_crs(crs)
    
    return grid_dummy

#%% Step 0: Check submitted (raw) data
''' ##################################
Check submitted (raw) data where needed


## DEMs
- have DEMs annually, 2021-2025.
- resolutions vary between 0.073, 0.069, 0.075, 2 (2024), 0.081
--> resample all execpt 2024 to 0.08 m; keep 2024 as is for CLEANED dir. 
--> for homogenized dir, resample all to decided resolution (maybe 2m or maybe something else)

## DHDT: only DEMs provided, so need to calculate DHDT from DEMs.
- calculate dhdt for each year, but at 2m resolution (so to have the same resolution for every year)
- calculate average annual dhdt for 2021-2025


## Velocity: no cleaning needed.

## Thickness: no cleaning needed


## Thickness profiles: 

## Bedrock: not provided;
Thickenss is from 2021 so use DEM from 2021 to calculate bedrock.
Calculate as `sarytor_bedrock = sarytor_DEM_2021 - sarytor_h_2021`.
---
TO DO: check grid resolutions and homogenize these

##################################
'''
sarytor_outline = gpd.read_file(os.path.join(path2data_raw, 'sarytor_outline_2021.shp'))


''' ##################################
DEMs
################################## '''
da_dem21 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2021.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem22 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2022.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem23 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2023.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem24 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2024.tif')
                             ).isel(band=0).drop_vars('band')['band_data']
da_dem25 = xr.open_mfdataset(os.path.join(path2data_raw, 'sarytor_DEM_2025.tif')
                             ).isel(band=0).drop_vars('band')['band_data']

assert da_dem21.rio.crs == da_dem22.rio.crs == da_dem23.rio.crs == da_dem24.rio.crs == da_dem25.rio.crs, "CRS of DEMs do not match"
# assert da_dem21.shape == da_dem22.shape == da_dem23.shape == da_dem24.shape, "Shape of DEMs do not match"
# assert da_dem21.rio.resolution() == da_dem22.rio.resolution() == da_dem23.rio.resolution() == da_dem24.rio.resolution(), "Resolution of DEMs do not match"
print('CRS of DEMs:', da_dem21.rio.crs)
print(da_dem21.shape, da_dem22.shape, da_dem23.shape, da_dem24.shape, da_dem25.shape)
print(da_dem21.rio.resolution(), da_dem22.rio.resolution(), da_dem23.rio.resolution(), da_dem24.rio.resolution(), da_dem25.rio.resolution()) ## 7.3 cm

## set up empty but regular grid
# da_dummy = create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution
# da_dem21_08 = reproject_match_grid(da_dummy, da_dem21, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# da_dem22_08 = reproject_match_grid(da_dummy, da_dem22, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# da_dem23_08 = reproject_match_grid(da_dummy, da_dem23, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
# ## skip 2024
# da_dem25_08 = reproject_match_grid(da_dummy, da_dem25, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal

# resampled and save DEMs to cleaned data directory
da_dummy = create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution

for year, da_dem in zip([2021, 2022, 2023, 2025], [da_dem21, da_dem22, da_dem23, da_dem25]):
    fname = f'sarytor_dem_{year}_8cm.tif'
    # da_dummy = create_regular_dummy_grid(da_dem21, grid_res=0.08, crs=da_dem21.rio.crs) ## 8 cm resolution

    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Reprojecting {year}')
        da_dem_08 = reproject_match_grid(da_dummy, da_dem, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## use nearest, since resampling is minimal
        print('CRS of DEMs:', da_dem_08.rio.crs, 'resolution:', da_dem_08.rio.resolution())
        print('Resolution of DEMs:', da_dem_08.rio.resolution())

        print(f'Saving {fname} to cleaned data directory...')
        da_dem_08.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")

fname = f'sarytor_dem_2024_2m.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dem24.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")

#%%
''' ##################################
dhdt
################################## '''
## also resample dem23 and dem25 to 2m for dhdt_ w.r.t 2024
da_dem21_2m = reproject_match_grid(da_dem24, da_dem21, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem22_2m = reproject_match_grid(da_dem24, da_dem22, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem23_2m = reproject_match_grid(da_dem24, da_dem23, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal
da_dem25_2m = reproject_match_grid(da_dem24, da_dem25, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## use nearest, since resampling is minimal

#%%
## plot only the 2m versions, plotting 8cm version takes too long
# da_dem23_2m.plot.imshow()
# sarytor_outline.boundary.plot(ax=plt.gca(), color='red')
da_dem24.plot.imshow()
sarytor_outline.boundary.plot(ax=plt.gca(), color='red')

#%%
da_dem_all = xr.concat([da_dem21_2m, da_dem22_2m, da_dem23_2m, da_dem24, da_dem25_2m], dim='time')
da_dem_all['time'] = [2021, 2022, 2023, 2024, 2025]

## Count px where each year has valid data
# value 5 has valid data all years; use this as mask for dhdt 
mask_valid = xr.where(~np.isnan(da_dem_all), 1, 0).sum(dim='time') 
da_dhdt_2m = da_dem_all.diff(dim='time') ## this will give the dhdt for each year, at 2m resolution
da_dhdt_2m = da_dhdt_2m.where(mask_valid>4) ## keep only px where all 5 years have valid data (so where the sum of the mask is 4, since diff reduces the number of time steps by 1)
# da_dhdt_2m['time'] = '2021-2022', '2022-2023', '2023-2024', '2024-2025' ## assign time values to the dhdt dataarray

# ## calculate average annual dhdt for 2021-2025
da_dhdt_2125_2m = da_dhdt_2m.mean(dim='time') ## this will give the average annual dhdt for 2021-2025, at 2m resolution

da_dhdt_2125_2m.plot.imshow(vmin=-5, vmax=5, cmap='RdBu_r')
sarytor_outline.boundary.plot(ax=plt.gca(), color='red')

#%%
for da_dhdt, year in zip(da_dhdt_2m, ['2021-2022', '2022-2023', '2023-2024', '2024-2025']):
    fname = f'sarytor_dhdt_{year}_2m.tif'

    if not os.path.exists(os.path.join(path2data_clean, fname)):
        print(f'Saving {fname} to cleaned data directory...')
        da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
    else:
        print(f"File {fname} already exists in cleaned data directory. Skipping save.")

fname = f'sarytor_dhdt_2021-2025_2m.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dhdt_2125_2m.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")
    

#%%
''' ##################################
Bedrock
################################## '''

da_dem = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_DEM_2021.tif')
                             ).isel(band=0).drop_vars('band')
da_h = xr.open_dataarray(os.path.join(path2data_raw, 'sarytor_h_2021.tif')
                           ).isel(band=0).drop_vars('band')

assert da_dem.rio.crs == da_h.rio.crs, "CRS of DEM and thickness grid do not match"
# assert da_dem.rio.resolution() == da_h.rio.resolution(), "Resolution of DEM and thickness grid do not match"
# assert da_dem.shape == da_h.shape, "Shape of DEM and thickness grid do not match"
print(f"CRS of DEM and thickness grid match: {da_dem.rio.crs}")

print(da_dem.rio.resolution(), da_h.rio.resolution()) ## thickness has lower resolution, so upsample this one

# ## thickness and DEM have different resolutions. DEM has 7 cm resolution.. so for first step upsample thickness but eventually we will want to downsample this.
## mm no maybe for DEM it's not that relevent that the resolution is 7 cm.. so go with downsampling after all.

# da_h_match = reproject_match_grid(da_dem, da_h, resample_method=rio.enums.Resampling.nearest, nodata_value=np.nan) ## from low to high res: nearest
da_dem_match = reproject_match_grid(da_h, da_dem, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan) ## from high to low res: bilinear

assert da_h.rio.resolution() == da_dem_match.rio.resolution(), "Resolution of DEM and thickness grid do not match"
assert da_dem_match.shape == da_h.shape, "Shape of DEM and thickness grid do not match"

## calculate difference, fill areas outside of glacier (where H is nan or 0) with DEM values (assuming bedrock = DEM there)
da_bedrock = da_dem_match - da_h
da_bedrock_filled = xr.where(np.isnan(da_bedrock), 
                               da_dem_match,  ## whre condition is ture
                               da_bedrock) ## where condition is false
da_bedrock_filled.plot.imshow()
sarytor_outline.boundary.plot(ax=plt.gca(), color='red')
## save to CLEAN directory
fname_bedr = 'sarytor_bedrock.tif'
if not os.path.exists(os.path.join(path2data_clean, fname_bedr)):
    da_bedrock_filled.rio.to_raster(os.path.join(path2data_clean, fname_bedr))
else:
    print(f"File {fname_bedr} already exists in cleaned data directory. Skipping save.")
    

