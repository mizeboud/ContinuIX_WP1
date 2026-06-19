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

path2data_raw = '../../ContinuIX_WP1_data/Data_Package/01_submitted_data/Gepatschferner/'
path2data_clean = '../../ContinuIX_WP1_data/Data_Package/02_raw-cleaned_data/Gepatschferner/'
path2data_homog = '../../ContinuIX_WP1_data/Data_Package/03_homogenized_data/Gepatschferner/'

#%% Functions

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

## dhdt: in meter --> convert to m/yr

## outlines: geojson to shapefile

## DEM: downloaded from data.tirol.gv.at; is in EPSG:31254 so should reproject to 25832
##################################
'''


gdf_2006 = gpd.read_file(os.path.join(path2data_raw, 'gepatsch_outline_2006.geojson'))
gdf_2017 = gpd.read_file(os.path.join(path2data_raw, 'gepatsch_outline_2017.geojson'))

assert gdf_2006.crs == gdf_2017.crs == 'EPSG:25832', "CRS of outlines do not match"

## save to shapefile in cleaned dir

fname = f'gepatsch_outline_2006.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    gdf_2006.to_file(os.path.join(path2data_clean, fname), driver='ESRI Shapefile')
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")

fname = f'gepatsch_outline_2017.shp'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    gdf_2017.to_file(os.path.join(path2data_clean, fname), driver='ESRI Shapefile')
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")



#%%
''' ########################
### dhdt
############################ '''
da_bedrock = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_bedrock_2006.tif')
                               ).isel(band=0).drop_vars('band') ## 10 m


dh_1 = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_dh_2006-2017.tif')
                             ).isel(band=0).drop_vars('band')

dh_2 = xr.open_dataarray(os.path.join(path2data_raw, 'gepatsch_dh_2006-2018.tif')
                             ).isel(band=0).drop_vars('band')

dh_1.rio.resolution() ## 1 m

## reproject to bedrock grid resolution (10 m)
da_dh_1 = reproject_match_grid(da_bedrock, dh_1, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_dh_2 = reproject_match_grid(da_bedrock, dh_2, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
da_dh = da_dh_1.combine_first(da_dh_2)

dt = 2017-2006
da_dhdt = da_dh / dt ## in m/yr

## clip to outline 
da_dhdt = da_dhdt.rio.clip(gdf_2006.geometry, gdf_2006.crs, drop=False) # drop=False to keep the same grid and not drop the pixels outside the outline (which will be set to nodata)

## store to cleaned data folder
fname = f'gepatsch_dhdt_2006-2017.tif'
if not os.path.exists(os.path.join(path2data_clean, fname)):
    print(f'Saving {fname} to cleaned data directory...')
    da_dhdt.rio.to_raster(os.path.join(path2data_clean, fname))
else:
    print(f"File {fname} already exists in cleaned data directory. Skipping save.")


# %%
''' ########################
### DEMs: from data.tirol.gv.at; is in 50cm resolution
1. combine DEMs at lower resolution (2m), in EPSG25832, save in 'submitted' folder
2. no further cleaning needed
3. reproject to 10m resolution and save in 'homogenized' folder
############################ '''
import glob 

file_to_dem = '../../ContinuIX_WP1_data/11_Gepatschferner/tirol-gov_DEM-DTM/DOM_DEM/'
filelist_dem17 = glob.glob(os.path.join(file_to_dem, '*.tif'))
# ds_DEM_2017 = xr.mf

## create dummy grid
da_dummy_2m = create_regular_dummy_grid(da_dhdt, grid_res=2, crs=25832, unit='m')
da_dem_list = []
for file in filelist_dem17:
    da_tile = xr.open_mfdataset(file).isel(band=0).drop_vars('band')['band_data']

    da_tile_2m = reproject_match_grid(da_dummy_2m, da_tile, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
    da_dem_list.append(da_tile_2m)

## combine DEMs at 2m resolution
da_dem_2017_2m = xr.combine_nested(da_dem_list, concat_dim='band').max(dim='band') ## patches all tiles and taking max means taking the single non-nan value.
# da_dem_2017_2m.plot.imshow()

assert da_dem_2017_2m.rio.crs == da_dhdt.rio.crs, "CRS of DEM and dummy grid do not match"
## save to submitted data folder
fname = 'gepatsch_DEM_2017.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to submitted data directory...')
    da_dem_2017_2m.rio.to_raster(os.path.join(path2data_raw, fname))
else:
    print(f"File {fname} already exists in submitted data directory. Skipping save.")

## do the same for tirol.gov BED (DTM)
file_to_dtm = '../../ContinuIX_WP1_data/11_Gepatschferner/tirol-gov_DEM-DTM/DGM_DTM/'
filelist_dtm17 = glob.glob(os.path.join(file_to_dtm, '*.tif'))
da_dtm_list = []
for file in filelist_dtm17:
    da_tile = xr.open_mfdataset(file).isel(band=0).drop_vars('band')['band_data']

    da_tile_2m = reproject_match_grid(da_dummy_2m, da_tile, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)
    da_dtm_list.append(da_tile_2m)

da_dtm_2017_2m = xr.combine_nested(da_dtm_list, concat_dim='band').max(dim='band') ## patches all tiles and taking max means taking the single non-nan value.
da_dtm_2017_2m.plot.imshow()

fname = 'gepatsch_DTM_2017.tif'
if not os.path.exists(os.path.join(path2data_raw, fname)):
    print(f'Saving {fname} to submitted data directory...')
    da_dtm_2017_2m.rio.to_raster(os.path.join(path2data_raw, fname),driver='COG')
else:
    print(f"File {fname} already exists in submitted data directory. Skipping save.")

#%%
''' # 3. to homogenized: 10 m '''
da_dummy_10m = create_regular_dummy_grid(da_dhdt, grid_res=10, crs=25832, unit='m')
da_dem_2017_10m = reproject_match_grid(da_dummy_10m, da_dem_2017_2m, resample_method=rio.enums.Resampling.bilinear, nodata_value=np.nan)

## save to submitted data folder
fname = 'gepatsch_DEM_2017.tif'
if not os.path.exists(os.path.join(path2data_homog, fname)):
    print(f'Saving {fname} to submitted data directory...')
    da_dem_2017_10m.rio.to_raster(os.path.join(path2data_homog, fname))
else:
    print(f"File {fname} already exists in submitted data directory. Skipping save.")

# %% Looking at stake veloicity
''' ########################
### STAKE VELOCITY

# from readme:
- each row represents coordinates between consequetive readings (date0 to date1; or date0 to datemid to date1).
- In rows where coordinates were filled only for date0 and date1, the distance between those coordinates can be used to derive an annual velocity. 
- If coordinates for "datemid" were filled, the annual velocity should be computed by combining the first period with the second 
    (for example, distance moved in X direction: (X0-X0_m)+(X1_m-X1)).


--> so e.g. stake P60:
    P60 row 1  : t0 --> t1
    P60 row 2  : t2 --> t3
    --> also get cross-row displacement/velocity? so t1-->t2? or not, since that's not instructed from the readme.
--> also: x0-x0_m seems weird, why not do x0_m-x0, since we want to get the velocity in the same direction as the movement (from t0 to t1)? 

--> for velo from distance: need to do same mid-step as distance, or just overall? should be same right
############################ '''

df_coords = pd.read_csv(os.path.join(path2data_raw, 'gepatsch_coordinates_2009_2024.csv'))
df_coords
df_crs = 'EPSG:25832' ## from readme

## convert date0, datemid, date1 to datetime
## rename columns
df_coords.rename(columns={'date0':'date0_str', 'datemid':'datemid_str', 'date1':'date1_str'}, inplace=True)
df_coords['date0'] = pd.to_datetime(df_coords['date0_str'], format='%d.%m.%Y', errors='coerce')
# df_coords['datemid'] = pd.to_datetime(df_coords['datemid_str'], format='%d.%m.%Y', errors='coerce')
df_coords['datemid'] = pd.to_datetime(df_coords['datemid_str'], format='%d.%m.%Y %H:%M')
df_coords['date1'] = pd.to_datetime(df_coords['date1_str'], format='%d.%m.%Y', errors='coerce')
df_coords['duration_days'] = (df_coords['date1'] - df_coords['date0']).dt.days
df_coords['dt_yr'] = df_coords['duration_days'] / 365.25 ## in years
df_coords

## PLOT SETTINGS
scale_factor = 5
norm_v = plt.Normalize(vmin=0, vmax=60)
cmap_v = plt.cm.plasma
cmap_v = plt.cm.cividis

colors_stakes = plt.cm.tab20#[0:df_coords['stake'].nunique()] # tab20 has 20 colors, so if more than 20 stakes, will need to use a different colormap or cycle through colors
norm_stakes = plt.Normalize(vmin=0, vmax=20)
norm_yr = plt.Normalize(vmin=df_coords['date0'].min().year, vmax=df_coords['date0'].max().year)


fig, ax = plt.subplots(figsize=(8, 8))
gdf_2006.plot(ax=ax, facecolor='steelblue', edgecolor='steelblue', # lightblue
                    linewidth=1.5, alpha=0.5, label='Glacier outline 2006')
gdf_2006.boundary.plot(ax=ax, edgecolor='steelblue', linewidth=1.5, alpha=0.5, label='Glacier outline 2006')

stake_i = 0
# # for df_stake in df_coords.groupby('stake'):
# #     print(df_stake)
    
#     stake_id = df_stake[0]
#     df_stake = df_stake[1]


# ## DEV: check one unique stake. Choose P66.
stake_ids = ['P60 (P51)', 'P61', 
             'P62 (P52)', 'P63', 'P64', 
             'P65 (P53)', 'P66', 'P67', 
             'P68', 'P69', 'P70', 'P71',
             'P72', 'P73', 'P74', 'P75']

for stake_id in stake_ids:
    # df_stake = df_coords[df_coords['stake'] == 'P66']
    df_stake = df_coords[df_coords['stake'] == stake_id]
    df_stake

    print(f"Stake {stake_id}:")
    # print(df_stake)
    # print("\n")


    '''## (1) calculate as specified in readme '''
    # ## (A) calculate distance between date0 and date1
    # df_stake['dx_01'] = df_stake['X0'] - df_stake['X1']
    # df_stake['dy_01'] = df_stake['Y0'] - df_stake['Y1']

    # ## (B) calculate distance between date0 and datemid, and datemid and date1
    # df_stake['dx_0m1'] = (df_stake['X0'] - df_stake['X0_m']) + (df_stake['X1'] - df_stake['X1_m'])
    # df_stake['dy_0m1'] = (df_stake['Y0'] - df_stake['Y0_m']) + (df_stake['Y1'] - df_stake['Y1_m'])

    '''## (2) calculate as I think it should'''
    ## (A) calculate distance between date0 and date1
    df_stake['dx_01'] = df_stake['X1'] - df_stake['X0']
    df_stake['dy_01'] = df_stake['Y1'] - df_stake['Y0']

    ## (B) calculate distance between date0 and datemid, and datemid and date1
    df_stake['dx_0m1'] = (df_stake['X0_m'] - df_stake['X0']) + (df_stake['X1'] - df_stake['X1_m'])
    df_stake['dy_0m1'] = (df_stake['Y0_m'] - df_stake['Y0']) + (df_stake['Y1'] - df_stake['Y1_m'])



    ''' ## ---'''

    ## select which distance to use: if datemid is not null, use dx_0m1 and dy_0m1, otherwise use dx_01 and dy_01
    df_stake['dx'] = np.where(df_stake['datemid'].notnull(), df_stake['dx_0m1'], df_stake['dx_01'])
    df_stake['dy'] = np.where(df_stake['datemid'].notnull(), df_stake['dy_0m1'], df_stake['dy_01'])
    df_stake

    ## convert to velocity 
    # df_stake['distance_m'] = np.sqrt(df_stake['dx']**2 + df_stake['dy']**2)
    # df_stake['velocity_myr'] = df_stake['distance_m'] / df_stake['dt_yr']

    df_stake['vx_myr'] = df_stake['dx'] / df_stake['dt_yr']
    df_stake['vy_myr'] = df_stake['dy'] / df_stake['dt_yr']
    df_stake['velocity_myr'] = np.sqrt(df_stake['vx_myr']**2 + df_stake['vy_myr']**2)

    ## angle of movement in degrees 
    angle_rad = np.arctan2(df_stake['dy'], df_stake['dx'])
    angle_deg = np.degrees(angle_rad) ## from -180 to 180 ; # 0: east, 90: north, 180 or -180: west, -90: south
    angle_deg_360 = (angle_deg + 360) % 360 ## from 0 to 360
    # 0°   = east / +x
    # 90°  = north / +y
    # 180° = west / -x
    # 270° = south / -y
    df_stake['angle'] = angle_deg_360

    ## SUBSELECT columns without datemid
    # df_plot = df_stake.dropna(subset=['datemid'])
    # df_plot = df_stake.loc[~df_stake['datemid'].isna(),:]    ## rows where datemid ISNOT nan

    ### %% PLOT stakes

    # fig, ax = plt.subplots(figsize=(8, 8))
    # gdf_2006.plot(ax=ax, facecolor='steelblue', edgecolor='steelblue', # lightblue
    #                     linewidth=1.5, alpha=0.5, label='Glacier outline 2006')

    ## arrow per stake
    # x = angle['x']
    # y = angle['y']   

    # arrow_length = 150
    # # arrow_length = dmg*8000 # scale lenngth with dmg
    # arrow_length = 250 + dmg*3000
    arrow_length = 100 # 300+dmg*3000
    scale = 500
    width=8
    headwidth=4


    '''## define arrow
    # 'xy': Arrow direction in data coordinates, 
            i.e. the arrows point from (x, y) to (x+u, y+v). 
            This is ideal for vector fields or gradient plots where the arrows 
            should directly represent movements or gradients in the x and y directions.
    # 'uv'; Arrow directions are based on display coordinates; 
            i.e. a 45° angle will always show up as diagonal on the screen, irrespective of figure or Axes aspect ratio or Axes data ranges. 
            This is useful when the arrows represent a quantity whose direction is not tied to the x and y data coordinates.
    '''
    # x = df_stake['X0'].values
    # y = df_stake['Y0'].values
    # # u = df_stake['vx_myr'].values ## using DX or VX_MYR gives the same plot
    # # v = df_stake['vy_myr'].values
    # u = df_stake['dx'].values ## using DX or VX_MYR gives the same plot
    # v = df_stake['dy'].values
    # ax.quiver(x,y, u,v, 
    #             angles='xy',
    #             units='x',
    #             # scale=0., #1.1,#0.8,#0.4,
    #             scale_units='width', 
    #             # scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
    #             width=width, #120,#100, # 0.005 typical starting value
    #             # headwidth=headwidth,#4, # default 3
    #             pivot='tail',
    #             color=colors_stakes(stake_i), # color by stake id
    #             )
    # ax.set_title('Stake velocity using x,y and angles=''xy'' ')
    
    # # angletype = 'uv' -- DONT USE !!
    # # x = df_stake['X0'].values
    # # y = df_stake['Y0'].values
    # # u = arrow_length * np.cos(np.deg2rad(df_stake['angle'].values) )
    # # v = arrow_length * np.sin(np.deg2rad(df_stake['angle'].values) )
    # # ax.quiver(x,y, u,v, 
    # #             angles='uv',
    # #             units='x',
    # #             # scale=scale, #1.1,#0.8,#0.4,
    # #             scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
    # #             width=width, #120,#100, # 0.005 typical starting value
    # #             # headwidth=headwidth,#4, # default 3
    # #             # pivot='tail',
    # #             color=colors_stakes(stake_i), # color by stake id
    # #             )
    # # ax.set_aspect('equal')
    # # fig.tight_layout()
    # # ax.set_title('Stake velocity using x,y and angles=''uv'' ')

    # # Arrows per stake
    for _, row in df_stake.iterrows():
        # if pd.isna(row['X0']) or pd.isna(row['Y0']):
        #     continue
        # dist = np.hypot(row['dx'], row['dy'])
        # if dist == 0:
        #     continue
        color = cmap_v(norm_v(row['velocity_myr']))
        norm = norm_v

        # color = cmap(norm_yr(row['date0'].year))
        # norm = norm_yr

        # ux = row['vx_myr'] * scale_factor
        # uy = row['vy_myr'] * scale_factor
        arrow_scale = 5
        # ux = row['dx'] / row['dx'] * row['velocity_myr'] * arrow_scale
        # uy = row['dy'] / row['dy'] * row['velocity_myr'] * arrow_scale
        ux = row['vx_myr'] * arrow_scale
        uy = row['vy_myr'] * arrow_scale

        # x = row['X0']
        # y = row['Y0']
        # arrow_length = 300 + row['velocity_myr'] * scale
        # u = arrow_length * np.cos(np.deg2rad(row['angle']))
        # v = arrow_length * np.sin(np.deg2rad(row['angle']))

        arrow_length = 1 
        # u = arrow_length * row['vx_myr'] 
        # v = arrow_length * row['vy_myr']

        x = row['X0']
        y = row['Y0']
        # u = row['vx_myr'] ## using DX or VX_MYR gives the same plot
        # v = row['vy_myr']
        u = row['dx'] * arrow_length ## scale the dx component to make the arrow length proportional to the velocity, while keeping the angle the same
        v = row['dy'] * arrow_length ## scale the dy component to make the arrow length proportional to the velocity, while keeping the angle the same
        # u = 
        
        # ax.scatter(row['X0'], row['Y0'],
        #             color=color, s=20, zorder=5,
        #             edgecolors='black', linewidths=0.5)
        ax.annotate('',
            xy=(row['X0'] + ux, row['Y0'] + uy),
            xytext=(row['X0'], row['Y0']),
            arrowprops=dict(arrowstyle='->', color=color, lw=1.0),
            zorder=5)

        # ax.quiver(x,y, u,v,
        #             angles='xy',
        #             units='x',
        #             # scale=0.00001, #1.1,#0.8,#0.4,
        #             scale_units='x', # sclae_unit 'x': unit will be 0.5 xax units -- setting scale=1 means 0.5xaxunit, scale > 1:  means <0.5xaxunit (smaller), scale=0-1: bigger
        #             width=width, #120,#100, # 0.005 typical starting value
        #             # headwidth=headwidth,#4, # default 3
        #             pivot='tail',
        #             color=color,
    #                 )
    # # Colorbar
    # sm = plt.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    # cbar.set_label('Velocity (m yr⁻¹)')

    # '''# Zoom to stake area'''
    # padding = 500
    # ax.set_xlim(df_stake['X0'].min() - padding,
    #             df_stake['X0'].max() + padding)
    # ax.set_ylim(df_stake['Y0'].min() - padding,
    #             df_stake['Y0'].max() + padding)


    ax.set_xlabel('X / Easting (m)')
    ax.set_ylabel('Y / Northing (m)')
    ax.grid(linestyle='--', alpha=0.3)
    # ax.set_title(f'Stake {stake_id}')

    stake_i += 1

# # Colorbar velo
sm = plt.cm.ScalarMappable(cmap=cmap_v, norm=norm_v)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Velocity (m yr⁻¹)')

# ## colorbar stakes
# # # Colorbar
# sm = plt.cm.ScalarMappable(cmap=colors_stakes, norm=norm_stakes)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
# cbar.set_label('Stake IDs'); cbar.set_ticks(np.arange(0.5, len(stake_ids)+0.5)); cbar.set_ticklabels(stake_ids)

# '''# Zoom to stake area: overall '''
padding = 500
ax.set_xlim(df_coords['X0'].min() - padding,
            df_coords['X0'].max() + padding)
ax.set_ylim(df_coords['Y0'].min() - padding,
            df_coords['Y0'].max() + padding)



ax.set_xlabel('X / Easting (m)')
ax.set_ylabel('Y / Northing (m)')
ax.grid(linestyle='--', alpha=0.3)
ax.set_title(f'Stake velocities')

# %%

# %%
