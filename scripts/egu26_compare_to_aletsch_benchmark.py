#%% validate my SMB fields to ALETSCH and MORTERATSCH benchmark data
# using ContinuIX aletsch benchmark data;
# provided in .dat files for every year, 
# has been extrapolated from point data so should be similar to glamos elevation-bins but then spatial

# M. Izeboud, April/May 2026

import os
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 
import geopandas as gpd
import rasterio as rio
import pandas as pd

target_crs = 'EPSG:32632' ## EPSG of Millan2022 (50 m resolution) --> where I have all my input/bruteForceOutput in
swiss_crs = 'EPSG:21781' # 'EPSG:2056' ## CH1903 / LV95 ## data of GLAMOS stakes
swiss_crs_morteratsch = 'EPSG:2056' ##  CH1903+ / LV95 
data_dir = '/Users/mizeboud/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data_iCloud/SMB2D/'
homedir = '/Users/mizeboud/Documents/Documents_mizeboud/PostDoc/2D-SMB/'

my_palette = ['#2b6f39','#efbb1a','#d490c6'] #  update the brown/yellow of cubeH hex: '#a1794a' to ....#efbb1a

save_fig = False 
path2save = os.path.join('/Users/mizeboud/Documents/Documents_mizeboud/', 
                         'Projects/ContinuIX/26-EGUposter/figures')


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

#%% load ALETSCH outlines

''' ------------------
### ALETSCH
---------------------- '''
glacier_rgiid = 'RGI60-11.01450' # Aletsch ['RGI60-11.01450']
rgi_swiss_file = os.path.join(data_dir, 'RGI/11_rgi60_Swiss/11_rgi60_Swiss_simplified.shp')
gl_outline_swiss = gpd.read_file(rgi_swiss_file)

''' ALL SWISS glaciers; use RGI outliens; only larger than >2km '''
gdf_aletsch = gl_outline_swiss.loc[gl_outline_swiss['RGIId']==glacier_rgiid].copy()
gdf_aletsch.to_crs(swiss_crs, inplace=True)

'''#%% load aletsch data'''
path2aletsch = '/Users/mizeboud/Library/CloudStorage/OneDrive-VrijeUniversiteitBrussel/ContinuIX/ContinuIX_WP1_data/11_Aletsch/aletsch_smb_extrapolated_2000-2025/'

files_smb_aletsch = sorted([f for f in os.listdir(path2aletsch) if f.endswith('.grid')])
files_smb_aletsch = [os.path.join(path2aletsch, f) for f in files_smb_aletsch]
times_aletsch = [int(os.path.basename(f).split('_')[0]) for f in files_smb_aletsch]

da_1 = xr.open_dataset(os.path.join(path2aletsch, files_smb_aletsch[0]), engine='rasterio')
ds_aletsch = xr.open_mfdataset(files_smb_aletsch, engine='rasterio', 
                               combine='nested', concat_dim='time'
                               ).isel(band=0).drop_vars('band').rename({'band_data':'SMB'})
# update time coords
ds_aletsch['time'] = times_aletsch

ds_aletsch.rio.crs # not given, need to set. README states "old LV95" grid which I think is 21781
ds_aletsch.rio.write_crs('EPSG:21781', inplace=True)

da_aletsch_avg = ds_aletsch.sel(time=slice(2000,2020))['SMB'].mean(dim='time') # average over 2000-2020 to compare to my 2000-2020 
# unit: unit: m w.e.


''' ----------
## Load my SMB -- m.w.e or mie/yr ? 
-------------- '''
path2smb = '/Users/mizeboud/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data_iCloud/SMB2D/SMB2D/bruteForce/bestParams_f001_F080_N9/'

da_smb0020 = xr.open_dataarray( os.path.join(path2smb, 'smb_2000-2020/', f'{glacier_rgiid}_smb-kgyr_f001_F080_N9.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_smb1020 = xr.open_dataarray( os.path.join(path2smb, 'smb_2010-2020/', f'{glacier_rgiid}_smb-kgyr_f001_F080_N9.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_smb1520 = xr.open_dataarray( os.path.join(path2smb, 'smb_2015-2020/', f'{glacier_rgiid}_smb-kgyr_f001_F080_N9.tif') ).isel(band=0).drop_vars('band').rename('SMB')


da_smb0020 = da_smb0020.rio.reproject(swiss_crs, inplace=True)
da_smb1020 = da_smb1020.rio.reproject(swiss_crs, inplace=True)
da_smb1520 = da_smb1520.rio.reproject(swiss_crs, inplace=True)


# % Prep for plot

''' ----------
## Aletsch average and spatial diff 
-------------- '''
## get aletsch average
## clip da_aletsch_avg to same bounds
myextent = da_smb0020.rio.bounds()
da_aletsch_avg = da_aletsch_avg.rio.clip_box(minx=myextent[0], miny=myextent[1], maxx=myextent[2], maxy=myextent[3])
## match da_aletsch_avg grid to my SMB fields
da_aletsch_avg_50m = reproject_match_grid(da_smb0020, da_aletsch_avg, resample_method=rio.enums.Resampling.bilinear)

## caclulcate pixel diff
diff_0020 = da_smb0020 - da_aletsch_avg_50m
diff_1020 = da_smb1020 - da_aletsch_avg_50m
diff_1520 = da_smb1520 - da_aletsch_avg_50m


### calculate MAE and RMSE
mae_0020 = np.nanmean( np.abs( diff_0020.values.flatten() ) )
rmse_0020 = np.sqrt( np.nanmean( diff_0020.values.flatten()**2 ) )
mae_1020 = np.nanmean( np.abs( diff_1020.values.flatten() ) )
rmse_1020 = np.sqrt( np.nanmean( diff_1020.values.flatten()**2 ) )
mae_1520 = np.nanmean( np.abs( diff_1520.values.flatten() ) )
rmse_1520 = np.sqrt( np.nanmean( diff_1520.values.flatten()**2 ) )
print(f'MAE 2000-2020: {mae_0020:.2f}, RMSE 2000-2020: {rmse_0020:.2f}')
print(f'MAE 2010-2020: {mae_1020:.2f}, RMSE 2010-2020: {rmse_1020:.2f}')
print(f'MAE 2015-2020: {mae_1520:.2f}, RMSE 2015-2020: {rmse_1520:.2f}')



''' ----------
## pandas dataframe for Regression plot 
-------------- '''

df_pxs = pd.DataFrame()
df_pxs_uav = da_aletsch_avg_50m.to_dataframe()
df_pxs_smb0020 = da_smb0020.to_dataframe() # .reset_index()
df_pxs_smb1020 = da_smb1020.to_dataframe() # .reset_index()
df_pxs_smb1520 = da_smb1520.to_dataframe() # .

## combine dataframes on x,y index
df_pxsA = df_pxs_uav[['SMB']].join(df_pxs_smb0020[['SMB']], lsuffix='_uav', rsuffix='_0020')
df_pxsB = df_pxs_uav[['SMB']].join(df_pxs_smb1020[['SMB']], lsuffix='_uav', rsuffix='_1020')
df_pxsC = df_pxs_uav[['SMB']].join(df_pxs_smb1520[['SMB']], lsuffix='_uav', rsuffix='_1520')
df_pxs = df_pxsA.join(df_pxsB[['SMB_1020']]).join(df_pxsC[['SMB_1520']])
df_pxs = df_pxs.dropna()
df_pxs.reset_index(inplace=True)
df_pxs


## get simple regression for each scatterplot
x1, y1 = df_pxs['SMB_uav'].values, df_pxs['SMB_0020'].values
x2, y2 = df_pxs['SMB_uav'].values, df_pxs['SMB_1020'].values
x3, y3 = df_pxs['SMB_uav'].values, df_pxs['SMB_1520'].values  
from sklearn.linear_model import LinearRegression
model1 = LinearRegression().fit(x1.reshape(-1,1), y1)
model2 = LinearRegression().fit(x2.reshape(-1,1), y2)
model3 = LinearRegression().fit(x3.reshape(-1,1), y3)
r2_1 = model1.score(x1.reshape(-1,1), y1)
r2_2 = model2.score(x2.reshape(-1,1), y2)
r2_3 = model3.score(x3.reshape(-1,1), y3)
print(f'R2 2000-2020: {r2_1:.2f}, R2 2010-2020: {r2_2:.2f}, R2 2015-2020: {r2_3:.2f}')




'''----------------------
#### PLOT FIGURE ALETSCH
------------------------- '''

fig,axs = plt.subplots(1,3, figsize=(15,6))

''' ######## SPATIAL plot aletsch average + my SMB ########### '''

ax=axs[0]#[0,0]
da_aletsch_avg.plot.imshow(ax=ax, vmin=-8, vmax=8, cmap='RdBu', 
                        add_colorbar=False,
                        # cbar_kwargs={'shrink': 0.7,'fraction':0.036}
                        ); 
gdf_aletsch.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
ax.set_title('SMB from field measurements'); 
ax.set_ylim(da_smb0020.y.min(), da_smb0020.y.max())
## add textbox with time period
ax.text(0.02, 0.95, '2000-2020', 
        horizontalalignment='left', verticalalignment='bottom', 
        transform=ax.transAxes)

ax=axs[1]
h = da_smb0020.plot.imshow(ax=ax, vmin=-8, vmax=8, cmap='RdBu', cbar_kwargs={'shrink': 0.7,'fraction':0.036}); 
gdf_aletsch.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
ax.set_title('SMB from remote sensing'); 
ax.set_ylim(da_smb0020.y.min(), da_smb0020.y.max())
ax.text(0.02, 0.95, '2000-2020', 
        horizontalalignment='left', verticalalignment='bottom', 
        transform=ax.transAxes)

[ax.set_xlabel('x [m]') for ax in axs.flatten()]; [ax.set_ylabel('y [m]') for ax in axs.flatten()]
[ax.set_aspect('equal') for ax in axs[:2]]
[ax.set_xticks([]) for ax in axs[:2]]; [ax.set_xlabel('') for ax in axs[:2]];
[ax.set_yticks([]) for ax in axs[:2]]; [ax.set_ylabel('') for ax in axs[:2]];

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    )
axs[0].add_artist(scalebar)

## move axis a bit to the left to make sapce for next plot
## also move colorbar of second plot to the left
pos = ax.get_position()
pos.x0 -= 0.05
pos.x1 -= 0.05
ax.set_position(pos)
cbar = h.colorbar
cbar.ax.set_position([pos.x1 + 0.01, pos.y0+pos.height*0.25, 0.01, pos.height*0.5])
cbar.set_ticks([-8, -4, 0, 4, 8])
cbar.set_ticklabels(['-8', '-4', '0', '4', '8'])
cbar.set_label('m w.e. yr$^{-1}$',)
# break

''' ######## RIGHT PANEL scatterplot + spatial diff '''


#### SCATTER REGRESSION
ax = axs[2]#,0]

ax.scatter(da_aletsch_avg_50m.values.flatten(), da_smb0020.values.flatten(), 
           s=5, marker='^', alpha=0.5, label='2000-2020', color=my_palette[0])   
xlim = ax.get_xlim(); ylim = ax.get_ylim();  
ax.plot(xlim, xlim, color='black', linestyle='--')
## add regression lines
x_vals = np.array(xlim)
y_vals1 = model1.intercept_ + model1.coef_[0] * x_vals
ax.plot(x_vals, y_vals1, color='black', linestyle='-', label=f'(R²={r2_1:.2f})')
ax.grid(True)

ax.set_xlabel('SMB from in-situ measurements [m w.e. yr$^{-1}$]')
ax.set_ylabel('SMB from remote sensing [m w.e. yr$^{-1}$]')
ax.set_title('SMB comparsion')
ax.legend()

## add MAE and RMSE as txt to figure
axs[2].text(0.95, 0.05, f'MAE: {mae_0020:.2f}\nRMSE: {rmse_0020:.2f}', horizontalalignment='right',
              transform=axs[2].transAxes, 
              color='black', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

# fig.tight_layout()
if save_fig:
    fig.savefig(os.path.join(path2save, 'compare_SMB_fields_and_scatter_aletsch.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(path2save, 'compare_SMB_fields_and_scatter_aletsch.pdf'), bbox_inches='tight')


#%% Also Morteratsch same style
''' ------------------
### MORTERATSCH
---------------------- '''

glacier_rgiid = 'RGI60-11.01946' # Morteratsch

gdf_morteratsch = gl_outline_swiss.loc[gl_outline_swiss['RGIId']==glacier_rgiid].copy()
gdf_morteratsch.to_crs(swiss_crs, inplace=True)

'''##  load morteratsch data '''
path2morteratsch = '/Users/mizeboud/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data_iCloud/SMB2D/vanTricht2021/'

## load mortertsch data
da_mort1718 = xr.open_dataarray( os.path.join(path2morteratsch, 'SMB1718.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_mort1819 = xr.open_dataarray( os.path.join(path2morteratsch, 'SMB1819.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_mort1920 = xr.open_dataarray( os.path.join(path2morteratsch, 'SMB1920.tif') ).isel(band=0).drop_vars('band').rename('SMB')
# da_mort1718.plot.imshow()

## get morteratsch average
da_mort_avg = xr.concat([da_mort1718, da_mort1819, da_mort1920], dim='time').mean(dim='time')

## Get My Morteratsch smb
path2smb = '/Users/mizeboud/Library/Mobile Documents/com~apple~CloudDocs/Documents/Data_iCloud/SMB2D/SMB2D/bruteForce/bestParams_f001_F080_N9/'
da_smb0020 = xr.open_dataarray( os.path.join(path2smb, 'smb_2000-2020/', f'{glacier_rgiid}_smb-kgyr_f001_F080_N9.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_smb0020 = da_smb0020.rio.reproject(swiss_crs, inplace=True)
da_smb1520 = xr.open_dataarray( os.path.join(path2smb, 'smb_2015-2020/', f'{glacier_rgiid}_smb-kgyr_f001_F080_N9.tif') ).isel(band=0).drop_vars('band').rename('SMB')
da_smb1520 = da_smb1520.rio.reproject(swiss_crs, inplace=True)


## clip da_mort to same bounds
myextent = da_smb0020.rio.bounds()
da_mort_avg = da_mort_avg.rio.clip_box(minx=myextent[0], miny=myextent[1], maxx=myextent[2], maxy=myextent[3])
## match da_mort grid to my SMB fields
da_mort_avg_50m = reproject_match_grid(da_smb0020, da_mort_avg, resample_method=rio.enums.Resampling.bilinear)


## caclulcate pixel diff
diff_0020 = da_smb0020 - da_mort_avg_50m
diff_1020 = da_smb1020 - da_mort_avg_50m
diff_1520 = da_smb1520 - da_mort_avg_50m


### calculate MAE and RMSE
mae_0020 = np.nanmean( np.abs( diff_0020.values.flatten() ) )
rmse_0020 = np.sqrt( np.nanmean( diff_0020.values.flatten()**2 ) )
mae_1020 = np.nanmean( np.abs( diff_1020.values.flatten() ) )
rmse_1020 = np.sqrt( np.nanmean( diff_1020.values.flatten()**2 ) )
mae_1520 = np.nanmean( np.abs( diff_1520.values.flatten() ) )
rmse_1520 = np.sqrt( np.nanmean( diff_1520.values.flatten()**2 ) )
print(f'MAE 2000-2020: {mae_0020:.2f}, RMSE 2000-2020: {rmse_0020:.2f}')
print(f'MAE 2010-2020: {mae_1020:.2f}, RMSE 2010-2020: {rmse_1020:.2f}')
print(f'MAE 2015-2020: {mae_1520:.2f}, RMSE 2015-2020: {rmse_1520:.2f}')


''' ----------
## pandas dataframe for Regression plot 
-------------- '''
import pandas as pd
import seaborn as sns
df_pxs = pd.DataFrame()
df_pxs_uav = da_mort_avg_50m.to_dataframe()
df_pxs_smb0020 = da_smb0020.to_dataframe() # .rese
df_pxs_smb1520 = da_smb1520.to_dataframe().rename(columns={'SMB': 'SMB_1520'}) # .reset_index()

## combine dataframes on x,y index
df_pxsA = df_pxs_uav[['SMB']].join(df_pxs_smb0020[['SMB']], lsuffix='_uav', rsuffix='_0020')
df_pxs = df_pxsA.join(df_pxs_smb1520[['SMB_1520']])
df_pxs = df_pxs.dropna()
df_pxs.reset_index(inplace=True)
df_pxs
## get simple regression for each scatterplot
x1, y1 = df_pxs['SMB_uav'].values, df_pxs['SMB_0020'].values
x3, y3 = df_pxs['SMB_uav'].values, df_pxs['SMB_1520'].values
from sklearn.linear_model import LinearRegression
model1 = LinearRegression().fit(x1.reshape(-1,1), y1)
r2_1 = model1.score(x1.reshape(-1,1), y1)
model3 = LinearRegression().fit(x3.reshape(-1,1), y3)
r2_3 = model3.score(x3.reshape(-1,1), y3)
print(f'R2 2000-2020: {r2_1:.2f}, R2 2015-2020: {r2_3:.2f}')


''' ----------
## PLOT MORTERATSCH
-------------- '''

fig,axs = plt.subplots(1,3, figsize=(15,6))

''' ######## SPATIAL plot aletsch average + my SMB ########### '''
da_smb_plot = da_smb0020.copy(); year_plot = '2000-2020'
da_smb_plot = da_smb1520.copy(); year_plot = '2015-2020'

ax=axs[0]#[0,0]
h1 = da_mort_avg.plot.imshow(ax=ax, vmin=-8, vmax=8, cmap='RdBu', 
                        # add_colorbar=False,
                        cbar_kwargs={'shrink': 0.7,'fraction':0.036}
                        ); 
gdf_morteratsch.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
ax.set_title('SMB from field measurements'); 
ax.set_ylim(da_smb_plot.y.min(), da_smb0020.y.max())
## add textbox with time period
ax.text(0.02, 0.95, '2017-2020', 
        horizontalalignment='left', verticalalignment='bottom', 
        transform=ax.transAxes)
xlim_1 = ax.get_xlim(); ylim_1 = ax.get_ylim()
cbar = h.colorbar
## remove cbar
cbar.remove()

ax=axs[1]
h = da_smb_plot.plot.imshow(ax=ax, vmin=-8, vmax=8, cmap='RdBu', cbar_kwargs={'shrink': 0.7,'fraction':0.036}); 
gdf_morteratsch.boundary.plot(ax=ax, edgecolor='black', linewidth=1)
ax.set_title('SMB from remote sensing'); 
ax.set_ylim(da_smb_plot.y.min(), da_smb0020.y.max())
ax.text(0.02, 0.95, year_plot, 
        horizontalalignment='left', verticalalignment='bottom', 
        transform=ax.transAxes)
# ax.set_xlim(xlim_1); ax.set_ylim(ylim_1)

[ax.set_xlabel('x [m]') for ax in axs.flatten()]; [ax.set_ylabel('y [m]') for ax in axs.flatten()]
# [ax.set_aspect('equal') for ax in axs[:2]]
[ax.set_xticks([]) for ax in axs[:2]]; [ax.set_xlabel('') for ax in axs[:2]];
[ax.set_yticks([]) for ax in axs[:2]]; [ax.set_ylabel('') for ax in axs[:2]];

## add scalebar
from matplotlib_scalebar.scalebar import ScaleBar
scalebar=ScaleBar(dx=1, # size of pixel
                    units='m',
                    location='lower left',
                    scale_loc='top',
                    box_alpha=0.5,
                    )
axs[0].add_artist(scalebar)
# ax.set_axis_off()

# move axis a bit to the left to make sapce for next plot
# also move colorbar of second plot to the left
pos = ax.get_position()
pos.x0 -= 0.05
pos.x1 -= 0.05
ax.set_position(pos)
cbar = h.colorbar
cbar.ax.set_position([pos.x1 + 0.01, pos.y0+pos.height*0.25, 0.01, pos.height*0.5])
cbar.set_ticks([-8, -4, 0, 4, 8])
cbar.set_ticklabels(['-8', '-4', '0', '4', '8'])
cbar.set_label('m w.e. yr$^{-1}$',)
# break

''' ######## RIGHT PANEL scatterplot + spatial diff '''

#### SCATTER REGRESSION
ax = axs[2]#,0]

ax.scatter(da_mort_avg_50m.values.flatten(), da_smb0020.values.flatten(), 
           s=5, marker='^', alpha=0.5, label='2000-2020', color=my_palette[0])   
ax.scatter(da_mort_avg_50m.values.flatten(), da_smb1520.values.flatten(), 
           s=5, marker='^', alpha=0.5, label='2015-2020', color=my_palette[2])  
xlim = ax.get_xlim(); ylim = ax.get_ylim();  
ax.plot(xlim, xlim, color='black', linestyle='--')
## add regression lines
x_vals = np.array(xlim)
y_vals1 = model1.intercept_ + model1.coef_[0] * x_vals
y_vals3 = model3.intercept_ + model3.coef_[0] * x_vals
ax.plot(x_vals, y_vals1, color=my_palette[0], linestyle='-', label=f'(R²={r2_1:.2f})')
ax.plot(x_vals, y_vals3, color=my_palette[2], linestyle='-', label=f'(R²={r2_3:.2f})')
ax.grid(True)

ax.set_xlabel('SMB from in-situ measurements [m w.e. yr$^{-1}$]')
ax.set_ylabel('SMB from remote sensing [m w.e. yr$^{-1}$]')
ax.set_title('SMB comparison')
ax.legend()

## add MAE and RMSE as txt to figure
axs[2].text(0.05, 0.05, f'MAE: {mae_0020:.2f}\nRMSE: {rmse_0020:.2f}', 
            horizontalalignment='left',
              transform=axs[2].transAxes, 
              color='black', fontsize=12, 
              bbox=dict(facecolor='white', alpha=0.7, edgecolor=my_palette[0]))
axs[2].text(0.95, 0.05, f'MAE: {mae_1520:.2f}\nRMSE: {rmse_1520:.2f}', horizontalalignment='right',
              transform=axs[2].transAxes, 
              color='black', fontsize=12, 
              bbox=dict(facecolor='white', alpha=0.7, edgecolor=my_palette[2]))

# fig.tight_layout()
if save_fig:
    fig.savefig(os.path.join(path2save, 'compare_SMB_fields_and_scatter_morteratsch.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(path2save, 'compare_SMB_fields_and_scatter_morteratsch.pdf'), bbox_inches='tight')

# %%
