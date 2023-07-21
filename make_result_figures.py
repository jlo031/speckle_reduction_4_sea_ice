# ---- This is <make_result_figures.py> ----

"""
Visualize data and results on map.
"""

import os
import pathlib
import shutil

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from osgeo import gdal

import cartopy
import cartopy.crs as ccrs

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

x_grid = [-45, -30, -15, 0, 15, 30]
y_grid = [76, 79, 82, 85]

from folder_structure import *

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# COLOR MAP

# get viridis color map (because it looks ok)
CMAP = mpl.cm.get_cmap('viridis')

# initialize normalized class colors with grey background
class_colors_norm = [[0.2, 0.2, 0.2]]

# append 4 equally spaced viridis colors
for idx in np.round(np.linspace(0,255,4)).astype(int):
    class_colors_norm.append(CMAP.colors[idx])

# define level boundaries for colormap
cmap_bounds = np.arange(
    -0.5,len(class_colors_norm)+0.5,1
)
cmap_values = np.convolve(
    cmap_bounds, np.ones(2)/2, mode='valid'
).astype(int)

# build colormap
cmap = mpl.colors.ListedColormap(
    class_colors_norm,
    name='belgica_bank_cmap'
)

# build a colormap index based on level boundaries
cmap_norm = mpl.colors.BoundaryNorm(cmap_bounds, cmap.N)

legend_entries = [
    'None',
    'OW / New Ice',
    'Young Ice',
    'Level Ice',
    'Deformed Ice'
]
legend_properties = {'weight':'bold'}

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

##orbit = '043029_05233F'
orbit = '043044_0523D1'

# load intensities
intensities_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_1x1_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_9x9_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_21x21_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_MuLoG_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_SARBM3D_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ines     = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ines_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)

# load labels
labels_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_ML_1x1_crop.tiff').as_posix()).ReadAsArray()
labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
labels_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_ML_21x21_crop.tiff').as_posix()).ReadAsArray()
labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_MuLoG_crop.tiff').as_posix()).ReadAsArray()
labels_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_SARBM3D_crop.tiff').as_posix()).ReadAsArray()
labels_ines     = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_valid_ines_crop.tiff').as_posix()).ReadAsArray()

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# make RGB of ines intensities

HH = intensities_ines [:,:,0]
HV = intensities_ines [:,:,1]

# set no data values to nan (they are zero in the geocoded intensity image)
HH[HH==0] = np.nan
HV[HV==0] = np.nan

# if you want to stack to false-color RGB, set new min/max values
new_min = 0
new_max = 1

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_scaled  = (HH - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_scaled  = (HV - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_scaled  = np.clip(HH_scaled, new_min, new_max)
HV_scaled  = np.clip(HV_scaled, new_min, new_max)

# stack to fals-color RGB
RGB = np.stack((HV_scaled, HH_scaled, HH_scaled),2)

# --------------------------------------------------------------- #

# OVERVIEW OF AOI

# make the figure without colorbar

plt.rcParams.update({'font.size': 15})

fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((12,5)))
axes = axes.ravel()
axes[0].imshow(RGB)
h_cbar = axes[1].imshow(labels_ines, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[0].set_title('Intensity RGB')
axes[1].set_title('Class labels')

axes[0].set_xticks([0,2500,5000,7500])
axes[0].set_yticks([0,2500,5000,7500])
axes[0].set_xticklabels(['0','100','200','300'])
axes[0].set_yticklabels(['0','100','200','300'])
axes[0].set_ylabel('Distance north (km)')
axes[0].set_xlabel('Distance east (km)')
axes[1].set_xlabel('Distance east (km)')

plt.savefig('tmp1.png', dpi=300)


# make a figure with same height to crop the colorbar

fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((7,5)))
h_cbar = ax.imshow(labels_ines, interpolation='nearest', cmap=cmap, norm=cmap_norm)
cbar = fig.colorbar(h_cbar, ticks=cmap_values, pad=0.075)
cbar.set_ticklabels(legend_entries, weight='bold')
cbar.ax.tick_params(rotation=-45)

plt.savefig('tmp2.png', dpi=300)

output_path = PAPER_FIG_DIR / f'AOI_result_overview_orbit_{orbit}.png'

crop_colorbar_cmd = 'convert tmp2.png -crop 500x1500+1600+0 colorbar.png'
crop_figure_cmd_1 = 'convert tmp1.png -crop 1500x1500+270+0 tmp3.png'
crop_figure_cmd_2 = 'convert tmp1.png -crop 1200x1500+2000+0 tmp4.png'
combine_figures_cmd = f'convert +append tmp3.png tmp4.png colorbar.png final.png'
crop_final_cmd = f'convert final.png -crop 3500x1410+0+90 {output_path}'


os.system(crop_colorbar_cmd)
os.system(crop_figure_cmd_1)
os.system(crop_figure_cmd_2)
os.system(combine_figures_cmd)
os.system(crop_final_cmd)
os.system('rm -rf tmp1.png tmp2.png tmp3.png tmp4.png colorbar.png final.png')

plt.close('all')

# --------------------------------------------------------------- #

# CLOSEUPS


"""
fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,6)))
axes = axes.ravel()
h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[1].imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[2].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[3].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[4].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[5].imshow(labels_ines, interpolation='nearest', cmap=cmap, norm=cmap_norm)

axes[0].set_title('ML 1x1')
axes[1].set_title('ML 9x9')
axes[2].set_title('ML 21x21')
axes[3].set_title('MuLoG')
axes[4].set_title('SARBM3D')
axes[5].set_title('ines')

plt.show()
"""
# ------------------------- #

# example of "artificial" class in ML due to averaging

fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
axes = axes.ravel()
h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[1].imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[2].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[3].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[4].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[5].imshow(labels_ines, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[0].set_title('ML 1x1')
axes[1].set_title('ML 9x9')
axes[2].set_title('ML 21x21')
axes[3].set_title('MuLoG')
axes[4].set_title('SARBM3D')
axes[5].set_title('ines')

axes[0].set_xlim([8350,8650])
axes[0].set_ylim([2850,2550])

axes[0].set_xticks([8400, 8500, 8600])
axes[0].set_yticks([2800,2700,2600])
axes[0].set_xticklabels(['336','340','344'])
axes[0].set_yticklabels(['112','108','104'])
axes[0].set_ylabel('Distance north (km)')
axes[3].set_ylabel('Distance north (km)')
axes[4].set_xlabel('Distance east (km)')

##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
##cbar.set_ticklabels(legend_entries, weight='bold')
##cbar.ax.tick_params(rotation=-45)

plt.savefig('tmp1.png', dpi=300)

# ------------------------- #

# example of small leads disappearing with ML

fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
axes = axes.ravel()
axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[1].imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[2].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[3].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[4].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[5].imshow(labels_ines, interpolation='nearest', cmap=cmap, norm=cmap_norm)
axes[0].set_title('ML 1x1')
axes[1].set_title('ML 9x9')
axes[2].set_title('ML 21x21')
axes[3].set_title('MuLoG')
axes[4].set_title('SARBM3D')
axes[5].set_title('ines')

axes[0].set_xlim([1150,1450])
axes[0].set_ylim([9900,9600])

axes[0].set_xticks([1200, 1300, 1400])
axes[0].set_yticks([9850,9750,9650])
axes[0].set_xticklabels(['48','52','56'])
axes[0].set_yticklabels(['394','390','386'])
axes[0].set_ylabel('Distance north (km)')
axes[3].set_ylabel('Distance north (km)')
axes[4].set_xlabel('Distance east (km)')


##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
##cbar.set_ticklabels(legend_entries, weight='bold')
##cbar.ax.tick_params(rotation=-45)

plt.savefig('tmp2.png', dpi=300)

# ------------------------- #

output_path_1 = PAPER_FIG_DIR / f'classification_closeup_orbit_{orbit}_example1.png'
output_path_2 = PAPER_FIG_DIR / f'classification_closeup_orbit_{orbit}_example2.png'

crop_cmd_1 = f'convert tmp1.png -crop 3050x2090+210+210 {output_path_1}'
crop_cmd_2 = f'convert tmp2.png -crop 3050x2090+210+210 {output_path_2}'

os.system(crop_cmd_1)
os.system(crop_cmd_2)

os.system('rm tmp1.png tmp2.png')

# --------------------------------------------------------------- #

plt.close('all')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <make_result_figures.py> ----
