# ---- This is <check_ROIs_and_results_visually.py> ----

"""
Visualize and check ROIs and classification results.
""" 

import os
import sys
import pathlib
from loguru import logger
import pickle

from osgeo import gdal

import labelme_utils.json_conversion as lm_json

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

#orbit = '043029_05233F'
orbit = '043044_0523D1'

# cropped image coordinates (must be the same as in "make_scaled_RGBs_from_AOI_crops.py")
xmin = 800
xmax = 10000
ymin = 2500
ymax = 10000

# parameters for RGB scaling
vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

# linewidth for ROIs
linewidth = 4

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# build color map and legend

# get viridis color map (because it looks ok)
CMAP = mpl.cm.get_cmap('viridis')

# initialize normalized class colors with grey background
class_colors_norm = [[0.2, 0.2, 0.2]]

# append 4 equally spaced viridis colors
for idx in np.round(np.linspace(0,255,4)).astype(int):
    class_colors_norm.append(CMAP.colors[idx])

# define level boundaries for colormap
cmap_bounds = np.arange(-0.5,len(class_colors_norm)+0.5,1)
cmap_values = np.convolve(cmap_bounds, np.ones(2)/2, mode='valid').astype(int)

# build colormap
cmap = mpl.colors.ListedColormap(class_colors_norm,name='belgica_bank_cmap')

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

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get class polygons

# define json path
json_path = S1_RGB_DIR / f'{orbit}_proposed_RGB_CROP.json'
labels_path = 'config/labels.txt'

# get class name list
class_names = lm_json.get_class_name_list_from_labels_txt(labels_path)

# get class_labels_dict
class_labels_dict = lm_json.get_label_index_mapping(class_names)

# get polygons from json file
shapes, label_index_dict = lm_json.load_training_shapes(json_path, label_index_mapping=class_labels_dict)

# get number of polygons
N_polygons = np.size(shapes)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

logger.info('Loading intensities ...')

# load intensities
intensities_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_1x1_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_proposed_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_9x9_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_21x21_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_MuLoG_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_SARBM3D_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_baseline_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)


logger.info('Loading labels ...')

# load labels
labels_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_1x1_crop.tiff').as_posix()).ReadAsArray()
labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()
labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
labels_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_21x21_crop.tiff').as_posix()).ReadAsArray()
labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_MuLoG_crop.tiff').as_posix()).ReadAsArray()
labels_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_SARBM3D_crop.tiff').as_posix()).ReadAsArray()
labels_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_baseline_crop.tiff').as_posix()).ReadAsArray()


logger.info('Loading validation mask ...')

# load validation mask
validation_mask = gdal.Open((S1_VAL_DIR / f'{orbit}_proposed_RGB_CROP_training_mask.img').as_posix()).ReadAsArray()

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

logger.info('Making intensity RGBs ...')

HH_ML_1x1   = intensities_ML_1x1[:,:,0]
HV_ML_1x1   = intensities_ML_1x1[:,:,1]
HH_proposed = intensities_proposed[:,:,0]
HV_proposed = intensities_proposed[:,:,1]
HH_ML_9x9   = intensities_ML_9x9[:,:,0]
HV_ML_9x9   = intensities_ML_9x9[:,:,1]
HH_ML_21x21 = intensities_ML_21x21[:,:,0]
HV_ML_21x21 = intensities_ML_21x21[:,:,1]
HH_MuLoG    = intensities_MuLoG[:,:,0]
HV_MuLoG    = intensities_MuLoG[:,:,1]
HH_SARBM3D  = intensities_SARBM3D[:,:,0]
HV_SARBM3D  = intensities_SARBM3D[:,:,1]
HH_baseline = intensities_baseline[:,:,0]
HV_baseline = intensities_baseline[:,:,1]

HH_ML_1x1[HH_ML_1x1==0] = np.nan
HV_ML_1x1[HV_ML_1x1==0] = np.nan
HH_proposed[HH_proposed==0] = np.nan
HV_proposed[HV_proposed==0] = np.nan
HH_ML_9x9[HH_ML_9x9==0] = np.nan
HV_ML_9x9[HV_ML_9x9==0] = np.nan
HH_ML_21x21[HH_ML_21x21==0] = np.nan
HV_ML_21x21[HV_ML_21x21==0] = np.nan
HH_MuLoG[HH_MuLoG==0] = np.nan
HV_MuLoG[HV_MuLoG==0] = np.nan
HH_SARBM3D[HH_SARBM3D==0] = np.nan
HV_SARBM3D[HV_SARBM3D==0] = np.nan
HH_baseline[HH_baseline==0] = np.nan
HV_baseline[HV_baseline==0] = np.nan


# set new min/max values for RGBs
new_min = 0
new_max = 1

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_1x1_scaled  = (HH_ML_1x1 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_1x1_scaled  = (HV_ML_1x1 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_1x1_scaled  = np.clip(HH_ML_1x1_scaled, new_min, new_max)
HV_ML_1x1_scaled  = np.clip(HV_ML_1x1_scaled, new_min, new_max)

# stack to false-color RGB
RGB_ML_1x1 = np.stack((HV_ML_1x1_scaled, HH_ML_1x1_scaled, HH_ML_1x1_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_proposed_scaled  = (HH_proposed - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_proposed_scaled  = (HV_proposed - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_proposed_scaled  = np.clip(HH_proposed_scaled, new_min, new_max)
HV_proposed_scaled  = np.clip(HV_proposed_scaled, new_min, new_max)

# stack to false-color RGB
RGB_proposed = np.stack((HV_proposed_scaled, HH_proposed_scaled, HH_proposed_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_9x9_scaled  = (HH_ML_9x9 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_9x9_scaled  = (HV_ML_9x9 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_9x9_scaled  = np.clip(HH_ML_9x9_scaled, new_min, new_max)
HV_ML_9x9_scaled  = np.clip(HV_ML_9x9_scaled, new_min, new_max)

# stack to false-color RGB
RGB_ML_9x9 = np.stack((HV_ML_9x9_scaled, HH_ML_9x9_scaled, HH_ML_9x9_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_21x21_scaled  = (HH_ML_21x21 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_21x21_scaled  = (HV_ML_21x21 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_21x21_scaled  = np.clip(HH_ML_21x21_scaled, new_min, new_max)
HV_ML_21x21_scaled  = np.clip(HV_ML_21x21_scaled, new_min, new_max)

# stack to false-color RGB
RGB_ML_21x21 = np.stack((HV_ML_21x21_scaled, HH_ML_21x21_scaled, HH_ML_21x21_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_MuLoG_scaled  = (HH_MuLoG - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_MuLoG_scaled  = (HV_MuLoG - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_MuLoG_scaled  = np.clip(HH_MuLoG_scaled, new_min, new_max)
HV_MuLoG_scaled  = np.clip(HV_MuLoG_scaled, new_min, new_max)

# stack to false-color RGB
RGB_MuLoG = np.stack((HV_MuLoG_scaled, HH_MuLoG_scaled, HH_MuLoG_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_SARBM3D_scaled  = (HH_SARBM3D - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_SARBM3D_scaled  = (HV_SARBM3D - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_SARBM3D_scaled  = np.clip(HH_SARBM3D_scaled, new_min, new_max)
HV_SARBM3D_scaled  = np.clip(HV_SARBM3D_scaled, new_min, new_max)

# stack to false-color RGB
RGB_SARBM3D = np.stack((HV_SARBM3D_scaled, HH_SARBM3D_scaled, HH_SARBM3D_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_baseline_scaled  = (HH_baseline - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_baseline_scaled  = (HV_baseline - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_baseline_scaled  = np.clip(HH_baseline_scaled, new_min, new_max)
HV_baseline_scaled  = np.clip(HV_baseline_scaled, new_min, new_max)

# stack to false-color RGB
RGB_baseline = np.stack((HV_baseline_scaled, HH_baseline_scaled, HH_baseline_scaled),2)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

logger.info('Cropping intensity and label images ...')

RGB_ML_1x1_crop   = RGB_ML_1x1[xmin:xmax,ymin:ymax,:]
RGB_proposed_crop = RGB_proposed[xmin:xmax,ymin:ymax,:]
RGB_ML_9x9_crop   = RGB_ML_9x9[xmin:xmax,ymin:ymax,:]
RGB_ML_21x21_crop = RGB_ML_21x21[xmin:xmax,ymin:ymax,:]
RGB_SARBM3D_crop  = RGB_SARBM3D[xmin:xmax,ymin:ymax,:]
RGB_MuLoG_crop    = RGB_MuLoG[xmin:xmax,ymin:ymax,:]
RGB_baseline_crop = RGB_baseline[xmin:xmax,ymin:ymax,:]

labels_ML_1x1_crop   = labels_ML_1x1[xmin:xmax,ymin:ymax]
labels_proposed_crop = labels_proposed[xmin:xmax,ymin:ymax]
labels_ML_9x9_crop   = labels_ML_9x9[xmin:xmax,ymin:ymax]
labels_ML_21x21_crop = labels_ML_21x21[xmin:xmax,ymin:ymax]
labels_SARBM3D_crop  = labels_SARBM3D[xmin:xmax,ymin:ymax]
labels_MuLoG_crop    = labels_MuLoG[xmin:xmax,ymin:ymax]
labels_baseline_crop = labels_baseline[xmin:xmax,ymin:ymax]

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# ONLY PROPOSED METHOD AND ROIS

fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
axes = axes.ravel()

axes[0].imshow(RGB_proposed_crop)
axes[1].imshow(validation_mask, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[2].imshow(labels_proposed_crop, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')

# loop over all polygons and draw them onto the figure
for i in np.arange(N_polygons):
    p   = shapes[i]
    cl  = label_index_dict[p['label']]
    polygon = np.array(p['points'])

    xmin, xmax = polygon[:,0]
    ymin, ymax = polygon[:,1]
    xvec = [xmin, xmax, xmax, xmin, xmin]
    yvec = [ymin, ymin, ymax, ymax, ymin]    

    axes[0].plot(xvec,yvec,color = class_colors_norm[cl], linewidth=linewidth)
    axes[2].plot(xvec,yvec,color = [0,0,0], linewidth=linewidth)

plt.show()


# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# PROPOSED METHOD AND ROIS TOGETHER WITH ONE COMPARISON


other_RGB    = RGB_ML_1x1_crop
other_labels = labels_ML_1x1_crop

linewidth = 4

fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes = axes.ravel()

axes[0].imshow(other_RGB)
axes[1].imshow(RGB_proposed_crop)
axes[2].imshow(validation_mask, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[3].imshow(other_labels, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[4].imshow(labels_proposed_crop, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')

# loop over all polygons and draw them onto the figure
for i in np.arange(N_polygons):
    p   = shapes[i]
    cl  = label_index_dict[p['label']]
    polygon = np.array(p['points'])

    xmin, xmax = polygon[:,0]
    ymin, ymax = polygon[:,1]
    xvec = [xmin, xmax, xmax, xmin, xmin]
    yvec = [ymin, ymin, ymax, ymax, ymin]    

    axes[0].plot(xvec,yvec,color = class_colors_norm[cl], linewidth=linewidth)
    axes[1].plot(xvec,yvec,color = class_colors_norm[cl], linewidth=linewidth)
    axes[3].plot(xvec,yvec,color = [0,0,0], linewidth=linewidth)
    axes[4].plot(xvec,yvec,color = [0,0,0], linewidth=linewidth)

plt.show()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <check_ROIs_and_results_visually.py> ----


