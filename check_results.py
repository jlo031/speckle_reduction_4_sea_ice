# ---- This is <check_results.py> ----

"""
Check classification results for input imgage.
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

vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

# cropped image coordinates (must be the same as in "make_scaled_RGBs_from_AOI_crops.py")
xmin = 800
xmax = 10000
ymin = 2500
ymax = 10000

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

logger.info('Loading data ...')

# load intensities
intensities_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_1x1_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_proposed_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)

# load labels
labels_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_1x1_crop.tiff').as_posix()).ReadAsArray()
labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()

"""
intensities_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_9x9_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_21x21_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_MuLoG_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_SARBM3D_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_baseline_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
labels_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_21x21_crop.tiff').as_posix()).ReadAsArray()
labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_MuLoG_crop.tiff').as_posix()).ReadAsArray()
labels_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_SARBM3D_crop.tiff').as_posix()).ReadAsArray()
labels_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_baseline_crop.tiff').as_posix()).ReadAsArray()
"""

# load training mask
training_mask = gdal.Open((S1_VAL_DIR / f'{orbit}_proposed_RGB_CROP_training_mask.img').as_posix()).ReadAsArray()

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# make intensity RGBs
logger.info('Making intensity RGBs ...')

HH_ML_1x1   = intensities_ML_1x1[:,:,0]
HV_ML_1x1   = intensities_ML_1x1[:,:,1]
HH_proposed = intensities_proposed[:,:,0]
HV_proposed = intensities_proposed[:,:,1]

HH_ML_1x1[HH_ML_1x1==0] = np.nan
HV_ML_1x1[HV_ML_1x1==0] = np.nan
HH_proposed[HH_proposed==0] = np.nan
HV_proposed[HV_proposed==0] = np.nan

"""
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
"""

# if you want to stack to false-color RGB, set new min/max values
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

"""
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
"""

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

RGB_ML_1x1_crop   = RGB_ML_1x1[xmin:xmax,ymin:ymax,:]
RGB_proposed_crop = RGB_proposed[xmin:xmax,ymin:ymax,:]
labels_ML_1x1_crop   = labels_ML_1x1[xmin:xmax,ymin:ymax]
labels_proposed_crop = labels_proposed[xmin:xmax,ymin:ymax]

"""
RGB_ML_9x9_crop   = RGB_ML_9x9[xmin:xmax,ymin:ymax,:]
RGB_ML_21x21_crop = RGB_ML_21x21[xmin:xmax,ymin:ymax,:]
RGB_SARBM3D_crop  = RGB_SARBM3D[xmin:xmax,ymin:ymax,:]
RGB_MuLoG_crop    = RGB_MuLoG[xmin:xmax,ymin:ymax,:]
RGB_baseline_crop = RGB_baseline[xmin:xmax,ymin:ymax,:]
labels_ML_9x9_crop   = labels_ML_9x9[xmin:xmax,ymin:ymax]
labels_ML_21x21_crop = labels_ML_21x21[xmin:xmax,ymin:ymax]
labels_SARBM3D_crop  = labels_SARBM3D[xmin:xmax,ymin:ymax]
labels_MuLoG_crop    = labels_MuLoG[xmin:xmax,ymin:ymax]
labels_baseline_crop = labels_baseline[xmin:xmax,ymin:ymax]
"""

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #



linewidth = 4

fig, axes = plt.subplots(2,2,sharex=True,sharey=True)
axes = axes.ravel()

axes[0].imshow(RGB_proposed_crop)
axes[1].imshow(training_mask, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
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

















linewidth = 4

fig, axes = plt.subplots(2,3,sharex=True,sharey=True)
axes = axes.ravel()

axes[0].imshow(RGB_ML_1x1_crop)
axes[1].imshow(RGB_proposed_crop)
axes[2].imshow(training_mask, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[3].imshow(labels_ML_1x1_crop, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
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

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #





# compute confusion matrices

y_true = training_mask[training_mask!=0]

y_ML_1x1   = labels_ML_1x1[xmin:xmax,ymin:ymax][training_mask!=0]
y_ML_9x9   = labels_ML_9x9[xmin:xmax,ymin:ymax][training_mask!=0]
y_ML_21x21 = labels_ML_21x21[xmin:xmax,ymin:ymax][training_mask!=0]
y_SARBM3D  = labels_SARBM3D[xmin:xmax,ymin:ymax][training_mask!=0]
y_MuLoG    = labels_MuLoG[xmin:xmax,ymin:ymax][training_mask!=0]
y_baseline = labels_baseline[xmin:xmax,ymin:ymax][training_mask!=0]
y_proposed = labels_proposed[xmin:xmax,ymin:ymax][training_mask!=0]








cm_y_ML_1x1 = confusion_matrix(y_true, y_ML_1x1, normalize='true')
cm_y_ML_9x9 = confusion_matrix(y_true, y_ML_9x9, normalize='true')
cm_y_ML_21x21 = confusion_matrix(y_true, y_ML_21x21, normalize='true')
cm_y_SARBM3D = confusion_matrix(y_true, y_SARBM3D, normalize='true')
cm_y_MuLoG = confusion_matrix(y_true, y_MuLoG, normalize='true')
cm_y_baseline = confusion_matrix(y_true, y_baseline, normalize='true')
cm_y_proposed = confusion_matrix(y_true, y_proposed, normalize='true')

CA_ML_1x1   = np.round(100*cm_y_ML_1x1.diagonal(),1)
CA_ML_9x9   = np.round(100*cm_y_ML_9x9.diagonal(),1)
CA_ML_21x21 = np.round(100*cm_y_ML_21x21.diagonal(),1)
CA_y_SARBM3D  = np.round(100*cm_y_SARBM3D.diagonal(),1)
CA_y_MuLoG    = np.round(100*cm_y_MuLoG.diagonal(),1)
CA_y_baseline = np.round(100*cm_y_baseline.diagonal(),1)
CA_y_proposed = np.round(100*cm_y_proposed.diagonal(),1)


logger.info(f'ML_1x1    OW CA: {CA_ML_1x1[0]}')
logger.info(f'ML_9x9    OW CA: {CA_ML_9x9[0]}')
logger.info(f'ML_21x21  OW CA: {CA_ML_21x21[0]}')
logger.info(f'SARBM3D   OW CA: {CA_y_SARBM3D[0]}')
logger.info(f'MuLoG     OW CA: {CA_y_MuLoG[0]}')
logger.info(f'baseline  OW CA: {CA_y_baseline[0]}')
logger.info(f'proposed  OW CA: {CA_y_proposed[0]}\n')

logger.info(f'ML_1x1    YI CA: {CA_ML_1x1[1]}')
logger.info(f'ML_9x9    YI CA: {CA_ML_9x9[1]}')
logger.info(f'ML_21x21  YI CA: {CA_ML_21x21[1]}')
logger.info(f'SARBM3D   YI CA: {CA_y_SARBM3D[1]}')
logger.info(f'MuLoG     YI CA: {CA_y_MuLoG[1]}')
logger.info(f'baseline  YI CA: {CA_y_baseline[1]}')
logger.info(f'proposed  YI CA: {CA_y_proposed[1]}\n')

logger.info(f'ML_1x1    LI CA: {CA_ML_1x1[2]}')
logger.info(f'ML_9x9    LI CA: {CA_ML_9x9[2]}')
logger.info(f'ML_21x21  LI CA: {CA_ML_21x21[2]}')
logger.info(f'SARBM3D   LI CA: {CA_y_SARBM3D[2]}')
logger.info(f'MuLoG     LI CA: {CA_y_MuLoG[2]}')
logger.info(f'baseline  LI CA: {CA_y_baseline[2]}')
logger.info(f'proposed  LI CA: {CA_y_proposed[2]}\n')

logger.info(f'ML_1x1    DI CA: {CA_ML_1x1[3]}')
logger.info(f'ML_9x9    DI CA: {CA_ML_9x9[3]}')
logger.info(f'ML_21x21  DI CA: {CA_ML_21x21[3]}')
logger.info(f'SARBM3D   DI CA: {CA_y_SARBM3D[3]}')
logger.info(f'MuLoG     DI CA: {CA_y_MuLoG[3]}')
logger.info(f'baseline  DI CA: {CA_y_baseline[3]}')
logger.info(f'proposed  DI CA: {CA_y_proposed[3]}\n')


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get class polygons

# define json path
json_path = ALIGNED_CROP_SCALED_DIR / f'{img_pair}_L_scaled.json'

# get polygons from json file
shapes, label_index_dict = lm_json.load_training_shapes(json_path, label_index_mapping=class_labels_dict)

# get number of polygons
N_polygons = np.size(shapes)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# define tick locations and labels for current AOI

x_tick_location_vec = [ 0,   200, 400]
x_tick_labels_vec   = ['0', '20', '40']
y_tick_location_vec = [ 0,   200, 400]
y_tick_labels_vec   = ['0', '20', '40']

x_tick_location_vec_fine = [ 0,   100,  200,  300,  400]
x_tick_labels_vec_fine   = ['0', '10', '20', '30', '40']
y_tick_location_vec_fine = [ 0,   100,  200,  300,  400]
y_tick_labels_vec_fine   = ['0', '10', '20', '30', '40']

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# FULL RGB IMAGE WITH TRAINING ROIS

logger.info('Drawing RGB with training polygons')

CA = CA_dict[img_pair]

print(' ')
print(img_pair)
for key in CA.keys():
    print(f'{key}: {np.round(100*CA[key],1)}')
print(' ')

ticklabelsize = 22
labelsize     = 22
linewidth     = 3

training_mask = training_mask.astype(float)
training_mask[training_mask==0] = np.nan

fig, axes = plt.subplots(2,3,figsize=((14,8)), sharex=True, sharey=True)
axes = axes.ravel()

axes[0].imshow(C_scaled/255)
axes[1].imshow(L_scaled/255)
axes[2].imshow(training_mask, cmap = cmap, norm = cmap_norm)

# loop over all polygons and draw them onto the figure
for i in np.arange(N_polygons):
    p   = shapes[i]
    cl  = label_index_dict[p['label']]
    polygon = np.array(p['points'])

    xmin, xmax = polygon[:,0]
    ymin, ymax = polygon[:,1]
    xvec = [xmin, xmax, xmax, xmin, xmin]
    yvec = [ymin, ymin, ymax, ymax, ymin]    

    axes[0].plot(xvec,yvec,color = class_colors_norm[cl-1], linewidth=linewidth)
    axes[1].plot(xvec,yvec,color = class_colors_norm[cl-1], linewidth=linewidth)


axes[3].imshow(C_labels, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[4].imshow(L_labels, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')
axes[5].imshow(LC_labels, cmap = cmap, norm = cmap_norm, interpolation = 'nearest')

plt.show()

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <check_results.py> ----


