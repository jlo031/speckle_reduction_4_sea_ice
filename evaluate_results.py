# ---- This is <evaluate_results.py> ----

"""
Numerically evaluate classification results from one individual orbit
""" 

import os
import sys
import pathlib
from loguru import logger
import pickle

from osgeo import gdal

import numpy as np

from sklearn.metrics import confusion_matrix

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# S1 orbit to process
#orbit = '043029_05233F'
orbit = '043044_0523D1'

# cropped image coordinates (must be the same as in "make_scaled_RGBs_from_AOI_crops.py")
xmin = 800
xmax = 10000
ymin = 2500
ymax = 10000

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

logger.info('Loading classification results ...')

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

# crop labels to same size as cropped RGB that was used for labeling

logger.info('Cropping label images ...')

labels_ML_1x1_crop   = labels_ML_1x1[xmin:xmax,ymin:ymax]
labels_proposed_crop = labels_proposed[xmin:xmax,ymin:ymax]
labels_ML_9x9_crop   = labels_ML_9x9[xmin:xmax,ymin:ymax]
labels_ML_21x21_crop = labels_ML_21x21[xmin:xmax,ymin:ymax]
labels_SARBM3D_crop  = labels_SARBM3D[xmin:xmax,ymin:ymax]
labels_MuLoG_crop    = labels_MuLoG[xmin:xmax,ymin:ymax]
labels_baseline_crop = labels_baseline[xmin:xmax,ymin:ymax]

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

logger.info('Calculating result statistics ...')


# select validation labels
y_true = validation_mask[validation_mask!=0]

y_ML_1x1   = labels_ML_1x1[xmin:xmax,ymin:ymax][validation_mask!=0]
y_ML_9x9   = labels_ML_9x9[xmin:xmax,ymin:ymax][validation_mask!=0]
y_ML_21x21 = labels_ML_21x21[xmin:xmax,ymin:ymax][validation_mask!=0]
y_SARBM3D  = labels_SARBM3D[xmin:xmax,ymin:ymax][validation_mask!=0]
y_MuLoG    = labels_MuLoG[xmin:xmax,ymin:ymax][validation_mask!=0]
y_baseline = labels_baseline[xmin:xmax,ymin:ymax][validation_mask!=0]
y_proposed = labels_proposed[xmin:xmax,ymin:ymax][validation_mask!=0]

# compute confusion matrices
cm_y_ML_1x1 = confusion_matrix(y_true, y_ML_1x1, normalize='true')
cm_y_ML_9x9 = confusion_matrix(y_true, y_ML_9x9, normalize='true')
cm_y_ML_21x21 = confusion_matrix(y_true, y_ML_21x21, normalize='true')
cm_y_SARBM3D = confusion_matrix(y_true, y_SARBM3D, normalize='true')
cm_y_MuLoG = confusion_matrix(y_true, y_MuLoG, normalize='true')
cm_y_baseline = confusion_matrix(y_true, y_baseline, normalize='true')
cm_y_proposed = confusion_matrix(y_true, y_proposed, normalize='true')

# extract CA
CA_ML_1x1   = np.round(100*cm_y_ML_1x1.diagonal(),1)
CA_ML_9x9   = np.round(100*cm_y_ML_9x9.diagonal(),1)
CA_ML_21x21 = np.round(100*cm_y_ML_21x21.diagonal(),1)
CA_y_SARBM3D  = np.round(100*cm_y_SARBM3D.diagonal(),1)
CA_y_MuLoG    = np.round(100*cm_y_MuLoG.diagonal(),1)
CA_y_baseline = np.round(100*cm_y_baseline.diagonal(),1)
CA_y_proposed = np.round(100*cm_y_proposed.diagonal(),1)


# print to screen
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

logger.info(f'ML_1x1    average per-class CA: {np.round(CA_ML_1x1.mean(),1)}')
logger.info(f'ML_9x9    average per-class CA: {np.round(CA_ML_9x9.mean(),1)}')
logger.info(f'ML_21x21  average per-class CA: {np.round(CA_ML_21x21.mean(),1)}')
logger.info(f'SARBM3D   average per-class CA: {np.round(CA_y_SARBM3D.mean(),1)}')
logger.info(f'MuLoG     average per-class CA: {np.round(CA_y_MuLoG.mean(),1)}')
logger.info(f'baseline  average per-class CA: {np.round(CA_y_baseline.mean(),1)}')
logger.info(f'proposed  average per-class CA: {np.round(CA_y_proposed.mean(),1)}\n')

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# ---- End of <evaluate_results.py> ----


