# ---- This is <show_results_in_radar_geometry.py> ----

"""
Visualize results in radar geometry for first inspection.
"""

import os
import pathlib
import shutil

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

from folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all S1 basenames
S1_list = [ f.split('.SAFE')[0] for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all S1 images
for S1_name in S1_list:

    logger.info(f'Processing S1 image: {S1_name}')

    # load all labels
    labels_ML_1x1   = gdal.Open((S1_RESULT_DIR/'ML_1x1'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()
    labels_ML_9x9   = gdal.Open((S1_RESULT_DIR/'ML_9x9'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()
    labels_ML_21x21 = gdal.Open((S1_RESULT_DIR/'ML_21x21'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()
    labels_MuLoG    = gdal.Open((S1_RESULT_DIR/'MuLoG'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()
    labels_SARBM3D  = gdal.Open((S1_RESULT_DIR/'SARBM3D'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()
    labels_ines     = gdal.Open((S1_RESULT_DIR/'ines'/f'{S1_name}_labels.img').as_posix()).ReadAsArray()

    # load all channels
    HH_ML_1x1   = gdal.Open((S1_FEAT_DIR/'ML_1x1'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_ML_1x1   = gdal.Open((S1_FEAT_DIR/'ML_1x1'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    HH_ML_9x9   = gdal.Open((S1_FEAT_DIR/'ML_9x9'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_ML_9x9   = gdal.Open((S1_FEAT_DIR/'ML_9x9'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    HH_ML_21x21 = gdal.Open((S1_FEAT_DIR/'ML_21x21'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_ML_21x21 = gdal.Open((S1_FEAT_DIR/'ML_21x21'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    HH_MuLoG    = gdal.Open((S1_FEAT_DIR/'MuLoG'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_MuLoG    = gdal.Open((S1_FEAT_DIR/'MuLoG'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    HH_SARBM3D  = gdal.Open((S1_FEAT_DIR/'SARBM3D'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_SARBM3D  = gdal.Open((S1_FEAT_DIR/'SARBM3D'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()
    HH_ines     = gdal.Open((S1_FEAT_DIR/'ines'/f'{S1_name}'/'Sigma0_HH_db.img').as_posix()).ReadAsArray()
    HV_ines     = gdal.Open((S1_FEAT_DIR/'ines'/f'{S1_name}'/'Sigma0_HV_db.img').as_posix()).ReadAsArray()

    # load valid mask
    valid_mask = gdal.Open((S1_FEAT_DIR/f'{S1_name}'/'valid.img').as_posix()).ReadAsArray()

# ------------------------------------------- #

    labels_ML_1x1[valid_mask==0] = 0
    labels_ML_9x9[valid_mask==0] = 0
    labels_ML_21x21[valid_mask==0] = 0
    labels_MuLoG[valid_mask==0] = 0
    labels_SARBM3D[valid_mask==0] = 0
    labels_ines[valid_mask==0] = 0

# ------------------------------------------- #

    logger.info('Saving figures...')

    logger.info('figure: ML_1x1')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_ML_1x1, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_ML_1x1, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_ML_1x1, interpolation='nearest')
    axes[0].set_title('HH, ML_1x1')
    axes[1].set_title('HV, ML_1x1')
    axes[2].set_title('labels, ML_1x1')
    plt.savefig((FIG_DIR / f'{S1_name}_ML_1x1.png').as_posix(), dpi=300)

    logger.info('figure: ML_9x9')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_ML_9x9, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_ML_9x9, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_ML_9x9, interpolation='nearest')
    axes[0].set_title('HH, ML_9x9')
    axes[1].set_title('HV, ML_9x9')
    axes[2].set_title('labels, ML_9x9')
    plt.savefig((FIG_DIR / f'{S1_name}_ML_9x9.png').as_posix(), dpi=300)

    logger.info('figure: ML_21x21')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_ML_21x21, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_ML_21x21, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_ML_21x21, interpolation='nearest')
    axes[0].set_title('HH, ML_21x21')
    axes[1].set_title('HV, ML_21x21')
    axes[2].set_title('labels, ML_21x21')
    plt.savefig((FIG_DIR / f'{S1_name}_ML_21x21.png').as_posix(), dpi=300)

    logger.info('figure: MuLoG')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_MuLoG, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_MuLoG, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_MuLoG, interpolation='nearest')
    axes[0].set_title('HH, MuLoG')
    axes[1].set_title('HV, MuLoG')
    axes[2].set_title('labels, MuLoG')
    plt.savefig((FIG_DIR / f'{S1_name}_MuLoG.png').as_posix(), dpi=300)

    logger.info('figure: SARBM3D')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_SARBM3D, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_SARBM3D, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_SARBM3D, interpolation='nearest')
    axes[0].set_title('HH, SARBM3D')
    axes[1].set_title('HV, SARBM3D')
    axes[2].set_title('labels, SARBM3D')
    plt.savefig((FIG_DIR / f'{S1_name}_SARBM3D.png').as_posix(), dpi=300)

    logger.info('figure: ines')
    fig, axes = plt.subplots(1,3,sharex=True,sharey=True,figsize=((12,5)))
    axes = axes.ravel()
    axes[0].imshow(HH_ines, vmin=vmin_HH, vmax=vmax_HH, cmap='gray')
    axes[1].imshow(HV_ines, vmin=vmin_HV, vmax=vmax_HV, cmap='gray')
    axes[2].imshow(labels_ines, interpolation='nearest')
    axes[0].set_title('HH, ines')
    axes[1].set_title('HV, ines')
    axes[2].set_title('labels, ines')
    plt.savefig((FIG_DIR / f'{S1_name}_ines.png').as_posix(), dpi=300)

    logger.info('figure: all results')
    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,9)))
    axes = axes.ravel()
    axes[0].imshow(labels_ML_1x1, interpolation='nearest')
    axes[1].imshow(labels_ML_9x9, interpolation='nearest')
    axes[2].imshow(labels_ML_21x21, interpolation='nearest')
    axes[3].imshow(labels_MuLoG, interpolation='nearest')
    axes[4].imshow(labels_SARBM3D, interpolation='nearest')
    axes[5].imshow(labels_ines, interpolation='nearest')
    axes[0].set_title('labels, ML_1x1')
    axes[1].set_title('labels, ML_1x1')
    axes[2].set_title('labels, ML_1x1')
    axes[3].set_title('labels, MuLoG')
    axes[4].set_title('labels, SARBM3D')
    axes[5].set_title('labels, ines')
    plt.savefig((FIG_DIR / f'{S1_name}_all_labels.png').as_posix(), dpi=300)

    plt.close('all')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <show_results_in_radar_geometry.py> ----
