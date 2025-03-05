# ---- This is <prepare_speckle_reduced_data.py> ----

"""
Prepare data from Ines and Loic for input to ice_type_classification module.
"""

import os
import pathlib

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

overwrite    = False
make_figures = False

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all tiff files from Loic
tiff_files = [ f for f in os.listdir(S1_LOIC_DIR) if f.endswith('tiff') ]

# ------------------------------------------- #

# loop over all tiff files
for tiff_file in tiff_files:

    logger.info(f'Processing tiff_file: {tiff_file}')

    tiff_path = S1_LOIC_DIR / tiff_file


    # read tiff file
    img = gdal.Open(tiff_path.as_posix()).ReadAsArray()

    # get image dimensions
    bands, Ny, Nx = img.shape

    # get HH and HV channel
    HH = img[0,:,:]
    HV = img[1,:,:]

# ------------------------------------------- #

    # some zeros from after the noise correction are problematic for dB conversion

    # find the minimum non-zero value for HH and HV and set all zero values to the min
    HH_non_zero_min = np.nanmin(HH[np.nonzero(HH)])
    HV_non_zero_min = np.nanmin(HV[np.nonzero(HV)])

    # and set all zero values to the min
    HH_non_zero = HH
    HH_non_zero[HH_non_zero==0] = HH_non_zero_min
    HV_non_zero = HV
    HV_non_zero[HV_non_zero==0] = HV_non_zero_min

# ------------------------------------------- #

    # convert original values to dB
    HH_dB = 10*np.log10(HH)
    HV_dB = 10*np.log10(HV)

    # convert non-zero to dB
    HH_non_zero_dB = 10*np.log10(HH_non_zero)
    HV_non_zero_dB = 10*np.log10(HV_non_zero)

# ------------------------------------------- #

    # get speckle reduction method and S1 basename from tiff file name
    speckle_reduction_method = pathlib.Path(tiff_file).stem.split('_')[-1]
    S1_base = '_'.join(pathlib.Path(tiff_file).stem.split('_')[0:9])

    # build path to feature folder for current speckle reduction method
    feature_folder = S1_FEAT_DIR / f'{speckle_reduction_method}' / f'{S1_base}'

    logger.info(f'speckle_reduction_method: {speckle_reduction_method}')
    logger.info(f'S1_base:                  {S1_base}')
    logger.info(f'feature_folder:           {feature_folder}')

    # create feature folder if needed
    feature_folder.mkdir(parents=True, exist_ok=True)

# ------------------------------------------- #

    # write HH and HV in dB to feature folder

    HH_output_path = feature_folder / 'Sigma0_HH_db.img'
    HV_output_path = feature_folder / 'Sigma0_HV_db.img'
 
    if HH_output_path.is_file() and not overwrite:
        logger.info('HH output file already exists')
    else:
        logger.info('Writing HH to file')
        out = gdal.GetDriverByName('Envi').Create(HH_output_path.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
        out.GetRasterBand(1).WriteArray(HH_non_zero_dB)
        out.FlushCache
        del out

    if HV_output_path.is_file() and not overwrite:
        logger.info('HV output file already exists')
    else:
        logger.info('Writing HV to file')
        out = gdal.GetDriverByName('Envi').Create(HV_output_path.as_posix(), Nx, Ny, 1, gdal.GDT_Float32)
        out.GetRasterBand(1).WriteArray(HV_non_zero_dB)
        out.FlushCache
        del out

# ------------------------------------------- #

    if make_figures:

        sub = 3
        HH_min= 0
        HH_max= 0.1
        HV_min= 0
        HV_max= 0.1
        HH_dB_min= -35
        HH_dB_max= 0
        HV_dB_min= -40
        HV_dB_max= -5

        fig, axes = plt.subplots(3,2,sharex=True,sharey=True)
        axes = axes.ravel()
        axes[0].imshow(HH[::sub,::sub], vmin=HH_min, vmax=HH_max, interpolation='nearest')
        axes[1].imshow(HV[::sub,::sub], vmin=HV_min, vmax=HV_max, interpolation='nearest')
        axes[2].imshow(HH_dB[::sub,::sub], vmin=HH_dB_min, vmax=HH_dB_max, interpolation='nearest')
        axes[3].imshow(HV_dB[::sub,::sub], vmin=HV_dB_min, vmax=HV_dB_max, interpolation='nearest')
        axes[4].imshow(HH_non_zero_dB[::sub,::sub], vmin=HH_dB_min, vmax=HH_dB_max, interpolation='nearest')
        axes[5].imshow(HV_non_zero_dB[::sub,::sub], vmin=HV_dB_min, vmax=HV_dB_max, interpolation='nearest')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <prepare_speckle_reduced_data.py> ----
