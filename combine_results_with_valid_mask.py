# ---- This is <combine_results_with_valid_mask.py> ----

"""
Mask out non-valid pixels in classification results.
"""

import os
import pathlib

from loguru import logger

from osgeo import gdal

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

subfolder_list = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'ines']

from folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all S1 basenames
S1_list = [ f.split('.SAFE')[0] for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all S1 images
for S1_name in S1_list:

    logger.info(f'Processing S1 image: {S1_name}')

    # build full path to safe folder
    safe_folder = S1_L1_DIR / f'{S1_name}.SAFE'

    for subfolder in subfolder_list:

        logger.info(f'Processing subfolder: {subfolder}')

        # build path to classification result
        labels_path = S1_RESULT_DIR / subfolder / f'{S1_name}_labels.img'

        # build path to valid_mask
        valid_path = S1_FEAT_DIR / f'{S1_name}' / 'valid.img'

        # read image data
        labels     = gdal.Open(labels_path.as_posix()).ReadAsArray()
        valid_mask = gdal.Open(valid_path.as_posix()).ReadAsArray()

        # mask out non-valid pixels
        labels[valid_mask==0] = 0

        # build path to output file
        output_path = S1_RESULT_DIR / subfolder / f'{S1_name}_labels_valid.img'

        # write valid_mask to file
        logger.info('Writing valid labels to file')
        Ny, Nx = valid_mask.shape
        out = gdal.GetDriverByName('Envi').Create(output_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
        out.GetRasterBand(1).WriteArray(labels)
        out.FlushCache
        del out

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <combine_results_with_valid_mask.py> ----
