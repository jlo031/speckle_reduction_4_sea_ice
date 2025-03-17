# ---- This is <extract_features.py> ----

"""
Extract HH, HV, and IA for ice type classification with different ML settings.
"""

import os
import pathlib

from loguru import logger

import S1_processing.S1_feature_extraction as S1_feat

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# list of ML processing levels
ML_list = ['1x1', '9x9', '21x21']

# overwrite already processed files
overwrite = False

# logelevel
loglevel = 'INFO'

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all safe folders
safe_folder_list = [ f for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all safe folders
for s in safe_folder_list:

    logger.info(f'Processing safe_folder: {s}\n')

    # get S1 basename
    S1_name = pathlib.Path(s).stem

    # build full path to safe folder
    safe_folder = S1_L1_DIR / s

# ------------------------------------------- #

    # build general feat folder
    general_feat_folder = S1_FEAT_DIR / f'{S1_name}'

    # get swath mask
    S1_feat.get_S1_swath_mask(
        safe_folder,
        general_feat_folder,
        overwrite = overwrite,
        loglevel = loglevel
    )

    # get lat/lon for current image
    S1_feat.get_S1_lat_lon(
        safe_folder,
        general_feat_folder,
        overwrite = overwrite,
        loglevel = loglevel
    )

# ------------------------------------------- #

    for ML in ML_list:

        logger.info(f'Processing ML: {ML}')

        # build the path to output feature folder
        feat_folder = S1_FEAT_DIR / f'ML_{ML}' / f'{S1_name}'

        logger.info(f'safe_folder: {safe_folder}')
        logger.info(f'feat_folder: {feat_folder}')

        # intensities 
        for intensity in ['HH', 'HV']:
            # intensity in log domain (dB)
            S1_feat.get_S1_intensity(
                safe_folder,
                feat_folder,
                intensity,
                ML = ML,
                dB = True,
                overwrite = overwrite,
                loglevel = loglevel
            )

        # incident angle
        S1_feat.get_S1_IA(
            safe_folder,
            feat_folder,
            overwrite = overwrite,
            loglevel = loglevel
        )

        logger.info(f'Finished ML: {ML}\n')

    logger.info(f'Finished safe_folder: {s}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <extract_features.py> ----
