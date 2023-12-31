# ---- This is <extract_features.py> ----

"""
Extract HH, HV, ad IA for ice type classification with different ML settings.
"""

import os
import pathlib

from loguru import logger

import S1_processing.S1_feature_extraction as S1_feat

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

ML_list = ['1x1', '9x9', '21x21']

from folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all safe folders
safe_folder_list = [ f for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all safe folders
for s in safe_folder_list:

    logger.info(f'Processing safe_folder: {s}')

    # get S1 basename
    S1_name = pathlib.Path(s).stem

    # build full path to safe folder
    safe_folder = S1_L1_DIR / s

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
                ML=ML,
                dB=True,
                loglevel='INFO'
            )

        # incident angle
        S1_feat.get_S1_IA(
            safe_folder,
            feat_folder,
            loglevel='INFO'
        )

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <extract_features.py> ----
