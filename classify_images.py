# ---- This is <classify_images.py> ----

"""
Classify images using different speckle reduction methods
"""

import os
import pathlib
import shutil

from loguru import logger

import ice_type_classification.classification as cl

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

S1_DIR = pathlib.Path('/Data/speckle_reduction_tests/Sentinel-1')
S1_L1_DIR = S1_DIR / 'L1'
S1_RESULT_DIR = S1_DIR / 'classification_results'

clf_model_path = pathlib.Path(
    '/home/jo/work/ice_type_classification/src/ice_type_classification/clf_models/belgica_bank_ice_types.pickle'
)

# list of speckle reduction methods to consider separately
subfolder_list = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'ines']

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all S1 basenames
S1_list = [ f.split('.SAFE')[0] for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all S1 images
for S1_name in S1_list:

    logger.info(f'Processing S1 image: {S1_name}')

    # loop through all processing methods
    for subfolder in subfolder_list:

        logger.info(f'Processing speckle reduction: {subfolder}')
 
        # build paths to feature and results folder
        feat_folder   = S1_DIR / 'features' / f'{subfolder}' / f'{S1_name}'
        result_folder = S1_RESULT_DIR / f'{subfolder}'

        if not feat_folder.is_dir():
            logger.error(f'Current feature folder does not exists: {feat_folder}')
            continue

        # list features
        feature_list = [ f for f in os.listdir(feat_folder) if f.endswith('.img') ]

        logger.debug(f'feat_folder:   {feat_folder}')
        logger.debug(f'result_folder: {result_folder}')
        logger.debug(f'feature_list:  {feature_list}')

        # copy IA images into feature folder if needed
        if subfolder in ['MuLoG', 'SARBM3D', 'ines'] and 'IA.img' not in feature_list:
            logger.info('Copying IA image into local feature folder')
            shutil.copyfile(S1_DIR/'features'/'ML_1x1'/f'{S1_name}'/'IA.img' , feat_folder/'IA.img')
            shutil.copyfile(S1_DIR/'features'/'ML_1x1'/f'{S1_name}'/'IA.hdr' , feat_folder/'IA.hdr')

        cl.classify_S1_image_from_feature_folder(
            feat_folder.as_posix(),
            result_folder.as_posix(),
            clf_model_path.as_posix(),
            valid_mask = False,
            block_size = 1000000.0,
            overwrite = False,
            loglevel = 'INFO',
        )


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classify_images.py> ----
