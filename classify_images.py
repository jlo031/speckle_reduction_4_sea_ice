# ---- This is <classify_images.py> ----

"""
Classify images using different speckle reduction methods.
"""

import os
import pathlib
import shutil

from loguru import logger

import GLIA_classifier.classification as glass

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

clf_model_path = pathlib.Path('config/belgica_bank_ice_types_2022.pickle').resolve()

procesing_methods = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'baseline', 'proposed']

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all S1 basenames
S1_list = [ f.split('.SAFE')[0] for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all S1 images
for S1_name in S1_list:

    logger.info(f'Processing S1 image: {S1_name}')

    # loop through all processing methods
    for procesing_method in procesing_methods:

        logger.info(f'Processing speckle reduction: {procesing_method}')
 
        # build paths to feature and results folder
        feat_folder   = S1_DIR / 'features' / f'{procesing_method}' / f'{S1_name}'
        result_folder = S1_RESULT_DIR / f'{procesing_method}'

        if not feat_folder.is_dir():
            logger.error(f'Current feature folder does not exists: {feat_folder}')
            continue

        # list features
        feature_list = [ f for f in os.listdir(feat_folder) if f.endswith('.img') ]

        logger.debug(f'feat_folder:   {feat_folder}')
        logger.debug(f'result_folder: {result_folder}')
        logger.debug(f'feature_list:  {feature_list}')

        # copy IA images into feature folder if needed
        if procesing_method in ['MuLoG', 'SARBM3D', 'baseline', 'proposed'] and 'IA.img' not in feature_list:
            logger.info('Copying IA image into local feature folder')
            shutil.copyfile(S1_DIR/'features'/'ML_1x1'/f'{S1_name}'/'IA.img' , feat_folder/'IA.img')
            shutil.copyfile(S1_DIR/'features'/'ML_1x1'/f'{S1_name}'/'IA.hdr' , feat_folder/'IA.hdr')

        # copy valid_mask into feature folder if needed
        if 'valid.img' not in feature_list:
            logger.info('Copying valid mask image into local feature folder')
            shutil.copyfile(S1_DIR / 'features' / f'{S1_name}' / 'valid.img' , feat_folder/'valid.img')
            shutil.copyfile(S1_DIR / 'features' / f'{S1_name}' / 'valid.hdr' , feat_folder/'valid.hdr')

# ------------------------------------------- #

        glass.classify_image_from_feature_folder(
            feat_folder,
            result_folder,
            clf_model_path,
            use_valid_mask = True,
            estimate_uncertainties = False,
            uncertainty_params_dict = [],
            overwrite = False,
            loglevel = 'INFO',
        )

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <classify_images.py> ----
