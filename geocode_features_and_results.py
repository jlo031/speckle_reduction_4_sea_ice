# ---- This is <geocode_features_and_results.py> ----

"""
Geocode features and results for all speckle reduction methods.
"""

import os
import pathlib

from loguru import logger

import geocoding.S1_geocoding as geo_S1

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

processing_methods = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'baseline', 'proposed']
target_epsg = 3996
pixel_spacing = 40

overwrite = False

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get list of all S1 basenames
S1_list = [ f.split('.SAFE')[0] for f in os.listdir(S1_L1_DIR) if f.endswith('SAFE') ]

# ------------------------------------------- #

# loop over all S1 images
for S1_name in S1_list:

    logger.info(f'Processing S1 image: {S1_name}\n')

    # build full path to safe folder
    safe_folder = S1_L1_DIR / f'{S1_name}.SAFE'

    # loop through all processing methods
    for processing_method in processing_methods:

        logger.info(f'Geocoding processing method: {processing_method}')


        # LABELS

        # build path to classification result
        img_path = S1_RESULT_DIR / processing_method / f'{S1_name}_labels.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{processing_method}_labels_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode classification result
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata = 0,
            dstnodata = 0,
            order = 3,
            resampling = 'near',
            keep_gcp_file = False,
            overwrite = overwrite,
            loglevel = 'INFO',
        )



        # HH

        # build path to HH folder
        img_path = S1_FEAT_DIR / processing_method / f'{S1_name}' / 'Sigma0_HH_db.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{processing_method}_HH_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode HH
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata = 0,
            dstnodata = 0,
            order = 3,
            resampling = 'near',
            keep_gcp_file = False,
            overwrite = overwrite,
            loglevel = 'INFO',
        )


        # HV

        # build path to HV folder
        img_path = S1_FEAT_DIR / processing_method / f'{S1_name}' / 'Sigma0_HV_db.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{processing_method}_HV_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode HV
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata = 0,
            dstnodata = 0,
            order = 3,
            resampling = 'near',
            keep_gcp_file = False,
            overwrite = overwrite,
            loglevel = 'INFO',
        )

        logger.info(f'Finished processing method: {processing_method}\n')

    logger.info(f'Finished S1 image: {S1_name}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <geocode_features_and_results.py> ----
