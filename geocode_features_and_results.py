# ---- This is <geocode_features_and_results.py> ----

"""
Geocode features and results for all speckle reduction methods.
"""

import os
import pathlib

from loguru import logger

import geocoding.S1_geocoding as geo_S1

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

subfolder_list = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'ines']

target_epsg = 3996
pixel_spacing = 40

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


        # LABELS

        # build path to classification result
        img_path = S1_RESULT_DIR / subfolder / f'{S1_name}_labels.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{subfolder}_labels_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode classification result
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata=0,
            dstnodata=0,
            order=3,
            resampling='near',
            keep_gcp_file=False,
            overwrite=False,
            loglevel='INFO',
        )



        # VALID LABELS

        # build path to classification result
        img_path = S1_RESULT_DIR / subfolder / f'{S1_name}_labels_valid.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{subfolder}_labels_valid_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode classification result
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata=0,
            dstnodata=0,
            order=3,
            resampling='near',
            keep_gcp_file=False,
            overwrite=False,
            loglevel='INFO',
        )



        # HH

        # build path to HH folder
        img_path = S1_FEAT_DIR / subfolder / f'{S1_name}' / 'Sigma0_HH_db.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{subfolder}_HH_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode HH
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata=0,
            dstnodata=0,
            order=3,
            resampling='near',
            keep_gcp_file=False,
            overwrite=False,
            loglevel='INFO',
        )


        # HV

        # build path to HV folder
        img_path = S1_FEAT_DIR / subfolder / f'{S1_name}' / 'Sigma0_HV_db.img'

        # build path to output tiff file
        output_tiff_path = S1_GEO_DIR / f'{S1_name}_{subfolder}_HV_epsg{target_epsg}_pixelspacing{pixel_spacing}.tiff'

        # geocode HV
        geo_S1.geocode_S1_image_from_safe_gcps(
            img_path,
            safe_folder,
            output_tiff_path,
            target_epsg,
            pixel_spacing,
            srcnodata=0,
            dstnodata=0,
            order=3,
            resampling='near',
            keep_gcp_file=False,
            overwrite=False,
            loglevel='DEBUG',
        )

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <geocode_features_and_results.py> ----
