# ---- This is <convert_json_2_training_masks.py> ----

"""
Loop through all json files with labeled ROIs.
Convert json files to training masks.
""" 

import os
import sys
import pathlib
from loguru import logger

import labelme_utils.json_conversion as lm_json

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# overwrite already processed files
overwrite = True

# path to labels.txt file
labels_path = './config/labels.txt'

# output file format (ENVI or GTIFF)
output_format = 'ENVI'

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# find all json files in ALIGNED_CROP_SCALED_DIR
json_files = [ f for f in os.listdir(S1_RGB_DIR) if f.endswith('.json') ]
json_files.sort()

# -------------------------------------------------------------------------- #

# loop over all json files

for ii, json_file in enumerate(json_files):

    logger.info(f'Processing json_file {ii+1}/{len(json_files)}: {json_file}')

    # build full json_path
    json_path = S1_RGB_DIR / json_file

    # create the training mask images
    lm_json.convert_json_file_2_mask(json_path, labels_path, S1_VAL_DIR, output_format=output_format, overwrite=overwrite)

    logger.info(f'Finished json_file {ii+1}/{len(json_files)}: {json_file}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <convert_json_2_training_masks.py> ----
