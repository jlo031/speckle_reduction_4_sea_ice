# ---- This is <merge_and_crop_orbits.py> ----

"""
Merge images and results from the same S1 orbits.
"""

import os
import pathlib

from loguru import logger

import geocoding.S1_geocoding as geo_S1

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

overwrite = False

subfolder_list = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'ines']

orbit_list = ['043029_05233F', '043044_0523D1']

from folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get complete list of geocoeded tiff files
file_list = [ f for f in os.listdir(S1_GEO_DIR) if f.endswith('tiff') ]

# ------------------------------------------- #

# loop over allorbits
for orbit in orbit_list:

    logger.info(f'Processing orbit: {orbit}')

    # build folder to current orbit
    orbit_folder = S1_ORBIT_DIR / f'{orbit}'
    orbit_folder.mkdir(parents=True, exist_ok=True)

    # get list of gepcoded tiff files for current orbit
    orbit_file_list = [ f for f in file_list if f'{orbit}' in f ]

# ------------------------------------------- #

    # loop over all subfolders
    for subfolder in subfolder_list:

        logger.info(f'Processing subfolder: {subfolder}')

        # get list of geocoded tiff files for current orbit and subfolder
        current_file_list = [ f for f in orbit_file_list if f'{subfolder}' in f ]
        current_file_list.sort()

        # get lists of geocoded tiff files for current orbit, subfolder, and feature
        HH_files = [ f for f in current_file_list if 'HH' in f ]
        HV_files = [ f for f in current_file_list if 'HV' in f ]
        labels_files = [ f for f in current_file_list if 'labels' in f and not 'valid' in f ]
        labels_valid_files = [ f for f in current_file_list if 'valid' in f ]

# ------------------------------------------- #

        if len(HH_files) > 1:

            # build path to merged output tiff file
            merged_output_tiff_path = orbit_folder / f'Sigma0_HH_db_{subfolder}.tiff'

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Current output file already exists')
                continue

            else:

                # combine tiff paths for current orbit to string
                files_2_merge = []
                for ff in HH_files:
                    files_2_merge.append((S1_GEO_DIR/ff).as_posix())
                files_2_merge_string = " ".join(files_2_merge)

                # merge tiff files
                cmd_gdal_merge = f'gdal_merge.py -o {merged_output_tiff_path.as_posix()} -of gtiff -n 0 {files_2_merge_string}'

                logger.debug(f'Exectuting: {cmd_gdal_merge}')
                os.system(cmd_gdal_merge)

# ------------------------------------------- #

        if len(HV_files) > 1:

            # build path to merged output tiff file
            merged_output_tiff_path = orbit_folder / f'Sigma0_HV_db_{subfolder}.tiff'

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Current output file already exists')
                continue

            else:

                # combine tiff paths for current orbit to string
                files_2_merge = []
                for ff in HV_files:
                    files_2_merge.append((S1_GEO_DIR/ff).as_posix())
                files_2_merge_string = " ".join(files_2_merge)

                # merge tiff files
                cmd_gdal_merge = f'gdal_merge.py -o {merged_output_tiff_path.as_posix()} -of gtiff -n 0 {files_2_merge_string}'

                logger.debug(f'Exectuting: {cmd_gdal_merge}')
                os.system(cmd_gdal_merge)

# ------------------------------------------- #

        if len(labels_files) > 1:

            # build path to merged output tiff file
            merged_output_tiff_path = orbit_folder / f'labels_{subfolder}.tiff'

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Current output file already exists')
                continue

            else:

                # combine tiff paths for current orbit to string
                files_2_merge = []
                for ff in labels_files:
                    files_2_merge.append((S1_GEO_DIR/ff).as_posix())
                files_2_merge_string = " ".join(files_2_merge)

                # merge tiff files
                cmd_gdal_merge = f'gdal_merge.py -o {merged_output_tiff_path.as_posix()} -of gtiff -n 0 {files_2_merge_string}'

                logger.debug(f'Exectuting: {cmd_gdal_merge}')
                os.system(cmd_gdal_merge)

# ------------------------------------------- #

        if len(labels_valid_files) > 1:

            # build path to merged output tiff file
            merged_output_tiff_path = orbit_folder / f'labels_valid_{subfolder}.tiff'

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Current output file already exists')
                continue

            else:

                # combine tiff paths for current orbit to string
                files_2_merge = []
                for ff in labels_valid_files:
                    files_2_merge.append((S1_GEO_DIR/ff).as_posix())
                files_2_merge_string = " ".join(files_2_merge)

                # merge tiff files
                cmd_gdal_merge = f'gdal_merge.py -o {merged_output_tiff_path.as_posix()} -of gtiff -n 0 {files_2_merge_string}'

                logger.debug(f'Exectuting: {cmd_gdal_merge}')
                os.system(cmd_gdal_merge)


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <merge_and_crop_orbits.py> ----
