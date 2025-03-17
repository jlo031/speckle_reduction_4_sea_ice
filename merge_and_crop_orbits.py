# ---- This is <merge_and_crop_orbits.py> ----

"""
Merge images and results from the same S1 orbits.
"""

import os
import pathlib

from loguru import logger

import geocoding.generic_geocoding as geo_gen
import geocoding.S1_geocoding as geo_S1

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# overwrite already processed files
overwrite = False

# logelevel
loglevel = 'INFO'

# pre-processing methods for classification
processing_methods = [ 'ML_1x1', 'ML_9x9', 'ML_21x21', 'MuLoG', 'SARBM3D', 'baseline', 'proposed']

# S1 orbits to process
orbit_list = ['043029_05233F', '043044_0523D1']

# AOI coordinates for cropping
crop_AOI = [-400000, 0, -1300000, -900000]

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# get complete list of geocoeded tiff files
file_list = [ f for f in os.listdir(S1_GEO_DIR) if f.endswith('tiff') ]

# initialize list of merged geocoeded tiff files per orbit to be cropped later
files_2_crop_list = []

# ------------------------------------------- #

# loop over all orbits
for orbit in orbit_list:

    logger.info(f'Processing orbit: {orbit}\n')

    # build folder to current orbit
    orbit_folder = S1_ORBIT_DIR / f'{orbit}'
    orbit_folder.mkdir(parents=True, exist_ok=True)

    # build folder to AOI for current orbit
    orbit_AOI_folder = S1_ORBIT_DIR / f'{orbit}' / f'AOIs'
    orbit_AOI_folder.mkdir(parents=True, exist_ok=True)

    # get list of gepcoded tiff files for current orbit
    orbit_file_list = [ f for f in file_list if f'{orbit}' in f ]

# ------------------------------------------- #

    # loop through all processing methods
    for processing_method in processing_methods:

        logger.info(f'Working on processing method: {processing_method}')

        # get list of geocoded tiff files for current orbit and processing_method
        current_file_list = [ f for f in orbit_file_list if f'{processing_method}' in f ]
        current_file_list.sort()

        # get lists of geocoded tiff files for current orbit, processing_method, and feature
        HH_files = [ f for f in current_file_list if 'HH' in f ]
        HV_files = [ f for f in current_file_list if 'HV' in f ]
        labels_files = [ f for f in current_file_list if 'labels' in f and not 'valid' in f ]

# ------------------------------------------- #

        if len(HH_files) > 1:

            # build path to merged output tiff file
            merged_output_tiff_path = orbit_folder / f'Sigma0_HH_db_{processing_method}.tiff'

            # append to list of files that should be cropped later
            files_2_crop_list.append(merged_output_tiff_path)

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Merged HH file already exists')

            else:
                logger.info('Merging HH files')

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
            merged_output_tiff_path = orbit_folder / f'Sigma0_HV_db_{processing_method}.tiff'

            # append to list of files that should be cropped later
            files_2_crop_list.append(merged_output_tiff_path)

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Merged HV file already exists')

            else:
                logger.info('Merging HV files')

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
            merged_output_tiff_path = orbit_folder / f'labels_{processing_method}.tiff'

            # append to list of files that should be cropped later
            files_2_crop_list.append(merged_output_tiff_path)

            if merged_output_tiff_path.is_file() and not overwrite:
                logger.info('Merged labels file already exists')

            else:
                logger.info('Merging label files')

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

        # stack HH and HV (for false-color visualization in QGIS)

        # build output path to HH HV stack
        HH_HV_output_path = orbit_folder / f'HH_HV_{processing_method}.tiff'

        # append to list of files that should be cropped later
        files_2_crop_list.append(HH_HV_output_path)

        if HH_HV_output_path.is_file() and not overwrite:
            logger.info('Stacked intensity file (HH HV orbit stack) already exists')

        else:
            logger.info('Stacking intensity files (HH HV orbit stack) already exists')

            input_tif_path1 = orbit_folder / f'Sigma0_HH_db_{processing_method}.tiff'
            input_tif_path2 = orbit_folder / f'Sigma0_HV_db_{processing_method}.tiff'

            geo_gen.geo_utils.stack_geocoded_images(
                input_tif_path1,
                input_tif_path2,
                HH_HV_output_path,
                overwrite = overwrite,
                loglevel = loglevel,
            )


        logger.info(f'Finished processing method: {processing_method}\n')

    logger.info(f'Finished orbit: {orbit}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# loop over all geocoded files_2_crop and crop to AOI

logger.info(f'Looping over files_2_crop_list with {len(files_2_crop_list)} entries')

for i,file_2_crop in enumerate(files_2_crop_list):

    logger.info(f'File {i+1}: {file_2_crop}')

    # build path to cropped output file
    cropped_output_path = file_2_crop.parent / 'AOIs' / f'{file_2_crop.stem}_crop.tiff'

    logger.info(f'cropped_output_path: {cropped_output_path}')

    if cropped_output_path.is_file() and not overwrite:
        logger.info('Cropped file already exists')
        logger.info(f'Finished file {file_2_crop}\n')

    else:

        logger.info('Cropping file to AOI')

        ulx = crop_AOI[0]
        uly = crop_AOI[3]
        lrx = crop_AOI[1]
        lry = crop_AOI[2]

        cmd_gdal_crop = f'gdal_translate ' + \
            f'-projwin {ulx} {uly} {lrx} {lry} ' + \
            f'{file_2_crop.as_posix()} ' + \
            f'{cropped_output_path}'

        logger.debug(f'Exectuting: {cmd_gdal_crop}')
        os.system(cmd_gdal_crop)

        logger.info(f'Finished file {file_2_crop}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <merge_and_crop_orbits.py> ----
