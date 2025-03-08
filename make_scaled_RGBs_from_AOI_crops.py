# ---- This is <make_scaled_RGBs_from_AOI_crops.py> ----

"""
Make RGBs for labelme from AOIs for both orbits
""" 

import os
import sys
import pathlib
from loguru import logger
import numpy as np

from osgeo import gdal

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

loglevel = 'INFO'

# remove default logger handler and add personal one
logger.remove()
logger.add(sys.stderr, level=loglevel)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# define input parameters

# overwrite exisiting scaled RGB image files
overwrite = False

# percentiles for cropping
min_perc = 5
max_perc = 95

# min/max values for scaled image
new_min = 0
new_max = 255

# RGB channels for scaled image
red   = 'HV'
green = 'HH'
blue  = 'HH'

orbit_list = ['043029_05233F', '043044_0523D1']
orbit_list = ['043044_0523D1']

procesing_methods = [ 'ML_1x1', 'baseline', 'proposed']

# cropped image coordinates (must be the same as in "evaluate_results.py")
xmin = 800
xmax = 10000
ymin = 2500
ymax = 10000

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# loop over all orbits
for orbit in orbit_list:

    logger.info(f'Processing orbit: {orbit}\n')

    # build folder to AOI for current orbit
    orbit_AOI_folder = S1_ORBIT_DIR / f'{orbit}' / f'AOIs'

    # get list of gepcoded tiff files for current orbit
    orbit_file_list = [ f for f in os.listdir(orbit_AOI_folder) if f.endswith('tiff') ]

    # loop through all processing methods
    for processing_method in procesing_methods:

        logger.info(f'Working on processing method: {processing_method}')

        # get list of geocoded tiff files for current orbit and processing_method
        current_file_list = [ f for f in orbit_file_list if f'{processing_method}' in f ]
        current_file_list.sort()

        # build path to final RGB output
        rgb_output_path      = S1_RGB_DIR / f'{orbit}_{processing_method}_RGB.tiff'
        rgb_output_path_crop = S1_RGB_DIR / f'{orbit}_{processing_method}_RGB_CROP.tiff'


        if rgb_output_path.is_file() and rgb_output_path_crop.is_file() and not overwrite:
            logger.info('RGB output file already exists')
            logger.info('Finished processing method: {processing_method}\n')
            continue

# ------------------------------------------- #

        logger.info('Reading data')

        # read data
        HH = gdal.Open((orbit_AOI_folder/f'Sigma0_HH_db_{processing_method}_crop.tiff').as_posix()).ReadAsArray()
        HV = gdal.Open((orbit_AOI_folder/f'Sigma0_HV_db_{processing_method}_crop.tiff').as_posix()).ReadAsArray()

# ------------------------------------------- #

        logger.info('Scaling data')

        # find min and max values based on data percentiles
        HH_min = np.nanpercentile(HH,min_perc)
        HH_max = np.nanpercentile(HH,max_perc)
        HV_min = np.nanpercentile(HV,min_perc)
        HV_max = np.nanpercentile(HV,max_perc)

        # or set them manually
        HH_min = -30
        HH_max = -5
        HV_min = -40
        HV_max = -15

        # clip to min and max
        HH[HH<HH_min] = HH_min
        HH[HH>HH_max] = HH_max
        HV[HV<HV_min] = HV_min
        HV[HV>HV_max] = HV_max

        # scale both channels
        HH_scaled = (HH - (HH_min)) * ((new_max - new_min) / ((HH_max) - (HH_min))) + new_min
        HV_scaled = (HV - (HV_min)) * ((new_max - new_min) / ((HV_max) - (HV_min))) + new_min


        # assign to RGB channels
        if red == 'HV':
            r = HV_scaled
        elif red == 'HH':
            r = HH_scaled
        elif red == 'zero':
            r = np.zeros(HH.shape)

        if green == 'HV':
            g = HV_scaled
        elif green == 'HH':
            g = HH_scaled
        elif green == 'zero':
            g = np.zeros(HH.shape)

        if blue == 'HV':
            b = HV_scaled
        elif blue == 'HH':
            b = HH_scaled
        elif blue == 'zero':
            b = np.zeros(HH.shape)


        # stack all channels to one array
        RGB = np.stack((r,g,b),0)

        # round and convert RGB to uint8
        RGB = RGB.astype(np.uint8)

# ------------------------------------------- #

        logger.info('Saving RGB')

        # write to tiff file
        n_bands, Ny, Nx = RGB.shape
        output = gdal.GetDriverByName('GTiff').Create(rgb_output_path.as_posix(), Nx, Ny, n_bands, gdal.GDT_Byte)
        for b in np.arange(n_bands):
            output.GetRasterBand(int(b+1)).WriteArray(RGB[b,:,:])

        output.FlushCache()
        output = None

# ------------------------------------------- #

        # make a cropped version for smaller file sizes in labelme
        RGB_crop = RGB[:,xmin:xmax,ymin:ymax]

        # write to tiff file
        n_bands, Ny, Nx = RGB_crop.shape
        output = gdal.GetDriverByName('GTiff').Create(rgb_output_path_crop.as_posix(), Nx, Ny, n_bands, gdal.GDT_Byte)
        for b in np.arange(n_bands):
            output.GetRasterBand(int(b+1)).WriteArray(RGB_crop[b,:,:])

        output.FlushCache()
        output = None

# ------------------------------------------- #

        logger.info(f'Finished processing method: {processing_method}\n')

    logger.info(f'Finished orbit: {orbit}\n')


# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <make_scaled_RGBs_from_AOI_crops.py> ----
