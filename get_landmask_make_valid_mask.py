# ---- This is <get_landmask_make_valid_mask.py> ----

"""
Extract landmask and make valid_mask.
"""

import os
import pathlib

from loguru import logger

import numpy as np
from osgeo import gdal

import geocoding.landmask as geo_lm

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

overwrite = False

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

    # build general feat folder
    general_feat_folder = S1_FEAT_DIR / f'{S1_name}'

    # build lat, lon, and landmask paths
    lat_path = general_feat_folder / 'lat.img'
    lon_path = general_feat_folder / 'lon.img'
    shapefile_path = osm_landmask_path
    output_path = general_feat_folder / 'landmask.img'

    # convert landmask to SAR geometry for current image
    geo_lm.convert_osm_landmask_2_SAR_geometry(
        lat_path,
        lon_path,
        shapefile_path,
        output_path,
        tie_points = 21,
        overwrite = overwrite,
        loglevel = 'INFO'
    )

# ------------------------------------------- #

    # combine swath_mask and landmask to valid_mask

    # define valid mask output path
    valid_mask_path = general_feat_folder / 'valid.img'

    if valid_mask_path.is_file() and not overwrite:
        logger.info('valid mask already exists')
        logger.info(f'Finished safe_folder: {s}\n')
        continue

    # read swath mask and landmask
    sm = gdal.Open((general_feat_folder/'swath_mask.img').as_posix()).ReadAsArray()
    lm = gdal.Open((general_feat_folder/'landmask.img').as_posix()).ReadAsArray()

    # initialize valid_mask
    valid_mask = np.ones(lm.shape)

    # exclude land and boundaries from valid mask
    valid_mask[lm==0] = 0
    valid_mask[sm==0] = 0


    # write valid_mask to file
    logger.info('Writing valid_mask to file')
    Ny, Nx = valid_mask.shape
    out = gdal.GetDriverByName('Envi').Create(valid_mask_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
    out.GetRasterBand(1).WriteArray(valid_mask)
    out.FlushCache
    del out

    logger.info(f'Finished safe_folder: {s}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <get_landmask_make_valid_mask.py> ----
