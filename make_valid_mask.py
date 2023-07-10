# ---- This is <make_valid_mask.py> ----

"""
Extract lat/lon, swath_mask, and landmask to combine to valid mask.
"""

import os
import pathlib

from loguru import logger

import numpy as np
from osgeo import gdal

import S1_processing.S1_feature_extraction as S1_feat
import geocoding.landmask as geo_lm

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

ML_list = ['1x1', '9x9', '21x21']

DATA_DIR      = pathlib.Path('/Data/speckle_reduction_tests')
S1_DIR        = DATA_DIR / Sentinel-1'
S1_L1_DIR     = S1_DIR / 'L1'
S1_FEAT_DIR   =  S1_DIR / 'features'
S1_RESULT_DIR = S1_DIR / 'classification_results'
S1_LOIC_DIR   = S1_DIR / 'from_loic'
FIG_DIR       = S1_DIR.parent / 'figures'

S1_FEAT_DIR.mkdir(parents=True, exist_ok=True)
S1_RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

osm_landmask_path = S1_DIR.parent / 'shapefiles' / 'land-polygons-split-4326' / 'land_polygons.shp'

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

    # create temporary feat folder
    tmp_feat_folder = S1_FEAT_DIR / f'{S1_name}'

    # get swath mask
    S1_feat.get_S1_swath_mask(
        safe_folder,
        tmp_feat_folder,
        loglevel='INFO'
    )

    # get lat/lon for current image
    S1_feat.get_S1_lat_lon(
        safe_folder,
        tmp_feat_folder,
        overwrite=False,
        dry_run=False,
        loglevel='INFO',
    )

    lat_path = tmp_feat_folder / 'lat.img'
    lon_path = tmp_feat_folder / 'lon.img'
    shapefile_path = osm_landmask_path
    output_path = tmp_feat_folder / 'landmask.img'

    # convert landmask to SAR geometry for current image
    geo_lm.convert_osm_landmask_2_SAR_geometry(
        lat_path,
        lon_path,
        shapefile_path,
        output_path,
        tie_points=21,
        overwrite=False,
        loglevel='INFO',
    )

# ------------------------------------------- #

    # combine swath_mask and landmask to valid_mask

    # read swath mask and landmask
    sm = gdal.Open((tmp_feat_folder/'swath_mask.img').as_posix()).ReadAsArray()
    lm = gdal.Open((tmp_feat_folder/'landmask.img').as_posix()).ReadAsArray()

    # initialize valid_mask
    valid_mask = np.ones(lm.shape)

    # exclude land and boundaries from valid mask
    valid_mask[lm==0] = 0
    valid_mask[sm==0] = 0

    # define valid mask output path
    valid_mask_path = tmp_feat_folder / 'valid.img'

    # write valid_mask to file
    logger.info('Writing valid_mask to file')
    Ny, Nx = valid_mask.shape
    out = gdal.GetDriverByName('Envi').Create(valid_mask_path.as_posix(), Nx, Ny, 1, gdal.GDT_Byte)
    out.GetRasterBand(1).WriteArray(valid_mask)
    out.FlushCache
    del out

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <make_valid_mask.py> ----
