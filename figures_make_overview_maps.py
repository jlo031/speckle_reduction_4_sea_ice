# ---- This is <figures_make_overview_maps.py> ----

"""
Visualize data and results on map.
"""

import os
import pathlib
import shutil

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from osgeo import gdal

import cartopy
import cartopy.crs as ccrs

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# parameters for RGB scaling
vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

# grid labels
x_grid = [-45, -30, -15, 0, 15, 30]
y_grid = [76, 79, 82, 85]

# S1 orbits to process
orbit_list = ['043029_05233F', '043044_0523D1']

# select pre-processing method
processing_method = 'proposed'

# overwrite already processed files
overwrite = False

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# define list of intensity image paths
geo_image_path_list = [
    S1_GEO_DIR / 'orbits' / '043029_05233F' / f'HH_HV_{processing_method}.tiff',
    S1_GEO_DIR / 'orbits' / '043044_0523D1' / f'HH_HV_{processing_method}.tiff',
]



# loop over all orbits
for orbit in orbit_list:

    logger.info(f'Processing orbit: {orbit}\n')
    logger.info(f'Working on processing method: {processing_method}')
    
# ------------------------------------------- #

    # build folder to current orbit and tiff file
    geo_image_path = S1_ORBIT_DIR / orbit / f'HH_HV_{processing_method}.tiff'

    # build path to output figure
    output_figure_path = FIG_DIR / f'overview_map_full_orbit_{orbit}_method_{processing_method}.pdf'

    if output_figure_path.is_file() and not overwrite:
        logger.info('Figure for current orbit and processing method already exists')
        logger.info(f'Finished processing method: {processing_method}\n')
        logger.info(f'Finished orbit: {orbit}\n')
        continue

# ------------------------------------------- #

    # read intensities and combine to RGB image

    logger.info('Reading and scaling intensities')

    intensities = gdal.Open((geo_image_path).as_posix()).ReadAsArray().transpose(1,2,0)

    HH = intensities[:,:,0]
    HV = intensities[:,:,1]

    # set no data values to nan (they are zero in the geocoded intensity image)
    HH[HH==0] = np.nan
    HV[HV==0] = np.nan

    # if you want to stack to false-color RGB, set new min/max values
    new_min = 0
    new_max = 1

    # scale both channels to [new_min,new_max] and clip values below and above
    # linear map from sigma0 in dB to new_min and new_max
    HH_scaled  = (HH - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
    HV_scaled  = (HV - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
    HH_scaled  = np.clip(HH_scaled, new_min, new_max)
    HV_scaled  = np.clip(HV_scaled, new_min, new_max)

    # stack to fals-color RGB
    RGB = np.stack((HV_scaled, HH_scaled, HH_scaled),2)

# --------------------------------------------------------------- #

    # PREPARE THE ACTUAL FIGURE

    logger.info(f'Making and saving figure: {output_figure_path}')

    # get image extent 
    gtif = gdal.Open((geo_image_path).as_posix())
    trans = gtif.GetGeoTransform()
    image_extent = (trans[0], trans[0] + gtif.RasterXSize*trans[1], trans[3] + gtif.RasterYSize*trans[5], trans[3])

    projection = ccrs.Stereographic(
        central_latitude = 90,
        central_longitude = 0,
        false_easting = 0,
        false_northing = 0,
        true_scale_latitude = 75,
        globe = None
    )

    # create figure and axes handle
    fig, ax = plt.subplots(1,1,figsize=((10,10)),subplot_kw={'projection': projection})

    # add cartopy features
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.coastlines(color='red',resolution='10m')
    gl = ax.gridlines(color='white',draw_labels=True, y_inline=False)
    gl.top_labels = False
    gl.left_labels = False

    gl.xlocator = mticker.FixedLocator(x_grid)
    gl.ylocator = mticker.FixedLocator(y_grid)

    ax.set_extent([image_extent[0],image_extent[1],image_extent[2],image_extent[3]], crs=projection)

    ax.imshow(RGB, extent=image_extent, transform=projection, zorder=1, alpha=0.75)

    ax.set_facecolor("gray")

    plt.savefig(output_figure_path, dpi=300)


    # clean up current figure
    plt.close('all')


    logger.info(f'Finished processing method: {processing_method}\n')
    logger.info(f'Finished orbit: {orbit}\n')

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <figures_make_overview_maps.py> ----
