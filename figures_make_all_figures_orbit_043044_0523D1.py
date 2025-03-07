# ---- This is <figures_make_all_figures_orbit_043044_0523D1.py> ----

"""
Visualize data and results on map.
"""

import os
import pathlib
import shutil
import sys

from loguru import logger

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from osgeo import gdal

import cartopy
import cartopy.crs as ccrs

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

make_AOI_overviews_RGB            = True
make_AOI_overviews_labels         = True
make_closup_label_comparisons     = True

make_colorbar                     = True
make_AOI_overviews_RGB_and_labels = True

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# set parameters for RGBs and plotting

vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

x_grid = [-45, -30, -15, 0, 15, 30]
y_grid = [76, 79, 82, 85]

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# build color map and legend

# get viridis color map (because it looks ok)
CMAP = mpl.cm.get_cmap('viridis')

# initialize normalized class colors with grey background
class_colors_norm = [[0.2, 0.2, 0.2]]

# append 4 equally spaced viridis colors
for idx in np.round(np.linspace(0,255,4)).astype(int):
    class_colors_norm.append(CMAP.colors[idx])

# define level boundaries for colormap
cmap_bounds = np.arange(-0.5,len(class_colors_norm)+0.5,1)
cmap_values = np.convolve(cmap_bounds, np.ones(2)/2, mode='valid').astype(int)

# build colormap
cmap = mpl.colors.ListedColormap(class_colors_norm,name='belgica_bank_cmap')

# build a colormap index based on level boundaries
cmap_norm = mpl.colors.BoundaryNorm(cmap_bounds, cmap.N)

legend_entries = [
    'None',
    'OW / New Ice',
    'Young Ice',
    'Level Ice',
    'Deformed Ice'
]
legend_properties = {'weight':'bold'}

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# load data
logger.info('Loading data ...')

##orbit = '043029_05233F'
orbit = '043044_0523D1'

# load intensities
intensities_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_1x1_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_9x9_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_ML_21x21_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_MuLoG_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_SARBM3D_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_baseline_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)
intensities_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_proposed_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)

# load labels
labels_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_1x1_crop.tiff').as_posix()).ReadAsArray()
labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
labels_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_21x21_crop.tiff').as_posix()).ReadAsArray()
labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_MuLoG_crop.tiff').as_posix()).ReadAsArray()
labels_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_SARBM3D_crop.tiff').as_posix()).ReadAsArray()
labels_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_baseline_crop.tiff').as_posix()).ReadAsArray()
labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# make intensity RGBs
logger.info('Making intensity RGBs ...')

HH_ML_1x1   = intensities_ML_1x1[:,:,0]
HV_ML_1x1   = intensities_ML_1x1[:,:,1]
HH_ML_9x9   = intensities_ML_9x9[:,:,0]
HV_ML_9x9   = intensities_ML_9x9[:,:,1]
HH_ML_21x21 = intensities_ML_21x21[:,:,0]
HV_ML_21x21 = intensities_ML_21x21[:,:,1]
HH_MuLoG    = intensities_MuLoG[:,:,0]
HV_MuLoG    = intensities_MuLoG[:,:,1]
HH_SARBM3D  = intensities_SARBM3D[:,:,0]
HV_SARBM3D  = intensities_SARBM3D[:,:,1]
HH_baseline = intensities_baseline[:,:,0]
HV_baseline = intensities_baseline[:,:,1]
HH_proposed = intensities_proposed[:,:,0]
HV_proposed = intensities_proposed[:,:,1]

HH_ML_1x1[HH_ML_1x1==0] = np.nan
HV_ML_1x1[HV_ML_1x1==0] = np.nan
HH_ML_9x9[HH_ML_9x9==0] = np.nan
HV_ML_9x9[HV_ML_9x9==0] = np.nan
HH_ML_21x21[HH_ML_21x21==0] = np.nan
HV_ML_21x21[HV_ML_21x21==0] = np.nan
HH_MuLoG[HH_MuLoG==0] = np.nan
HV_MuLoG[HV_MuLoG==0] = np.nan
HH_SARBM3D[HH_SARBM3D==0] = np.nan
HV_SARBM3D[HV_SARBM3D==0] = np.nan
HH_baseline[HH_baseline==0] = np.nan
HV_baseline[HV_baseline==0] = np.nan
HH_proposed[HH_proposed==0] = np.nan
HV_proposed[HV_proposed==0] = np.nan

# if you want to stack to false-color RGB, set new min/max values
new_min = 0
new_max = 1

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_1x1_scaled  = (HH_ML_1x1 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_1x1_scaled  = (HV_ML_1x1 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_1x1_scaled  = np.clip(HH_ML_1x1_scaled, new_min, new_max)
HV_ML_1x1_scaled  = np.clip(HV_ML_1x1_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_ML_1x1 = np.stack((HV_ML_1x1_scaled, HH_ML_1x1_scaled, HH_ML_1x1_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_9x9_scaled  = (HH_ML_9x9 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_9x9_scaled  = (HV_ML_9x9 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_9x9_scaled  = np.clip(HH_ML_9x9_scaled, new_min, new_max)
HV_ML_9x9_scaled  = np.clip(HV_ML_9x9_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_ML_9x9 = np.stack((HV_ML_9x9_scaled, HH_ML_9x9_scaled, HH_ML_9x9_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_ML_21x21_scaled  = (HH_ML_21x21 - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_ML_21x21_scaled  = (HV_ML_21x21 - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_ML_21x21_scaled  = np.clip(HH_ML_21x21_scaled, new_min, new_max)
HV_ML_21x21_scaled  = np.clip(HV_ML_21x21_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_ML_21x21 = np.stack((HV_ML_21x21_scaled, HH_ML_21x21_scaled, HH_ML_21x21_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_MuLoG_scaled  = (HH_MuLoG - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_MuLoG_scaled  = (HV_MuLoG - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_MuLoG_scaled  = np.clip(HH_MuLoG_scaled, new_min, new_max)
HV_MuLoG_scaled  = np.clip(HV_MuLoG_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_MuLoG = np.stack((HV_MuLoG_scaled, HH_MuLoG_scaled, HH_MuLoG_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_SARBM3D_scaled  = (HH_SARBM3D - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_SARBM3D_scaled  = (HV_SARBM3D - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_SARBM3D_scaled  = np.clip(HH_SARBM3D_scaled, new_min, new_max)
HV_SARBM3D_scaled  = np.clip(HV_SARBM3D_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_SARBM3D = np.stack((HV_SARBM3D_scaled, HH_SARBM3D_scaled, HH_SARBM3D_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_baseline_scaled  = (HH_baseline - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_baseline_scaled  = (HV_baseline - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_baseline_scaled  = np.clip(HH_baseline_scaled, new_min, new_max)
HV_baseline_scaled  = np.clip(HV_baseline_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_baseline = np.stack((HV_baseline_scaled, HH_baseline_scaled, HH_baseline_scaled),2)

# ------------------ #

# scale both channels to [new_min,new_max] and clip values below and above
# linear map from sigma0 in dB to new_min and new_max
HH_proposed_scaled  = (HH_proposed - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
HV_proposed_scaled  = (HV_proposed - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
HH_proposed_scaled  = np.clip(HH_proposed_scaled, new_min, new_max)
HV_proposed_scaled  = np.clip(HV_proposed_scaled, new_min, new_max)

# stack to fals-color RGB
RGB_proposed = np.stack((HV_proposed_scaled, HH_proposed_scaled, HH_proposed_scaled),2)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #


# SHOW RGB OVERVIEW OF AOI

if make_AOI_overviews_RGB:

    plt.rcParams.update({'font.size': 8})

    logger.info('Making AOI overviews showing intensities ...')

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_ML_1x1'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_ML_1x1)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nML 1x1')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_ML_9x9'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_ML_9x9)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nML 9x9')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_ML_21x21'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_ML_21x21)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nML 21x21')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_MuLoG'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_MuLoG)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nMuLoG')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_SARBM3D'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_SARBM3D)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nSARBM3D')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_baseline'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_baseline)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nbaseline method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_proposed'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_proposed)
    ax.set_title(f'orbit: {orbit}, false-color intensities\nproposed method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW LABELS OVERVIEW OF AOI

if make_AOI_overviews_labels:

    plt.rcParams.update({'font.size': 8})

    logger.info('Making AOI overviews showing labels ...')

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_ML_1x1'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nML 1x1')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.png', dpi=300)
    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_ML_9x9'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nML 9x9')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #
    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_ML_21x21'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nML 21x21')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_MuLoG'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nMuLoG')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_SARBM3D'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nSARBM3D')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_baseline'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nbaseline method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_proposed'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    ax.set_title(f'orbit: {orbit}, class labels\nproposed method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# move figures into subfolders

png_dir = PAPER_FIG_DIR / 'PNG'
svg_dir = PAPER_FIG_DIR / 'SVG'
pdf_dir = PAPER_FIG_DIR / 'PDF'

png_dir.mkdir(parents=True, exist_ok=True)
svg_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)

os.system(f'mv {PAPER_FIG_DIR}/*png {png_dir}/.')
os.system(f'mv {PAPER_FIG_DIR}/*svg {svg_dir}/.')
os.system(f'mv {PAPER_FIG_DIR}/*pdf {pdf_dir}/.')

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW LABELS AS CLOSEUP COMPARISON

if make_closup_label_comparisons:

    plt.rcParams.update({'font.size': 14})

    logger.info('Making closeup comparison of labels ...')

    """
    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,6)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')
    plt.show()
    """
# --------------------------------------------------------------- #

    # example 1:
    # swath boundary effect
    # "artificial" class in ML due to averaging

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_1'

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')

    axes[0].set_xlim([8350,8650])
    axes[0].set_ylim([2850,2550])

    axes[0].set_xticks([8400, 8500, 8600])
    axes[0].set_yticks([2800, 2700, 2600])

    axes[0].set_xticklabels(['336','340','344'])
    axes[0].set_yticklabels(['112','108','104'])
    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    ##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
    ##cbar.set_ticklabels(legend_entries, weight='bold')
    ##cbar.ax.tick_params(rotation=-45)

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    # example 2:
    # small leads disappearing with ML

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_2'

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')

    axes[0].set_xlim([1150,1450])
    axes[0].set_ylim([9900,9600])

    axes[0].set_xticks([1200, 1300, 1400])
    axes[0].set_yticks([9850, 9750, 9650])

    axes[0].set_xticklabels(['48','52','56'])
    axes[0].set_yticklabels(['394','390','386'])
    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    ##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
    ##cbar.set_ticklabels(legend_entries, weight='bold')
    ##cbar.ax.tick_params(rotation=-45)

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    # example 3:
    # swath boundary effect

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_3'

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')

    axes[0].set_xlim([4350,4650])
    axes[0].set_ylim([7950,7650])

    axes[0].set_xticks([4400, 4500, 4600])
    axes[0].set_yticks([7900, 7800, 7700])

    axes[0].set_xticklabels(['176','180','184'])
    axes[0].set_yticklabels(['316','312','308'])
    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    ##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
    ##cbar.set_ticklabels(legend_entries, weight='bold')
    ##cbar.ax.tick_params(rotation=-45)

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    # example 4:
    # swath boundary effect

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_4'

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')

    axes[0].set_xlim([8250,8550])
    axes[0].set_ylim([2750,2450])

    axes[0].set_xticks([8300, 8400, 8500])
    axes[0].set_yticks([2700, 2600, 2500])

    axes[0].set_xticklabels(['332','336','340'])
    axes[0].set_yticklabels(['108','104','100'])
    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    ##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
    ##cbar.set_ticklabels(legend_entries, weight='bold')
    ##cbar.ax.tick_params(rotation=-45)

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    # example 5:

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_5'

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    h_cbar = axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('ML 1x1')
    axes[1].set_title('ML 21x21')
    axes[2].set_title('MuLoG')
    axes[3].set_title('SARBM3D')
    axes[4].set_title('baseline method')
    axes[5].set_title('proposed method')

    axes[0].set_xlim([3450,3750])
    axes[0].set_ylim([9250,8950])

    axes[0].set_xticks([3500, 3600, 3700])
    axes[0].set_yticks([9200, 9100, 9000])

    axes[0].set_xticklabels(['140','144','148'])
    axes[0].set_yticklabels(['368','364','360'])
    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    ##cbar = fig.colorbar(h_cbar, ax=[axes[2],axes[5]], ticks=cmap_values, pad=0.075)
    ##cbar.set_ticklabels(legend_entries, weight='bold')
    ##cbar.ax.tick_params(rotation=-45)

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# move figures into subfolders

png_dir = PAPER_FIG_DIR / 'PNG'
svg_dir = PAPER_FIG_DIR / 'SVG'
pdf_dir = PAPER_FIG_DIR / 'PDF'

png_dir.mkdir(parents=True, exist_ok=True)
svg_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)

os.system(f'mv {PAPER_FIG_DIR}/*png {png_dir}/.')
os.system(f'mv {PAPER_FIG_DIR}/*svg {svg_dir}/.')
os.system(f'mv {PAPER_FIG_DIR}/*pdf {pdf_dir}/.')

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# MAKE COLORBAR

if make_colorbar:

    logger.info('Making colorbar figure ...')

    plt.rcParams.update({'font.size': 15})

    output_path = PAPER_FIG_DIR / f'colorbar'

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((7,5)))
    h_cbar = ax.imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    cbar = fig.colorbar(h_cbar, ticks=cmap_values, pad=0.075)
    cbar.set_ticklabels(legend_entries, weight='bold')
    cbar.ax.tick_params(rotation=-45)

    plt.savefig('tmp.pdf', dpi=300)

    os.system('pdfcrop --margins "-340 1 1 1" tmp.pdf cbar_tmp.pdf')
    os.system(f'pdfcrop --margins "1 1 1 1" cbar_tmp.pdf {output_path}.pdf')
    os.system('rm *.pdf')
    
    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

    convert_png_command = f'pdftoppm {output_path}.pdf {output_path} -r 300 -png -singlefile'
    os.system(convert_png_command)

# --------------------------------------------------------------- #

    # move figures into subfolders

    png_dir = PAPER_FIG_DIR / 'PNG'
    svg_dir = PAPER_FIG_DIR / 'SVG'
    pdf_dir = PAPER_FIG_DIR / 'PDF'

    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    os.system(f'mv {PAPER_FIG_DIR}/*png {png_dir}/.')
    os.system(f'mv {PAPER_FIG_DIR}/*svg {svg_dir}/.')
    os.system(f'mv {PAPER_FIG_DIR}/*pdf {pdf_dir}/.')

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

sys.exit()

# SHOW OVERVIEW OF AOI

"""
This did not work properly with spacing between sublots.
Ended up making this manually in inkscape
"""

if make_AOI_overviews_RGB_and_labels:


    # overview of aoi: proposed

    fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=((14,5)))
    axes = axes.ravel()
    axes[0].imshow(RGB_proposed)
    h_cbar = axes[1].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[0].set_title('Intensity RGB')
    axes[1].set_title('Class labels')

    axes[0].set_xticks([0,2500,5000,7500])
    axes[0].set_yticks([0,2500,5000,7500])
    axes[0].set_xticklabels(['0','100','200','300'])
    axes[0].set_yticklabels(['0','100','200','300'])
    axes[0].set_ylabel('Distance north (km)')
    axes[0].set_xlabel('Distance east (km)')
    axes[1].set_xlabel('Distance east (km)')

    cbar = fig.colorbar(h_cbar, ticks=cmap_values, pad=0.075)
    cbar.set_ticklabels(legend_entries, weight='bold')
    cbar.ax.tick_params(rotation=-45)

    plt.tight_layout()
    #plt.show()

    plt.savefig('tmp.pdf', dpi=300)



    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" tmp1.pdf tmp1.pdf'
    os.system(crop_pdf_cmd)

    """
    pdfcrop --margins "200 1 -570 1" tmp.pdf left.pdf
    pdfcrop --margins "-510 1 1 1" tmp.pdf right.pdf
    pdfjam left.pdf right.pdf --nup 2x1 --landscape --outfile final.pdf
    """

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# ---- End of <figures_make_all_figures_orbit_043044_0523D1.py> ----
