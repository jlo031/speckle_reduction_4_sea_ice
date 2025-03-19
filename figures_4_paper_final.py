# ---- This is <figures_4_paper_final.py> ----

"""
Make the final figures for the paper (revised version).
"""

import os
import pathlib
import shutil
import sys

from loguru import logger

import labelme_utils.json_conversion as lm_json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

from osgeo import gdal

from config.folder_structure import *

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# S1 orbit to process
orbit = '043044_0523D1'

# parameters for RGB scaling
vmin_HH = -35
vmax_HH = 0
vmin_HV = -40
vmax_HV = -5

# set new min/max values for RGBs
new_min = 0
new_max = 1

# linewidth for ROIs
linewidth_small = 2
linewidth_large = 3

# cropped image coordinates (must be the same as in "make_scaled_RGBs_from_AOI_crops.py")
xmin = 800
xmax = 10000
ymin = 2500
ymax = 10000

# define label txt file path
labels_path = 'config/labels.txt'

fig1a = False
fig1a_polygons = True
fig1a_title = False

fig1b = False
fig1b_polygons = False
fig1b_title = False

fig2 = True
fig2_polygons = True
fig2_title = True

fig3 = True
fig3_polygons = True
fig3_title = True

fig_colorbar = True

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# build color map and legend (needed for all figures)

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

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# load class polygons (needed for all figures)

# define json path
json_path = S1_RGB_DIR / f'{orbit}_proposed_RGB_CROP.json'

# get class name list
class_names = lm_json.get_class_name_list_from_labels_txt(labels_path)

# get class_labels_dict
class_labels_dict = lm_json.get_label_index_mapping(class_names)

# get polygons from json file
shapes, label_index_dict = lm_json.load_training_shapes(json_path, label_index_mapping=class_labels_dict)

# get number of polygons
N_polygons = np.size(shapes)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW RGB OVERVIEW OF AOI WITH/WITHOUT POLYGONS

if fig1a:

    logger.info('Preparing Fig 1a')

    # ------------------------ #

    if fig1a_polygons:
        output_string = '_ROIs'
    else:
        output_string = ''

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_intensities_proposed{output_string}'

    plt.rcParams.update({'font.size': 13})

    # ------------------------ #

    # load intensities
    intensities_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'HH_HV_proposed_crop.tiff').as_posix()).ReadAsArray().transpose(1,2,0)

    # make intensity RGB
    HH_proposed = intensities_proposed[:,:,0]
    HV_proposed = intensities_proposed[:,:,1]
    HH_proposed[HH_proposed==0] = np.nan
    HV_proposed[HV_proposed==0] = np.nan

    HH_proposed_scaled  = (HH_proposed - (vmin_HH)) * ((new_max - new_min) / ((vmax_HH) - (vmin_HH))) + new_min
    HV_proposed_scaled  = (HV_proposed - (vmin_HV)) * ((new_max - new_min) / ((vmax_HV) - (vmin_HV))) + new_min
    HH_proposed_scaled  = np.clip(HH_proposed_scaled, new_min, new_max)
    HV_proposed_scaled  = np.clip(HV_proposed_scaled, new_min, new_max)

    RGB_proposed = np.stack((HV_proposed_scaled, HH_proposed_scaled, HH_proposed_scaled),2)

    RGB_proposed[np.isnan(RGB_proposed)] = 0

    # ------------------------ #

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(RGB_proposed)

    if fig1a_title:
        ax.set_title(f'orbit: {orbit}, false-color intensities\nproposed method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    # ------------------------ #

    if fig1a_polygons:
        # loop over all polygons and draw them onto the figure
        for i in np.arange(N_polygons):
            p   = shapes[i]
            cl  = label_index_dict[p['label']]
            polygon = np.array(p['points'])

            x_start, x_end = polygon[:,0]
            y_start, y_end = polygon[:,1]
            x_start = x_start + ymin
            x_end = x_end + ymin
            y_start = y_start + xmin
            y_end = y_end + xmin
            xvec = [x_start, x_end, x_end, x_start, x_start]
            yvec = [y_start, y_start, y_end, y_end, y_start]    

            ax.plot(xvec,yvec,color = class_colors_norm[cl], linewidth=linewidth_large)

    # ------------------------ #

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW LABELS OVERVIEW OF AOI WITH/WITHOUT POLYGONS

if fig1b:

    logger.info('Preparing Fig 1b')

    if fig1b_polygons:
        output_string = '_ROIs'
    else:
        output_string = ''

    output_path = PAPER_FIG_DIR / f'AOI_overview_orbit_{orbit}_labels_proposed{output_string}'

    plt.rcParams.update({'font.size': 13})

    # ------------------------ #

    # load labels
    labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()

    # ------------------------ #

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((6,5)))
    ax.imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    if fig1b_title:
        ax.set_title(f'orbit: {orbit}, class labels\nproposed method')

    ax.set_xticks([0,2500,5000,7500])
    ax.set_yticks([0,2500,5000,7500])
    ax.set_xticklabels(['0','100','200','300'])
    ax.set_yticklabels(['0','100','200','300'])
    ax.set_ylabel('Distance north (km)')
    ax.set_xlabel('Distance east (km)')

    # ------------------------ #

    if fig1b_polygons:
        # loop over all polygons and draw them onto the figure
        for i in np.arange(N_polygons):
            p   = shapes[i]
            cl  = label_index_dict[p['label']]
            polygon = np.array(p['points'])

            x_start, x_end = polygon[:,0]
            y_start, y_end = polygon[:,1]
            x_start = x_start + ymin
            x_end = x_end + ymin
            y_start = y_start + xmin
            y_end = y_end + xmin
            xvec = [x_start, x_end, x_end, x_start, x_start]
            yvec = [y_start, y_start, y_end, y_end, y_start]    

            ax.plot(xvec,yvec,color = class_colors_norm[cl], linewidth=linewidth_large)

    # ------------------------ #

    plt.savefig(f'{output_path}.pdf', dpi=300)
    
    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "3 3 3 3" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW LABELS AS CLOSEUP COMPARISON WITH/WITHOUT POLYGONS

if fig2:

    logger.info('Preparing Fig 2')

    if fig2_polygons:
        output_string = '_ROIs'
    else:
        output_string = ''

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_3{output_string}'

    plt.rcParams.update({'font.size': 13})

    # ------------------------ #

    # load labels
    labels_ML_1x1   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_1x1_crop.tiff').as_posix()).ReadAsArray()
    labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()
    labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
    labels_ML_21x21 = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_21x21_crop.tiff').as_posix()).ReadAsArray()
    labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_MuLoG_crop.tiff').as_posix()).ReadAsArray()
    labels_SARBM3D  = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_SARBM3D_crop.tiff').as_posix()).ReadAsArray()
    labels_baseline = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_baseline_crop.tiff').as_posix()).ReadAsArray()

    # ------------------------ #

    fig, axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=((12,8)))
    axes = axes.ravel()
    axes[0].imshow(labels_ML_1x1, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_ML_21x21, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_SARBM3D, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_baseline, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    if fig2_title:
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

    # ------------------------ #

    if fig2_polygons:
        # loop over all polygons and draw them onto the figure
        for i in np.arange(N_polygons):
            p   = shapes[i]
            cl  = label_index_dict[p['label']]
            polygon = np.array(p['points'])

            x_start, x_end = polygon[:,0]
            y_start, y_end = polygon[:,1]
            x_start = x_start + ymin
            x_end = x_end + ymin
            y_start = y_start + xmin
            y_end = y_end + xmin
            xvec = [x_start, x_end, x_end, x_start, x_start]
            yvec = [y_start, y_start, y_end, y_end, y_start]    

            axes[0].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[1].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[2].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[3].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[4].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[5].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)

    # ------------------------ #

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# SHOW LABELS AS CLOSEUP COMPARISON WITH/WITHOUT POLYGONS

if fig3:

    logger.info('Preparing Fig 3')

    plt.rcParams.update({'font.size': 13})

    if fig3_polygons:
        output_string = '_ROIs'
    else:
        output_string = ''

    output_path = PAPER_FIG_DIR / f'labels_closeup_orbit_{orbit}_example_1{output_string}'

    # ------------------------ #

    # load labels
    labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()
    labels_ML_9x9   = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_ML_9x9_crop.tiff').as_posix()).ReadAsArray()
    labels_MuLoG    = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_MuLoG_crop.tiff').as_posix()).ReadAsArray()

    # ------------------------ #

    fig, axes = plt.subplots(2,3,figsize=((12,8)))
    axes = axes.ravel()
    axes[0].imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[1].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[2].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[3].imshow(labels_ML_9x9, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[4].imshow(labels_MuLoG, interpolation='nearest', cmap=cmap, norm=cmap_norm)
    axes[5].imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    if fig3_title:
        axes[0].set_title('ML 9x9')
        axes[1].set_title('MuLoG')
        axes[2].set_title('proposed method')
        axes[3].set_title('ML 9x9')
        axes[4].set_title('MuLoG')
        axes[5].set_title('proposed method')

    axes[0].set_xlim([8350,8650])
    axes[0].set_ylim([2850,2550])
    axes[1].set_xlim([8350,8650])
    axes[1].set_ylim([2850,2550])
    axes[2].set_xlim([8350,8650])
    axes[2].set_ylim([2850,2550])
    axes[3].set_xlim([3450,3750])
    axes[3].set_ylim([9250,8950])
    axes[4].set_xlim([3450,3750])
    axes[4].set_ylim([9250,8950])
    axes[5].set_xlim([3450,3750])
    axes[5].set_ylim([9250,8950])

    axes[0].set_xticks([8400, 8500, 8600])
    axes[0].set_yticks([2800, 2700, 2600])
    axes[1].set_xticks([8400, 8500, 8600])
    axes[1].set_yticks([2800, 2700, 2600])
    axes[2].set_xticks([8400, 8500, 8600])
    axes[2].set_yticks([2800, 2700, 2600])
    axes[3].set_xticks([3500, 3600, 3700])
    axes[3].set_yticks([9200, 9100, 9000])
    axes[4].set_xticks([3500, 3600, 3700])
    axes[4].set_yticks([9200, 9100, 9000])
    axes[5].set_xticks([3500, 3600, 3700])
    axes[5].set_yticks([9200, 9100, 9000])

    axes[0].set_yticklabels(['112','108','104'])
    axes[1].set_yticklabels(['','',''])
    axes[2].set_yticklabels(['','',''])
    axes[0].set_xticklabels(['336','340','344'])
    axes[1].set_xticklabels(['336','340','344'])
    axes[2].set_xticklabels(['336','340','344'])

    axes[3].set_yticklabels(['368','364','360'])
    axes[4].set_yticklabels(['','',''])
    axes[5].set_yticklabels(['','',''])
    axes[3].set_xticklabels(['140','144','148'])
    axes[4].set_xticklabels(['140','144','148'])
    axes[5].set_xticklabels(['140','144','148'])

    axes[0].set_ylabel('Distance north (km)')
    axes[3].set_ylabel('Distance north (km)')
    axes[4].set_xlabel('Distance east (km)')

    # ------------------------ #

    if fig3_polygons:
        # loop over all polygons and draw them onto the figure
        for i in np.arange(N_polygons):
            p   = shapes[i]
            cl  = label_index_dict[p['label']]
            polygon = np.array(p['points'])

            x_start, x_end = polygon[:,0]
            y_start, y_end = polygon[:,1]
            x_start = x_start + ymin
            x_end = x_end + ymin
            y_start = y_start + xmin
            y_end = y_end + xmin
            xvec = [x_start, x_end, x_end, x_start, x_start]
            yvec = [y_start, y_start, y_end, y_end, y_start]    

            axes[0].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[1].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[2].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[3].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[4].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)
            axes[5].plot(xvec,yvec,color = [1,0,0], linewidth=linewidth_small)

    # ------------------------ #

    plt.savefig(f'{output_path}.pdf', dpi=300)

    plt.close('all')

    crop_pdf_cmd = f'pdfcrop --margins "1 1 1 1" {output_path}.pdf {output_path}.pdf'
    os.system(crop_pdf_cmd)

    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# MAKE COLORBAR

if fig_colorbar:

    logger.info('Making colorbar figure ...')

    output_path = PAPER_FIG_DIR / f'colorbar'

    plt.rcParams.update({'font.size': 13})

    # ------------------------ #

    # load labels
    labels_proposed = gdal.Open((S1_ORBIT_DIR/f'{orbit}'/'AOIs'/'labels_proposed_crop.tiff').as_posix()).ReadAsArray()

    # ------------------------ #

    plt.rcParams.update({'font.size': 15})

    fig, ax = plt.subplots(1,1,sharex=True,sharey=True,figsize=((7,5)))
    h_cbar = ax.imshow(labels_proposed, interpolation='nearest', cmap=cmap, norm=cmap_norm)

    cbar = fig.colorbar(h_cbar, ticks=cmap_values, pad=0.075)
    cbar.set_ticklabels(legend_entries, weight='bold')
    cbar.ax.tick_params(rotation=-45)

    # ------------------------ #

    plt.savefig('tmp.pdf', dpi=300)

    os.system('pdfcrop --margins "-340 1 1 1" tmp.pdf cbar_tmp.pdf')
    os.system(f'pdfcrop --margins "1 1 1 1" cbar_tmp.pdf {output_path}.pdf')
    os.system('rm *.pdf')
    
    convert_svg_command = f'pdf2svg {output_path}.pdf {output_path}.svg'
    os.system(convert_svg_command)

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# move figures into subfolders

svg_dir = PAPER_FIG_DIR / 'SVG'
pdf_dir = PAPER_FIG_DIR / 'PDF'

svg_dir.mkdir(parents=True, exist_ok=True)
pdf_dir.mkdir(parents=True, exist_ok=True)

os.system(f'mv {PAPER_FIG_DIR}/*svg {svg_dir}/.')
os.system(f'mv {PAPER_FIG_DIR}/*pdf {pdf_dir}/.')

# --------------------------------------------------------------- #
# --------------------------------------------------------------- #

# ---- End of <figures_4_paper_final.py> ----
