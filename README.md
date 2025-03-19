# speckle_reduction_4_sea_ice

Collection of scripts to test different speckle reduction methods for sea ice type classification.  
Can be used to reproduce the sea ice work for the paper "Joint despeckling and thermal noise compensation: Application to Sentinel-1 EW GRDM images of the Arctic"


### Background
Ines (and others) have developed a CNN-based method for joint despeckling and thermal noise compensation of S1 EW GRDM imagery.  
Here, we test the effect of this method on high-resolutiuon, pixel-wise classification of sea ie types.
We compare to results obtained from simple multi-looking with different window sizes and other deep-learning based methods.

Sentinel-1 images over Beligica Bank during the CIRFA-22 are selected by Johannes.

Tiff files with calibrated and speckle reduced-data are provided by Loic and Ines.

### Set up conda env
Either run everything in the environment specified in the workflow (some packages like matplotlib or cartopy might be missing in standard installations), or create a new environment with all libraries installed.

    # create and activate new environment
    conda create -y -n sr4si gdal
    conda activate sr4si

    # install required packages
    conda install -y loguru matplotlib scikit-learn cartopy scipy pillow lxml python-dotenv
    pip install ipython
    pip install labelme

    # install own libraries
    pip install git+https://github.com/jlo031/labelme_utils
    pip install git+https://github.com/jlo031/geocoding
    pip install git+https://github.com/jlo031/GLIA

Processing step 3 of the workflow below requires correct installation of the S1_processing library, which needs the correct gpt_path set.
Refer to the github page for more details (https://github.com/jlo031/S1_processing).


### Workflow

1. Unpack the zip-files.

       ./unpack_data.sh

2. Prepare the speckle reduced data (tiff) for input into the *ice_type_classification" module*.

       conda run -n GLIA python prepare_speckle_reduced_data.py

3. Extract HH, HV, IA, swath_mask, landmask for different ML levels.

       conda run -n S1_processing python extract_features.py

4. Get landmask and make valid mask.

       conda run -n geocoding python get_landmask_make_valid_mask.py

5. Classify all images using all speckle reduction methods separately.

       conda run -n GLIA python classify_images.py

6. Geocode features and results for visualization on map.

       conda run -n geocoding python geocode_features_and_results.py

7. Merge features and labels from same orbits and crop to AOI.

       conda run -n geocoding python merge_and_crop_orbits.py

8. Make RGBs for labelme. This script can/should be adjusted to produced multi-channel and single-channel false-color RGBs for easy labeling.

       conda run -n LABELME python make_scaled_RGBs_from_AOI_crops.py 

9. Label ROIs along the swath boundaries.

       ./label_geocoded_AOI_images.sh /media/jo/LaCie_Q/EO_data/speckle_reduction_tests/Sentinel-1/RGBs config/labels.txt 

10. Convert ROI json files to validation masks.

        conda run -n LABELME python convert_json_2_training_masks.py

11. Inspect ROIs and results visually. Best to run interactively for plots.

        conda run -n LABELME python check_ROIs_and_results_visually.py

12. Evaluate results numerically.

        conda run -n LABELME python evaluate_results.py


### Figures

All scripts for visualization start with *figures_*.  

*figures_make_overview_maps.py*  
Makes simple overview figures of the entire orbits.

*figures_4_paper_test.py*  
Tests possible figure setups for the paper. Reads lots of data, may run out of memory.

*figures_4_paper_final.py*  
Makes final figures for the revised paper.

Actual final figures for the paper are combined versions of the outputs from *figures_4_paper_final.py* and saved in the SVG folder of this repository.

