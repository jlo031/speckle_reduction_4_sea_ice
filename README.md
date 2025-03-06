# speckle_reduction_4_sea_ice

Collection of scripts to test different speckle reduction methods for sea ice type classification.  
Can be used to reproduce the sea ice work for the paper "Joint despeckling and thermal noise compensation: Application to Sentinel-1 EW GRDM images of the Arctic"


### Background
Ines (and others) hav developed a CNN-based method for joint despeckling and thermal noise compensationof S1 EW GRDM imagery.  
Here, we test the effect of this method on high-resolutiuon, pixel-wise classification of sea ie types. We compare to results obtained from simple multi-looking with different window sizes and other deep-learning based methods.

Sentinel-1 images over Beligica Bank during the CIRFA-22 are selected by Johannes.

Tiff files with calibrated and speckle reduced-data are provided by Loic and Ines.


### Workflow

1. Unpack the zip-files

       ./unpack_data.sh

2. Prepare the speckle reduced data (tiff) for input into the *ice_type_classification" module*

       conda run -n GLIA python prepare_speckle_reduced_data.py

3. Extract HH, HV, IA, swath_mask, landmask for different ML levels

       conda run -n S1_processing python extract_features.py

4. Get landmask and make valid mask

       conda run -n geocoding python get_landmask_make_valid_mask.py

5. Classify all images using all speckle reduction methods separately

       conda run -n GLIA python classify_images.py

6. Geocode features and results for visualization on map

       conda run -n geocoding python geocode_features_and_results.py

7. Merge features and labels from same orbits and crop to AO

       conda run -n GLIA python merge_and_crop_orbits.py**



These steps are deprecated/not needed anymore
10) Mask out invalid pixels in classification result: **combine_results_with_valid_mask.py**
11) Visualize in radar geometry for first inspection: **show_results.py**
