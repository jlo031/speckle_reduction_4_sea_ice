# speckle_reduction_4_sea_ice

Collection of scripts to test different speckle reduction methods for sea ice type classification.


### Background
In her PhD, Ines has developed a novel CNN-based method for speckle reduction in S1 EW mode imagery. Here, we test the effect of this method on high-resolutiuon, pixel-wise classification of sea ie types. We compare to results obtained from simple multi-looking with different window sizes and other deep-learning based methods.

Sentinel-1 images over Beligica Bank during the CIRFA-22 are selected by Johannes.

Tiff files with calibrated and speckle reduced-data are provided by Loic and Ines.


### Workflow
1) Unpack the zip-files: **unpack_data.sh**
2) Prepare the speckle reduced data (tiff) for input into the *ice_type_classification" module*: **prepare_speckle_reduced_data.py**
3) Extract HH, HV, IA, swath_mask, landmask for different ML levels: **extract_features.py**
4) Classify all selected images using all speckle reduction methods separately: **classify_images.py**
5) Make valid mask (based on swath and landmask): **make_valid_mask.py**
6) Mask out invalid pixels in classification result: **combine_results_with_valid_mask.py**
7) Visualize in radar geometry for first inspection: **show_results.py**
8) Geocode features and results for visualization on a map: **geocode_features_and_results.py**
9) Merge features and labels from same orbits and crop to AOI: **merge_and_crop_orbits.py**
