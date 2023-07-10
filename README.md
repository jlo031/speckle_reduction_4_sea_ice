# speckle_reduction_4_sea_ice

Collection of scripts to test speckle reduction methods developed by Ines, Loic, and others for sea ice type classification.


### Background
Loic and Ines have developed novel CNN-based methods for speckle reduction in S1 EW mode imagery. Here, we test the effect of these methods on high-resolutiuon, pixel-wise classification of sea ie types. We compare to results obtained from simple multi-looking with different window sizes.

Sentinel-1 images over Beligica Bank during the CIRFA-22 are selected by Johannes.

Tiff files with calibrated and speckle reduced-data are provided by Loic and Ines.


### Workflow
1) Prepare the speckle reduced data (tiff) for input into the *ice_type_classification" module*: **prepare_speckle_reduced_data.py**
2) Extract HH, HV, IA, swath_mask, landmask for different ML levels: **extract_features.py**
3) Classify all selected images using all speckle reduction methods separately: **classify_images.py**
4) Geocode features and results for easy visualization on a map: **goecode_features_and_results.py**

