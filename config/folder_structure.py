# ---- This is <folder_structure.py> ----

"""
Define folder structure for speckle reduction tests.
"""

import pathlib

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

##DATA_DIR      = pathlib.Path('/media/Data/speckle_reduction_tests')
DATA_DIR      = pathlib.Path('/media/jo/EO_disk/data/speckle_reduction_tests')
##DATA_DIR      = pathlib.Path('/media/johannes/LaCie_Q/EO_data/speckle_reduction_tests')

# -------------------------------------------------------------------------- #

osm_landmask_path = pathlib.Path('/media/Data/osm_shapefiles/land-polygons-split-4326/land_polygons.shp')

# -------------------------------------------------------------------------- #

S1_DIR        = DATA_DIR / 'Sentinel-1'

S1_L1_DIR     = S1_DIR / 'L1'
S1_FEAT_DIR   = S1_DIR / 'features'
S1_RESULT_DIR = S1_DIR / 'classification_results'
S1_LOIC_DIR   = S1_DIR / 'from_loic'
S1_GEO_DIR    = S1_DIR / 'geocoded'
S1_ORBIT_DIR  = S1_GEO_DIR / 'orbits'
S1_RGB_DIR    = S1_DIR / 'RGBs'
S1_VAL_DIR    = S1_DIR / 'validation_masks'

FIG_DIR       = S1_DIR.parent / 'figures'
PAPER_FIG_DIR = FIG_DIR / 'figures_4_paper'

S1_FEAT_DIR.mkdir(parents=True, exist_ok=True)
S1_RESULT_DIR.mkdir(parents=True, exist_ok=True)
S1_GEO_DIR.mkdir(parents=True, exist_ok=True)
S1_ORBIT_DIR.mkdir(parents=True, exist_ok=True)
S1_RGB_DIR.mkdir(parents=True, exist_ok=True)
S1_VAL_DIR.mkdir(parents=True, exist_ok=True)

FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <folder_structure.py> ----
