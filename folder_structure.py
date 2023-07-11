# ---- This is <folder_structure.py> ----

"""
Define folder structure for speckle reduction tests.
"""

import pathlib

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

DATA_DIR      = pathlib.Path('/Data/speckle_reduction_tests')
S1_DIR        = DATA_DIR / 'Sentinel-1'
S1_L1_DIR     = S1_DIR / 'L1'
S1_FEAT_DIR   = S1_DIR / 'features'
S1_RESULT_DIR = S1_DIR / 'classification_results'
S1_LOIC_DIR   = S1_DIR / 'from_loic'
S1_GEO_DIR    = S1_DIR / 'geocoded'
S1_ORBIT_DIR  = S1_GEO_DIR / 'orbits'
FIG_DIR       = S1_DIR.parent / 'figures'

S1_FEAT_DIR.mkdir(parents=True, exist_ok=True)
S1_RESULT_DIR.mkdir(parents=True, exist_ok=True)
S1_GEO_DIR.mkdir(parents=True, exist_ok=True)
S1_ORBIT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <folder_structure.py> ----
