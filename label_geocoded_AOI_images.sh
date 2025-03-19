# ---- This is <./label_geocoded_AOI_images.sh> ----
#!/bin/bash
#
# Start labelme for images in input folder
#
# Typical usage example for this project:
#     ./label_geocoded_AOI_images.sh /media/jo/EO_disk/data/speckle_reduction_tests/Sentinel-1/RGBs config/labels.txt 
#     ./label_geocoded_AOI_images.sh /media/jo/LaCie_Q/EO_data/speckle_reduction_tests/Sentinel-1/RGBs config/labels.txt 
#
# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# check that at least one argument is given
if [ $# -lt 2 ]; then
  echo " "
  echo "ERROR: At least two positional parameters required"
  echo "Usage: ./label_geocoded_AOI_images.sh IMAGE_DIR LABELS_FILE"
  echo " "
  exit
fi

#set IMAGE_DIR
IMAGE_DIR=$1

# check that IMAGE_DIR exists
if [ ! -d "${IMAGE_DIR}" ]; then
  echo " "
  echo "ERROR: Could not find IMAGE_DIR: '${IMAGE_DIR}'"
  echo "Usage: ./label_geocoded_AOI_images.sh IMAGE_DIR LABELS_FILE"
  echo " "
  exit
fi

# set LABELS_FILE
LABELS_FILE=$2

# check that LABELS_FILE exists
if [ ! -f "${LABELS_FILE}" ]; then
  echo " "
  echo "ERROR: Could not find LABELS_FILE: '${LABELS_FILE}'"
  echo "Usage: ./label_geocoded_AOI_images.sh IMAGE_DIR LABELS_FILE"
  echo " "
  exit
fi

echo " "
echo "Labeling scaled cropped images"
echo "IMAGE_DIR:   ${IMAGE_DIR}"
echo "LABELS_FILE: ${LABELS_FILE}"
echo " "

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# run labelme
labelme ${IMAGE_DIR} --labels ${LABELS_FILE} --nodata

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #

# ---- End of <label_geocoded_AOI_images.sh> ----
