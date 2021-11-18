#!/bin/bash

#################################################################################
#
#             Script Title:   Speed benchmarking test for Environment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################

ENV_NAME="Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0"
NUM_ENC_FRAMES=5000
NUM_VIZ_FRAMES=100
NUM_RESETS=500
WRAPPERS="canniballs_one_hot,encoding_only"
VISUAL="false"
LOG_LEVEL="info"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.benchmark \
  --env-name=$ENV_NAME \
  --num-enc-frames=$NUM_ENC_FRAMES \
  --num-viz-frames=$NUM_VIZ_FRAMES \
  --num-resets=$NUM_RESETS \
  --wrappers=$WRAPPERS \
  --visual=$VISUAL \
  --log-level=$LOG_LEVEL
