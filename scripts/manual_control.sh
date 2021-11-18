#!/bin/bash

#################################################################################
#
#             Script Title:   Manual control of specific Gym environment (mostly for testing)
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################

ENV_NAME="Cameleon-Canniballs-Medium-NStep-Consumption-12x12-v0"
# ENV_NAME="Cameleon-Canniballs-Easy-12x12-v0"
KEY_HANDLER="cameleon"
SEED=-1
TILE_SIZE=32
VERBOSE="true"

# SHould probably add wrappers as well

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.manual_control \
  --env-name=$ENV_NAME \
  --key-handler=$KEY_HANDLER \
  --seed=$SEED \
  --tile-size=$TILE_SIZE \
  --verbose=$VERBOSE
