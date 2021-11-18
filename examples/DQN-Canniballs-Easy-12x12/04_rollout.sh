#!/bin/bash
#################################################################################
#
#             Script Title:   Rollout bash script for environment and agent
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

# Environment, wrappers and model
ENV_NAME="Cameleon-Canniballs-Medium-12x12-v0"
WRAPPERS="canniballs_one_hot,encoding_only"

# Number of episodes and steps to roll out for
NUM_EPISODES=100
NUM_TIMESTEPS=0

#Set the checkpoint dynamically for example
DATE=`date "+%Y.%m.%d"`
CHECKPOINT_PATH="models/DQN_torch_Cameleon-Canniballs-Easy-12x12-v0_rs42_w4_$DATE/checkpoint_010000/checkpoint-10000"
OUTPUT_DIR="rollouts/"
STORE_VIDEO="true"
NO_RENDER="true"
SEED=42

# Hardware requirements
NUM_WORKERS=4
NUM_GPUS=1

# Hickle only useful when
# compressing full image
# frame information
NO_FRAME="true"
CONFIG="{}"

# Need to remove whitespaces
CONFIG="${CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../../"
clear

# Run the script
python -m cameleon.bin.rollout \
  --model-name=$MODEL_NAME \
  --env-name=$ENV_NAME \
  --checkpoint-path=$CHECKPOINT_PATH \
  --num-episodes=$NUM_EPISODES \
  --num-timesteps=$NUM_TIMESTEPS \
  --outdir=$OUTPUT_DIR \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --seed=$SEED \
  --no-frame=$NO_FRAME \
  --config=$CONFIG 
