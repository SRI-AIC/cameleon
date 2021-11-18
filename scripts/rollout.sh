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

OUTPUT_DIR="rollouts/"
STORE_VIDEO="true"
# ENV_NAME="Cameleon-Canniballs-Hard-12x12-v0"
ENV_NAME="Cameleon-Canniballs-Hard-12x12-v0"
# ENV_NAME="Cameleon-Canniballs-Medium-NStep-Avoidance-12x12-v0"

# Usually makes 5x what you put here, not sure why
EPISODES=200
TIMESTEPS=0

CHECKPOINT_PATH="models/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0_rs42_w14_2021.08.23/checkpoint_002020/checkpoint-2020"
NO_RENDER="true"
WRAPPERS="canniballs_one_hot,encoding_only"
NUM_WORKERS=10
NUM_GPUS=1
SEED=42
USE_HICKLE="false"
NO_FRAME="true"
# Items to be collected beyond obs, reward, action, done
TO_COLLECT="action_dist,action_logits,value_function"
STORE_IMAGO="false"
IMAGO_DIR="data/imago/"
IMAGO_FEATURES="observation,action_dist,action_logits,value_function"
BUNDLE_ONLY="false"
BUNDLE_ONLY_DIR=""
SYNC_BUNDLES="false"
LOG_LEVEL="info"

# Rollout process can pick up trained config if a checkpoint is given
# otherwise, specify information here
CONFIG="{}"

# Need to remove whitespaces
CONFIG="${CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../"
clear

# Run the script
python -m cameleon.bin.rollout \
  --env-name=$ENV_NAME \
  --checkpoint-path=$CHECKPOINT_PATH \
  --num-episodes=$EPISODES \
  --num-timesteps=$TIMESTEPS \
  --model-name=$MODEL_NAME \
  --outdir=$OUTPUT_DIR \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --seed=$SEED \
  --no-frame=$NO_FRAME \
  --use-hickle=$USE_HICKLE \
  --to-collect=$TO_COLLECT \
  --store-imago=$STORE_IMAGO \
  --imago-dir=$IMAGO_DIR \
  --imago-features=$IMAGO_FEATURES \
  --bundle-only=$BUNDLE_ONLY \
  --bundle-only-dir=$BUNDLE_ONLY_DIR \
  --sync-bundles=$SYNC_BUNDLES \
  --log-level=$LOG_LEVEL \
  --config=$CONFIG 
