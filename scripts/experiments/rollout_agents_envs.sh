# !/bin/bash
#################################################################################
#
#             Script Title:   Rollout several agents together, potentially over time
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Set variable names
#######################################################################

# Required arguments
EXPERIMENT_CONFIG="""{
'APPO':{
          'checkpoint_root':'models/APPO_torch_Cameleon-Canniballs-Medium-12x12-v0_rs42_w14_2021.08.23/',
          'checkpoints':[600,800,1000,1200,1400,1600,1800]
         }
}"""


ENV_NAMES="Cameleon-Canniballs-Medium-12x12-v0"

# Static information across execution
OUTPUT_DIR="rollouts/"
STORE_VIDEO="true"

# Usually makes 5x what you put here, not sure why
EPISODES=100
TIMESTEPS=0
NO_RENDER="true"
WRAPPERS="canniballs_one_hot,encoding_only"
NUM_WORKERS=10
NUM_GPUS=1
SEED=42
USE_HICKLE="false"
NO_FRAME="true"
STORE_IMAGO="false"
IMAGO_DIR="data/imago/"
IMAGO_FEATURES="observation,action_dist,action_logits,value_function"
BUNDLE_ONLY="false"
BUNDLE_ONLY_DIR=""
SYNC_BUNDLES="true"
LOG_LEVEL="info"

# Rollout process can pick up trained config if a checkpoint is given
# otherwise, specify information here
CONFIG="{}"

# Need to remove whitespaces
CONFIG="${CONFIG//[$'\t\r\n ']}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG//[$'\t\r\n ']}"

#######################################################################
# Run the script for training
#######################################################################

# change to project root directory (in case invoked from other dir)
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$DIR/../../"
clear

# Run the script
python -m cameleon.bin.experiments.rollout_agents_envs \
  --experiment-config=$EXPERIMENT_CONFIG \
  --env-names=$ENV_NAMES \
  --num-episodes=$EPISODES \
  --num-timesteps=$TIMESTEPS \
  --outdir=$OUTPUT_DIR \
  --store-video=$STORE_VIDEO \
  --no-render=$NO_RENDER \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --seed=$SEED \
  --no-frame=$NO_FRAME \
  --use-hickle=$USE_HICKLE \
  --store-imago=$STORE_IMAGO \
  --imago-dir=$IMAGO_DIR \
  --imago-features=$IMAGO_FEATURES \
  --bundle-only=$BUNDLE_ONLY \
  --bundle-only-dir=$BUNDLE_ONLY_DIR \
  --sync-bundles=$SYNC_BUNDLES \
  --log-level=$LOG_LEVEL \
  --config=$CONFIG 
