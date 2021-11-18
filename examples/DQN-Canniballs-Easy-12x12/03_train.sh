#!bin/bash
################################################################################
#
#             Script Title:   Training bash script for environment with DQN
#                             and Canniballs Easy environment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################

# Model environment setup
OUTPUT_DIR="models/"
ENV_NAME="Cameleon-Canniballs-Easy-12x12-v0"
MODEL_NAME="DQN"
WRAPPERS="canniballs_one_hot,encoding_only"

# Training information
NUM_EPOCHS=10000
CHECKPOINT_EPOCHS=20

# Hardware requirements
NUM_WORKERS=4
NUM_GPUS=1

# Model and execution information
FRAMEWORK="torch"
TUNE="false"
CONFIG="""{
'model':{'dim':12,
         'conv_filters':[[16,[4,4],1],
                         [32,[3,3],2],
                         [512,[6,6],1]]
        }
}"""

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
python -m cameleon.bin.train \
  --num-epochs=$NUM_EPOCHS \
  --env-name=$ENV_NAME \
  --model-name=$MODEL_NAME \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --checkpoint-epochs=$CHECKPOINT_EPOCHS \
  --outdir=$OUTPUT_DIR \
  --checkpoint-path=$CHECKPOINT_PATH \
  --framework=$FRAMEWORK \
  --tune=$TUNE \
  --config=$CONFIG 
