#!bin/bash
################################################################################
#
#             Script Title:   Training bash script for environment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Set variable names
#######################################################################


ENV_NAMES="Cameleon-Canniballs-Easy-12x12-v0,Cameleon-Canniballs-Hard-12x12-v0"
MODEL_NAMES="APPO,IMPALA"

OUTPUT_DIR="models/"
NUM_EPOCHS=0
NUM_EPISODES=0
NUM_TIMESTEPS=50000000
WRAPPERS="canniballs_one_hot,encoding_only"
CHECKPOINT_EPOCHS=20
# CHECKPOINT_PATH="models/DONT_DELETE_DQN_tf2_Cameleon-Canniballs-Easy-12x12-v0_2021.07.22/checkpoint_009962/checkpoint-9962"
NUM_WORKERS=14
NUM_GPUS=1
FRAMEWORK="torch"
SEED=42
TUNE="false"
VERBOSE="true"
RAY_OBJ_STORE_MEM=3500000000
LOG_LEVEL="info"
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
python -m cameleon.bin.experiments.train_agents_envs \
  --env-names=$ENV_NAMES \
  --model-names=$MODEL_NAMES \
  --num-epochs=$NUM_EPOCHS \
  --num-episodes=$NUM_EPISODES \
  --num-timesteps=$NUM_TIMESTEPS \
  --wrappers=$WRAPPERS \
  --num-workers=$NUM_WORKERS \
  --num-gpus=$NUM_GPUS \
  --checkpoint-epochs=$CHECKPOINT_EPOCHS \
  --outdir=$OUTPUT_DIR \
  --checkpoint-path=$CHECKPOINT_PATH \
  --framework=$FRAMEWORK \
  --tune=$TUNE \
  --seed=$SEED \
  --ray-obj-store-mem=$RAY_OBJ_STORE_MEM \
  --config=$CONFIG \
  --email-updates=$EMAIL_UPDATES \
  --email-server=$EMAIL_SERVER \
  --email-sender=$EMAIL_SENDER \
  --email-receiver=$EMAIL_RECEIVER \
  --log-level=$LOG_LEVEL 
