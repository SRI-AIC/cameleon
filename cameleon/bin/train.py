################################################################################
#
#             Project Title:  Training API for RLlib and Cameleon
#             Author:         Sam Showalter
#             Date:           2021-07-07
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

# Logistical Packages
import os
import json
import sys
import traceback
import logging
from tqdm import tqdm, trange
import argparse
import importlib
import time
import datetime as dt

# DL and distributed libs
import ray
import torch

# Environment libs
import gym
import gym_minigrid
import cameleon.envs
from gym import envs

# Import all ray models for zoo and registration for ray
from ray.tune.registry import register_env
from ray import tune
from cameleon.callbacks.rllib.tune_progress import CameleonRLlibTuneReporter
from cameleon.utils.env import wrap_env,load_env, cameleon_logger_creator
from cameleon.utils.parser import str2bool, str2dict, dict2str, str2wrapper,str2model,\
    update_config, str2framework, str2int, str2str, str2log_level
from cameleon.utils.messages import CameleonEmailBot
from cameleon.utils.general import _write_pkl, _save_metadata


#####################################################################################
# Argparse formation
#####################################################################################

def create_parser(parser_creator = None):
    """Create training argument parser

    :parser_creator: Creator for argument parser
    :returns: parser

    """
    parser = argparse.ArgumentParser(description='Cameleon Training API with RLlib')

    # Required
    parser.add_argument('--env-name', default = None, required = True, help = "Any argument registered with gym, including Gym Minigrid and Cameleon Environments")
    parser.add_argument('--model-name', default=None,required = True, type = str, help='SAC, PPO, PG, A2C, A3C, IMPALA, ES, DDPG, DQN, MARWIL, APEX, or APEX_DDPG')

    # Add optiona arguments
    parser = create_optional_args(parser)

    return parser


def create_optional_args(parser):
    """Add optional arguments to parser

    :parser: Argparse.parser: Argument parser
    :returns: parser

    """
    # Optional
    parser.add_argument('--outdir', default='../models/', help='Directory to output results')
    parser.add_argument('--num-epochs', default = 100,type =int, help="Number of training iterations for algorithm")
    parser.add_argument('--num-episodes', default = 0,type =int, help="Number of episodes to train for.")
    parser.add_argument('--num-timesteps', default = 0,type =int, help="Number of timesteps to train for")
    parser.add_argument('--num-workers', default = 4, type = int,help="Number of rollout workers to utilize during training")
    parser.add_argument('--num-gpus', default = 1,type = int, help="Number of GPUs to utilize during training. Generally this is not bottleneck, so 1 is often sufficient")
    parser.add_argument('--checkpoint-epochs', default = 5, type = int, help="Number of epochs before a checkpoints is saved")
    parser.add_argument('--config',default = None,type = str2dict,help = 'JSON string configuration for RLlib training')
    parser.add_argument('--checkpoint-path', default = None, help = "Model directory, if a pretrained system exists already")
    parser.add_argument('--framework', default = "tf2",type=str2framework, help = "Deep learning framework to use on backend. Important that this is one of ['tf2','torch']")
    parser.add_argument('--log-level', default = "info", type=str2log_level, help = "Get logging level from input args: 'info' | 'warn','warning' | 'error' | 'critical' ")
    parser.add_argument('--email-updates', default = True, type=str2bool, help = "Determine if email updates should be sent")
    parser.add_argument('--email-server', default = 'smtp.mail.yahoo.com', help = "Email server to utilize")
    parser.add_argument('--email-sender', default = None,type=str2str, help = "Sender email to utilize (type should match email-server)")
    parser.add_argument('--email-receiver', default = None,type=str2str, help = "Receiver email address (does not need to be of the same type as email-server)")
    parser.add_argument('--tune', default = False, type=str2bool, help = "Determine if the tune wrapper from Ray should be used for training")
    parser.add_argument('--ray-obj-store-mem', default = 3.5e9, type=int, help = "Maximum object store memory for Ray")
    parser.add_argument("--init-ray", default=True,type=str2bool,help="Whether or not to init Ray (may already be running)")
    parser.add_argument("--seed",default=42,type = str2int,
                            help="Random seed for training execution. If not provided, training"
                            "is not seeded and cannot be reproduced.")
    parser.add_argument('--wrappers', default="", type = str2wrapper,  help=
                    """
                    Wrappers to encode the environment observation in different ways.
                    Wrappers will be executed left to right, and the options are as follows:
                           + (example: 'encoding_only,canniballs_one_hot'):
                    - partial_obs.{obs_size}:         Partial observability - must include size (odd int)
                    - encoding_only:                  Provides only the encoded representation of the environment
                    - rgb_only:                       Provides only the RGB screen of environment
                    - canniballs_one_hot              Canniballs specific one-hot
                    """)
    return parser

#######################################################################
# Helper arguments for argparse execution
#######################################################################

def _determine_stopping_criteria(args):
    """Determine stopping criteria based on user input args

    :args: Argparse.Args: User defined arguments

    """
    args.stopping_crit = None
    if args.num_epochs > 0:
        args.stopping_crit = "Epochs"
    elif args.num_episodes > 0:
        args.stopping_crit = "Episodes"
    elif args.num_timesteps > 0:
        args.stopping_crit = "Timesteps"

    assert args.stopping_crit,\
        "ERROR: No stopping criteria set, for training"

def _zero_if_none(s):
    if s is None:
        return 0
    return s

def _keep_going(epochs, timesteps, episodes,
                num_epochs, num_timesteps, num_episodes):
    """Determine whether we've collected enough data"""

    if (num_epochs) and (epochs >= num_epochs):
        return False
    # If num_episodes is set, stop if limit reached.
    elif num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_timesteps and timesteps >= num_timesteps:
        return False

    # Otherwise, keep going.
    return True

def train_agent_tune(agent,
                     args,
                     config):
    """Train agent with tune. Can be useful for hyperparameter
    tuning and memory management

    :agent: Trainable:    Trainable object with Ray API
    :args: Argparse.Args: Input arguments
    :config: Dict:        Configuration dict for agent

    """

    # Get names for everything
    config["env"] = args.env_name
    name = args.env_name
    trial_dirname_lambda = lambda trial: args.tune_dirname
    stop_crit_dict = {"Epochs":["training_iteration",args.num_epochs],
                      "Episodes":['episodes_total',args.num_episodes],
                      "Timesteps":['timesteps_total',args.num_timesteps]}
    sc = stop_crit_dict[args.stopping_crit]

    # Initial save if training fails
    args.config = config
    metadata = {"train":vars(args)}
    _save_metadata(metadata, args.outdir)

    # Run experiment
    tune.run(
        agent,
        config = config,

        # Dynamic setting of these parameters
        stop = {sc[0]:sc[1]},

        # Naming conventions
        local_dir = args.outdir_root + "/tune",
        name = name,
        trial_dirname_creator=trial_dirname_lambda,

        # Can set the resources per trial
        # resources_per_trial = None
        progress_reporter=CameleonRLlibTuneReporter(
            args = args,
            print_intermediate_tables = True),

        #Checkpoint frequency
        checkpoint_freq=args.checkpoint_epochs,

        # Set verbosity, needs to be 1 for messages
        verbose = 1,

        # Restore directory if needed
        restore = args.checkpoint_path,

        # scheduler=None
        checkpoint_at_end=True,
    )

    # Final save after training
    args.config = config
    metadata = {"train":vars(args)}
    _save_metadata(metadata, args.outdir)

def train_agent_standalone(agent,
                           args,
                           config):
    """Train the RL agent on the chosen environment

    :num_epochs: Int:        Number of training epochs
    :checkpoint_epochs: Int: Number of epochs between checkpoints
    :outdir: str:            Output directory for saved models
    :verbose: bool:          If the execution should print to console

    """
    #Status callback
    status = "{:2d} reward min/mean/max {:6.2f}/{:6.2f}/{:6.2f} mean_len {:4.2f} saved {}"

    args.epochs_total = _zero_if_none(agent._iteration)
    args.episodes_total = _zero_if_none(agent._episodes_total)
    args.timesteps_total = _zero_if_none(agent._timesteps_total)

    started = False

    # Iterate through training loop with update messages
    result = None
    while _keep_going(args.epochs_total, args.episodes_total, args.timesteps_total,
                      args.num_epochs, args.num_episodes, args.num_timesteps):

        # Save checkpoint for random agent
        if not started:
            chkpt_file = agent.save(args.outdir)
            started = True

        # Train agent
        result = agent.train()
        chkpt_file = "None"
        args.epochs_total = result['training_iteration']
        args.episodes_total = result['episodes_total']
        args.timesteps_total = result['timesteps_total']
        args.time_total_s = result['time_total_s']

        # Save checkpoint if required
        if ((args.epochs_total % args.checkpoint_epochs) == 0):
            chkpt_file = agent.save(args.outdir)

        #Print if needed
        if (args.epochs_total > 0):
            result['checkpoint_file'] = chkpt_file
            make_progress_update(args,
                                 result)
            sys.stdout.flush()

    # Make final checkpoint
    chkpt_file = agent.save(args.outdir)
    result['checkpoint_file'] = chkpt_file
    make_progress_update(args,
                            result)
    sys.stdout.flush()

def train_agent(agent,
                args,
                config,
                tune = False):
    """Agent training orchestrator

    :args: Argparse.Args: User-defined arguments
    :config: Dict: Configuration for Rllib
    :tune: bool: Boolean on whether or not to use tune

    """

    # Initial save if training fails
    args.config = config
    metadata = {"train":vars(args)}
    _save_metadata(metadata, args.outdir)

    try:
        args.failure_message = None
        if tune:
            train_agent_tune(agent,
                            args,
                            config)
        else:
            assert args.framework in ['tf','torch','tf2'],\
                "ERROR: Framework must be tf, torch or tf2"

            train_agent_standalone(agent,
                                args,
                                config)
    except Exception as e:
        if args.email_updates:
            args.failure_message = str(e)
            args.failure_stacktrace = traceback.format_exc()
            args.execution_type = "training"
            args.mail_bot.send_email("failure",args)
        logging.info("Execution has failed. Terminating")

    # Save metadata after training
    args.config = config
    metadata = {"train":vars(args)}
    _save_metadata(metadata, args.outdir)

def make_progress_update(
                         args,
                         result):

    """Make progress update for training output

    :args: Argparse.Args: User-defined arguments
    :result: Dict: Training output artifact (per iteration)

    """
    # Get total time
    total_time = result.get('time_total_s','')
    current_iter = args.epochs_total

    # Get stopping criteria information
    stop_crit_dict = {"Epochs":[args.num_epochs,result['training_iteration']],
                      "Episodes":[args.num_episodes,result['episodes_total']],
                      "Timesteps":[args.num_timesteps,result['timesteps_total']]}
    sc = stop_crit_dict[args.stopping_crit]

    # Get metrics
    avg_per_training_iteration = round(total_time / current_iter,2)
    avg_per_iteration= total_time / sc[1]
    time_left_s = (sc[0] - sc[1])*avg_per_iteration
    percent_complete = round(sc[1]*100 / sc[0],2)

    time_status = "Model: {}\nEnv: {}\n\nEpoch {:2d} | ETA {} | {:6.2f}% complete | Avg. Epoch {:6.2f} sec.\n".format(
                    args.model_name,
                    args.env_name,
                    current_iter,
                    dt.timedelta(seconds = round(time_left_s)),
                    percent_complete,
                    avg_per_training_iteration)

    reward_status = " - Reward min|mean|max {:6.2f} | {:6.2f} | {:6.2f} - Mean length {:4.2f}"
    reward_status = reward_status.format(
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"]
                    )

    curr_timestamp = dt.datetime.now()
    current_time = " - {} remaining: {}\n\n"\
                    " - Epochs total: {}\n"\
                    " - Episodes total: {}\n"\
                    " - Timesteps total: {}\n\n"\
                    " - Current time: {}\n"\
                    " - Est. datetime at finish: {}\n"\
                    " - Total elapsed {}"\
        .format(args.stopping_crit,
                sc[0] - sc[1],
                result['training_iteration'],
                result['episodes_total'],
                result['timesteps_total'],
                curr_timestamp.strftime("%y-%m-%d %H:%M:%S"),
                (curr_timestamp + dt.timedelta(seconds = round(time_left_s))).strftime("%y-%m-%d %H:%M:%S"),
                dt.timedelta(seconds = round(result['time_total_s'])))


    logging.info(time_status)
    logging.info(current_time)
    logging.info(reward_status)
    logging.info("====="*15)
    sys.stdout.flush()


#######################################################################
# Training orchestration function
#######################################################################

def train(args,parser = None):
    """Train agent

    :args: Argparse.args: User-defined arguments

    """

    # Set logging level
    logging.basicConfig(level= args.log_level,
                        format='%(message)s')

    # Initialize mail bot
    if args.email_updates:
        args.mail_bot = CameleonEmailBot(email_sender = args.email_sender,
                                    email_receiver = args.email_receiver,
                                    email_server = args.email_server)

    # Initialize Ray - and try to prevent OOM
    #Spin up Ray only if it is not already running
    if args.init_ray:
        ray.init(object_store_memory = args.ray_obj_store_mem)

    # Set up environment
    env = gym.make(args.env_name)

    # Wrap environment
    env = wrap_env(env, args.wrappers)

    # Register environment with Ray
    register_env(args.env_name, lambda config: env)

    # Set model and config
    model, config = str2model(args.model_name, config = True)

    #Add to config for compute resources
    config['num_workers'] = args.num_workers
    config['num_gpus'] = args.num_gpus
    config['framework'] = args.framework
    config['seed'] = args.seed
    _determine_stopping_criteria(args)

    #Update config if one was passed
    if args.config:
        config = update_config(config, args.config)

    # Update outdir
    args.outdir_root = args.outdir
    args.outdir = "{}{}_{}_{}_rs{}_w{}_{}".format(args.outdir,
                                    args.model_name,
                                    args.framework,
                                    args.env_name,
                                    args.seed,
                                    args.num_workers,
                                    dt.datetime.now().strftime("%Y.%m.%d"))

    args.tune_dirname = "{}_{}_rs{}_w{}_{}".format(
                                    args.model_name,
                                    args.framework,
                                    args.seed,
                                    args.num_workers,
                                    dt.datetime.now().strftime("%Y.%m.%d"))


    # Set up agent
    agent = model(env = args.env_name,
                config = config,
                logger_creator=cameleon_logger_creator(
                        args.outdir))

    # Change to pretrained model if needed
    if args.checkpoint_path:
        agent.restore(args.checkpoint_path)

    if args.tune:
        agent = args.model_name


    # Train the agent
    train_agent(agent,
                args,
                config,
                tune = args.tune)

    # Shutdown Ray (ensures fresh start for random seeds)
    ray.shutdown()

    # Send email update, if necessary
    if args.email_updates and not args.failure_message:
        args.mail_bot.send_email("train_finished", args)

#######################################################################
# Main method
#######################################################################

def main():
    """Main method for argparse and rllib training

    """

    #Parse all arguments
    parser = create_parser()
    args = parser.parse_args()

    # Train agent
    train(args)


#######################################################################
# Main method
#######################################################################

if __name__ == "__main__":
    main()

