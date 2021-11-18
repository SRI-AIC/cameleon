#################################################################################
#
#             Project Title:  Rollout several agents, potentially over time
#             Author:         Sam Showalter
#             Date:           2021-08-16
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import re
import glob
import logging
import datetime as dt
import subprocess
import copy
import shutil
import argparse
from pathlib import Path

# Gym stuff
import gym
import gym_minigrid.envs
from gym import wrappers as gym_wrappers

# Custom imports
import cameleon.envs
from cameleon.utils.parser import str2bool, str2int, str2list, str2dict
from cameleon.bin.rollout import create_optional_args,rollout


#################################################################################
#  Create argument parser
#################################################################################

def create_parser(parser_creator = None):
    """Create parser for rollout agent experiments
    :returns: Argparser.Args: User-defined arguments

    """

    parser_creator = parser_creator or argparse.ArgumentParser

    parser = parser_creator(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description="Roll out several agents together on the same settings, perhaps across timesteps")
    # Required arguments
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("--experiment-config",type=str2dict,required = True,
                        help="Checkpoint dictionary from which to roll out. May include multiple checkpoints and timesteps")
    required_named.add_argument("--env-names",type=str2list,required = True,
                        help="The environment(s) specifier to use (may include multiple as comma-delimited string). This could be an openAI gym "
                        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
                        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")

    parser = create_optional_args(parser)
    parser.add_argument("--sync-across-models",type = str2bool,default = True,
                        help = "Whether or not to sync rollouts between models for environments")
    return parser


#######################################################################
#    Functions for preprocessing input and validting config
#######################################################################

def _validate_dir(directory):
    """Validate directory exists

    :directory: Make sure directory exists
    :returns: bool: Whether or not dir exists

    """
    root = Path(directory).parent
    return (os.path.exists(directory) and
            os.path.isdir(directory) and
            os.path.exists("{}/metadata.json".format(root)))

def _validate_dirs(args):
    """Validate all directories given by argparse
    ahead of time

    :args: Argparse.Args: User-defined arguments

    """
    for model in args.experiment_config.keys():
        sub_experiment_batch = args.experiment_config[model]
        chkpt_root = sub_experiment_batch['checkpoint_root']
        checkpoints = sub_experiment_batch['checkpoints']
        for c in checkpoints:
            assert _validate_dir("{}/checkpoint_{:06}".format(chkpt_root,c)),\
                "ERROR: Directory for model and/or metadata.json not found:\n - {}"\
                .format("{}/checkpoint_{:06}".format(chkpt_root,c))

def _run_rollout_subexperiment(args,parser):
    """Run single rollout as part of larger
    experiment

    :args: Argparse.Args: User-defined arguments

    """
    rollout(args,parser)

def _sync_across_dirs(directories,
                      env):
    """Sync all rollouts across directories
    so everything directly alignes

    :directories: List of output directories

    """
    logging.info("----------"*7)
    logging.info(f"Synchronizing rollouts to match across models for env {env}")
    logging.info("----------"*7)
    rollout_bundle_set = None
    rmdirs = {}
    for directory in directories:
        bundle = set([d for d in os.listdir(directory) if os.path.isdir(directory+d)])
        rmdirs[directory] = bundle

        if not rollout_bundle_set:
            rollout_bundle_set = bundle
        else:
            rollout_bundle_set = (rollout_bundle_set & bundle)

    for directory,bundle in rmdirs.items():
        for b in bundle:
            if b not in rollout_bundle_set:
                logging.info("Removing {}".format(directory +b))
                shutil.rmtree(directory + b)

def sync_bundled_data(writer_dir,
                 ext = "hkl"):
    """Sync rollout dirs across runs to ensure
    that there is no missing data

    :rollout_dirs: List of final rollout directories
    :ext: Filename extension

    """
    # Get all directories
    rollout_dirs = [d for d in os.listdir(writer_dir) if os.path.isdir(writer_dir + d)]
    max_files =0

    # Find max data in all bundle folders
    for d in rollout_dirs:
        cur_files =len(list(glob.glob(writer_dir + d+ "/*.{}".format(ext))))
        max_files = max(max_files, cur_files)

    # Remove any folders with missing data
    for d in rollout_dirs:
        cur_files =len(list(glob.glob(writer_dir + d + "/*.{}".format(ext))))
        if cur_files < max_files:
            shutil.rmtree(writer_dir + d)

def run_rollout_experiment(master_args,parser):
    """Run full set of rollout sub_experiments

    :args: Argparse.Args: User-defined args

    """

    # Start by initializing Ray
    init_ray = True

    # Set logging level
    logging.basicConfig(level=master_args.log_level,
                        format='%(message)s')

    # Validate given directories
    logging.info("Validating directories")
    _validate_dirs(master_args)
    timestamp = dt.datetime.now().strftime("%Y.%m.%d")

    for env in master_args.env_names:
        sync_dirs = []
        master_args.env_name = env

        for model in master_args.experiment_config.keys():
            args = copy.deepcopy(master_args)
            sub_experiment_batch = args.experiment_config[model]
            chkpt_root = sub_experiment_batch['checkpoint_root']
            checkpoints = sub_experiment_batch['checkpoints']
            args.model_name = model

            sync_dir = "{}{}_{}_{}_ep{}_ts{}_rs{}_w{}_{}/"\
                            .format(args.outdir,
                                    args.model_name,
                                    args.framework,
                                    args.env_name,
                                    args.num_episodes,
                                    args.num_timesteps,
                                    args.seed,
                                    args.num_workers,
                                    timestamp)

            sync_dirs.append(sync_dir)

            for c in checkpoints:

                # Initialize Ray only for first iteration
                args.init_ray = init_ray
                if init_ray:
                    init_ray = False

                # Gather rest of information
                path_suffix = 'checkpoint_{0:06}/checkpoint-{0}'.format(c)
                args.checkpoint_path = chkpt_root + path_suffix

                # Run the sub-experiment
                logging.info("=========="*7)
                logging.info("Running rollout experiment for:\n - Model: {}\n - Checkpoint: {}\n - Environment: {}".format(model,c,env))
                logging.info("=========="*7)
                _run_rollout_subexperiment(args,parser)

        # Sync rollouts across directories
        # Only across models, not envs
        _sync_across_dirs(sync_dirs,env)

#######################################################################
# Main method for full experiment
#######################################################################

def main():
    """Run execution

    """
    parser = create_parser()
    args = parser.parse_args()

    # Run rollout experiment
    run_rollout_experiment(args,parser)

#######################################################################
# Run main method
#######################################################################

if __name__ == "__main__":
    main()








