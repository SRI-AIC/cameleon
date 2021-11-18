#################################################################################
#
#             Project Title:  Train several agents, potentially on many environments
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
import copy
import logging
import datetime as dt
import subprocess
import argparse

# Gym stuff
import gym
import gym_minigrid.envs
from gym import wrappers as gym_wrappers

# Custom imports
import cameleon.envs
from cameleon.utils.parser import str2bool, str2int, str2list, str2dict
from cameleon.bin.train import create_optional_args, train


#################################################################################
#   Function-Class Declaration
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
    required_named.add_argument("--model-names",type=str2list,required = True,
                        help="Model list from which to train agents on environments. May include multiple mode")
    required_named.add_argument("--env-names",type=str2list,required = True,
                        help="The environment(s) specifier to use (may include multiple as comma-delimited string). This could be an openAI gym "
                        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
                        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")

    return create_optional_args(parser)

#######################################################################
#    Functions for preprocessing input and validting config
#######################################################################

def _run_train_subexperiment(args,parser):
    """Run single train as part of larger
    experiment

    :args: Argparse.Args: User-defined arguments

    """

    train(args,
          parser = parser)

def run_train_experiment(master_args,parser):
    """Run full set of train sub_experiments

    :args: Argparse.Args: User-defined args

    """

    # # Start by initializing Ray
    # init_ray = True

    # Set logging level
    logging.basicConfig(level=master_args.log_level,
                        format='%(message)s')

    sync_dirs = []
    orig_outdir = master_args.outdir
    for model in master_args.model_names:
        master_args.model_name = model

        for env in master_args.env_names:
            args = copy.deepcopy(master_args)
            args.env_name = env
            args.outdir = orig_outdir

            # Initialize Ray only for first iteration
            # args.init_ray = init_ray
            # if init_ray:
            #     init_ray = False

            # Run the sub-experiment
            logging.info("=========="*7)
            logging.info("Running training experiment for model {} on env {}".format(model,env))
            logging.info("=========="*7)
            _run_train_subexperiment(args,parser)

#######################################################################
# Main method for full experiment
#######################################################################

def main():
    """Run execution

    """
    parser = create_parser()
    args = parser.parse_args()

    # Run rollout experiment
    run_train_experiment(args,parser)

#######################################################################
# Run main method
#######################################################################

if __name__ == "__main__":
    main()








