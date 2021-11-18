################################################################################
#
#             Project Title:  Speed benchmarking for Cameleon Games
#             Author:         Sam Showalter
#             Date:           2021-07-13
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

from tqdm import tqdm
import logging
import time
import argparse

import gym
import gym_minigrid
import gym.envs
from gym_minigrid.wrappers import *

import cameleon.envs
from cameleon.wrappers import *
from cameleon.utils.parser import str2wrapper, str2bool, str2log_level
from cameleon.utils.env import wrap_env


#################################################################################
# Define argument parser arguments
#################################################################################


parser = argparse.ArgumentParser("Speed Benchmark for Cameleon Gym Games")
parser.add_argument("--env-name",required = True,help="Cameleon or gym environment to load")
parser.add_argument("--num-resets", default=1000, type = int, help = "Number of environment resets to time for reset rendering test")
parser.add_argument("--num-viz-frames", default=1000, type = int, help = "Number of frame to examine for visual rendering test")
parser.add_argument("--num-enc-frames", default=5000, type = int, help = "Number of frame to examine for encoded rendering test")
parser.add_argument("--visual", default=False, type = str2bool, help="Whether or not to check visual rendering")
parser.add_argument('--log-level', default = "info", type=str2log_level, help = "Get logging level from input args: 'info' | 'warn','warning' | 'error' | 'critical' ")
parser.add_argument('--wrappers', default="encoding_only", type = str2wrapper,  help=
                  """
                  Wrappers to encode the environment observation in different ways. Wrappers will be executed left to right, and the options are as follow:
                  - partial_obs.{obs_size}:         Partial observability - must include size (odd int)
                  - encoding_only:                  Provides only the encoded representation of the environment
                  - rgb_only:                       Provides only the RGB screen of environment
                  - canniballs_one_hot              Canniballs specific one-hot
                  - episode_writer                  Writes episodes for agent
                  """)
#######################################################################
# Helper functions
#######################################################################


# Benchmark env.reset
def reset_rendering_test(args):
    """Reset rendering speed test

    :args: Argparse arguments
    :returns: reset rendering speed (ms)

    """
    t0 = time.time()
    logging.info("Reset rendering speed test")
    for i in tqdm(range(int(args.num_resets))):
        args.env.reset()
    t1 = time.time()
    dt = t1 - t0
    reset_time = (1000 * dt) / args.num_resets
    return reset_time

def visual_rendering_test(args):
    """Visual rendering speed test. This is how fast
    the system can cycle for a human with matplotlib

    :args: Argparse arguments
    :returns: Visual rendering speed (FPS)

    """
    # Benchmark rendering
    logging.info("Visual Rendering")
    t0 = time.time()
    for i in tqdm(range(args.num_viz_frames)):
        args.env.render('rgb_array')
    t1 = time.time()
    dt = t1 - t0
    frames_per_sec = args.num_viz_frames / dt
    return frames_per_sec

def encoded_rendering_test(args):
    """Encoded representation rendering test

    :args: Argparse arguments
    :returns: encoded rendering speed (FPS)

    """

    t0 = time.time()

    #Benchmark agent view
    logging.info("Encoded Rendering Test")

    args.env.reset()
    for i in tqdm(range(args.num_enc_frames)):
        obs, reward, done, info = args.env.step(2)

        if done:
            args.env.reset()

    t1 = time.time()
    dt = t1 - t0
    agent_view_fps = args.num_enc_frames / dt

    return agent_view_fps


#################################################################################
#   Main Method
#################################################################################


def main():
    """Main method for benchmarking speed of environment

    """
    # Init argparse args
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.log_level,
                        format='%(message)s')

    #Build environment
    env = gym.make(args.env_name)

    # Wrap environment
    env = wrap_env(env, args.wrappers)
    args.env = env

    # Run speed tests
    reset_time = reset_rendering_test(args)
    enc_fps = encoded_rendering_test(args)
    viz_fps = None
    if args.visual:
        viz_fps = visual_rendering_test(args)


    # print a summary
    logging.info('\nEnv reset time: {:.1f} ms'.format(reset_time))
    logging.info('Encoded Env FPS: {:.0f}'.format(enc_fps))
    if args.visual:
        logging.info('Visual rendering FPS : {:.0f}'.format(viz_fps))

#######################################################################
# Main
#######################################################################

if __name__ == "__main__":
    main()


