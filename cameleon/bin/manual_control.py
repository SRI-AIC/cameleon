#################################################################################
#
#             Script Title:   Manual Control for game environment (largely for testing)
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################

#######################################################################
# Imports
#######################################################################

import time
import argparse
import numpy as np

import gym
import gym_minigrid.envs
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

import cameleon.envs
from cameleon.utils.parser import str2bool

#######################################################################
# Helper functions
#######################################################################

def redraw(args):
    """
    Redraw the image with rgb rendering for human play
    """
    img = args.env.render('rgb_array', tile_size=args.tile_size)
    args.window.show_img(img)

def reset(args):
    """
    Run environment with specific seed
    """

    # Environment seeding
    if args.seed != -1:
        args.env.seed(args.seed)

    # New observation resets env
    args.obs = args.env.reset()

    # Check if mission exists, and add it to window
    if hasattr(args.env, 'mission'):
        args.window.set_caption(args.env.mission)

    redraw(args)

def step(action,args):
    """
    Wrapper around step function of the environment
    """
    obs, reward, done, info = args.env.step(action)
    # print(info['image'][:,:,2])

    if args.verbose:
        print('step=%s, reward=%.2f' % (args.env.step_count, reward))

    # Print if the game has finished, otherwise redraw the grid
    if done:
        if args.verbose:
            print('\nGame finished!\n')
        reset(args)
    else:
        redraw(args)

def cameleon_key_handler(event,args):
    """
    Key handler for Cameleon games
    """
    if args.verbose:
        print('pressed', event.key)

    if event.key == 'escape':
        args.window.close()
    elif event.key == 'backspace':
        reset(args)
    elif event.key == 'left':
        step(args.env.actions.left,args)
    elif event.key == 'right':
        step(args.env.actions.right,args)
    elif event.key == 'up':
        step(args.env.actions.up,args)
    elif event.key == 'down':
        step(args.env.actions.down,args)
    elif event.key == ' ':
        step(args.args.env.actions.toggle,args)
    elif event.key == 'pageup':
        step(args.env.actions.pickup,args)
    elif event.key == 'pagedown':
        step(args.env.actions.drop,args)
    elif event.key == 'enter':
        step(args.env.actions.done,args)

def minigrid_key_handler(event,args):
    """
    Key handler for Cameleon games
    """
    if args.verbose:
        print('pressed', event.key)

    if event.key == 'escape':
        args.window.close()
    elif event.key == 'backspace':
        reset(args)
    elif event.key == 'left':
        step(args.env.actions.left,args)
    elif event.key == 'right':
        step(args.env.actions.right,args)
    elif event.key == 'up':
        step(args.env.actions.forward,args)
    elif event.key == ' ':
        step(args.env.actions.toggle,args)
    elif event.key == 'pageup':
        step(args.env.actions.pickup,args)
    elif event.key == 'pagedown':
        step(args.env.actions.drop,args)
    elif event.key == 'enter':
        step(args.env.actions.done,args)

def str2key_handler(key_handler):
    """Key handler for game. If custom, a new function must be added
    to this file and included in the "key_handler zoo" in this function

    :key_handler: string defining key_handler
    :returns: key_handler

    """
    kh_zoo = {"cameleon": cameleon_key_handler,
              "minigrid": minigrid_key_handler}

    assert key_handler in kh_zoo, "ERROR: Key handler not recognized"

    return kh_zoo[key_handler]


#######################################################################
# Configure Argparse Parameters
#######################################################################

parser = argparse.ArgumentParser("Manual Control for Cameleon-MiniGrid Environment")

# Args
parser.add_argument("--env-name",
                    required = True,
                    help="Gym environment (including Cameleon or MiniGrid) to load")
parser.add_argument("--seed",type=int,
                    help="Random seed to generate the environment with",default = 42)
parser.add_argument("--tile-size",type=int,
                    help="Size at which to render tiles",default=32)
parser.add_argument("--key-handler",type=str2key_handler,default = "cameleon",
                    help = "Key handler to take in keyboard input from user. Depends on game played")
parser.add_argument('--verbose', default = True, type=str2bool,
                    help = "Determine if output should be sent to console")

#######################################################################
# Main method for Argparse
#######################################################################

def main():
    """
    Main method for running a game manually

    """

    # Parse arguments
    args = parser.parse_args()

    # Make environments
    env = gym.make(args.env_name)
    args.env =env

    #Build window and register key_handler
    window = Window('Cameleon - ' + args.env_name)
    args.window = window
    window.fig.canvas.mpl_connect("key_press_event",
                                  lambda event: args.key_handler(event, args))

    # Reset the game
    reset(args)

    # Blocking event loop
    window.show(block=True)

#######################################################################
# Main
#######################################################################

if __name__ == "__main__":
    main()

