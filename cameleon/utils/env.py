
#################################################################################
#
#             Project Title:  General utilities for Cameleon Env
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import importlib

#Import information from Gym minigrid
from gym_minigrid.rendering import *

# Ray registry for RLlib
from ray.tune.logger import Logger, UnifiedLogger

# Import information from cameleon
from cameleon.base_objects import *
from cameleon.envs import *

#################################################################################
#   Function-Class Declaration
#################################################################################

def _render_tile(obj,
                tile_size=32,
                subdivs=3):
    """
    Render a tile and cache the result

    :obj: WorldObj: Object to place
    :highlight: bool: Whether or not to highlight
    :tile_size: Int: Render tile size
    :subdivs: Int: Subdivisions
    """

    # Hash map lookup key for the cache
    # Need to look into this hash map
    img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

    # Draw the grid lines (top and left edges)
    fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
    fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

    if obj != None:
        obj.render(img)

    # Downsample the image to perform supersampling/anti-aliasing
    img = downsample(img, subdivs)

    return img

def render_encoded_env(
    encoded_env,
    tile_size=32,
    subdivs = 3):
    """
    Render this grid at a given scale

    :tile_size: Int:             Render tile size
    :highlight_mask: np.ndarray: Highlight mask

    """

    # Compute the total grid size
    width, height = encoded_env.shape[:2]
    width_px = width * tile_size
    height_px = height * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for j in range(0, height):
        for i in range(0, width):
            obj = IDX_TO_INIT_OBJECT[encoded_env[i,j,0]]
            if obj:
                obj.color = IDX_TO_COLOR[encoded_env[i,j,1]]

            tile_img = _render_tile(
                obj,
                tile_size=tile_size,
                subdivs = subdivs
            )

            ymin = j * tile_size
            ymax = (j+1) * tile_size
            xmin = i * tile_size
            xmax = (i+1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img


def cameleon_logger_creator(custom_path):
    """
    Customize way that cameleon saves off information

    :custom_path: Custom logging path

    """
    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = custom_path
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def wrap_env(env,wrappers):
    """Wrap environment

    :env: Environments
    :returns: Environment

    """
    for i in range(len(wrappers)):
        if isinstance(wrappers[i], tuple):
            env = wrappers[i][0](env,
                       agent_view_size = wrappers[i][1])
        else:
            env = wrappers[i](env)

    return env

def load_env(env_id):

    # Get environment spec
    env_spec = gym.envs.registry.env_specs[env_id]

    #Get entry point
    env_spec_name = env_spec.entry_point

    # Load the environment
    mod_name, attr_name = env_spec_name.split(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)

    return fn()

#######################################################################
# Main
#######################################################################

