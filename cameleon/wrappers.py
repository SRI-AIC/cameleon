#################################################################################
#
#             Project Title:  Wrappers for Cameleon Environments
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#    Wraps the observation space of the environment and passes it to agent to train
#    NOTE: ALL wrappers implemented here must support returning their observation with .gen_obs()
#    ALSO: Wrappers that reduce the amount of information present (RGB, ENC only) should be wrapped LAST
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import copy
import hashlib
import sys
import numpy as np

import gym
from gym import error, spaces, utils
from gym_minigrid.wrappers import *

from cameleon.grid import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX
from cameleon.utils.general import _tup_equal, _write_pkl, _read_pkl, _write_hkl, _read_hkl

#################################################################################
#  Parial Observability Wrapper
#################################################################################

class RGBImgObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as the only observation output,
    no language/mission. This can be used to have the agent to solve the
    gridworld in pixel space.
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        self.observation_space.spaces['image'] = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.width * tile_size, self.env.height * tile_size, 3),
            dtype='uint8'
        )

    def observation(self, obs):
        env = self.unwrapped

        rgb_img = env.render(
            mode='rgb_array',
            highlight=False,
            tile_size=self.tilesize
        )

        return {
            'mission': obs['mission'],
            'image': rgb_img}

    def gen_obs(self):
        """Generate observation

        """
        obs = self.env.gen_obs()
        return self.observation(obs)

    def __str__(self):
        return "rgb_only"

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space.spaces['image']

    def gen_obs(self):
        """Generate observation

        """
        obs = self.env.gen_obs()
        return self.observation(obs)

    def observation(self, obs):
        return obs['image']

    def __str__(self):
        return "encoding_only"

class PartialObsWrapper(gym.core.Wrapper):

    """Partial Observability Wrapper

    Regular gym_minigrid made the design choice of
    having all observations be partially observable by
    default. We feel this is a flawed method of approach, and
    instead have opted to always provide the fully observable
    state, which can then be altered by a wrapper. This ensures
    that the functionality of the state and observation space
    is environment agnostic

    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        self.agent_view_size = agent_view_size
        self.shift = self.agent_view_size // 2

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })


    def __str__(self):
        return "partial_obs"

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_obs_view_location(self):
        """Get location of observation box

        """
        agent_pos = self.agent.cur_pos

        # Top left of obs box
        top_left = (max(0,agent_pos[0] - self.shift),
               max(0, agent_pos[1] - self.shift))

        # Bottom right of obs box
        bottom_right = (min(top_left[0] + self.agent_view_size, self.width),
                        min(top_left[1] + self.agent_view_size, self.height))

        return top_left, bottom_right


    def get_obs_view(self, obs):
        """Returns partial observability slice of observation
        space. Pads space with zeros when there is an incomplete
        observation

        :obs: Current observation space of agent
        :top_left: Top left coordinates of view slice
        :bottom_right: Bottom right coordinates of view slice

        :returns: Observation for agent

        """

        top_left, bottom_right = self.get_obs_view_location()

        # Reduce view if in top corner and overflowing the grid
        view_reduction_x = self.shift - (self.agent.cur_pos[0] - top_left[0])
        view_reduction_y = self.shift - (self.agent.cur_pos[1] - top_left[1])


        obs_view = obs[top_left[0]: bottom_right[0] - view_reduction_x,
                       top_left[1]: bottom_right[1] - view_reduction_y,
                       :]

        obs_shape_wh = obs_view.shape[:2]
        obs_shape_d = obs_view.shape[2]
        if not _tup_equal((self.agent_view_size, self.agent_view_size),
                          obs_shape_wh):
            missing_x = self.agent_view_size - obs_shape_wh[0]
            missing_y = self.agent_view_size - obs_shape_wh[1]

            if missing_x:
                x_fill = np.zeros((missing_x,obs_shape_wh[1], obs_shape_d))
                x_fill[:,:,0] = OBJECT_TO_IDX[None]
                x_fill[:,:,1] = COLOR_TO_IDX[None]
                x_fill[:,:,2] = 0

                # Add to correct side of observation
                if (top_left[0] == 0):
                    obs_view = np.concatenate((x_fill,obs_view), axis = 0)

                else:
                    obs_view = np.concatenate((obs_view,x_fill), axis = 0)

            if missing_y:
                y_fill = np.zeros((self.agent_view_size, missing_y, obs_shape_d))
                y_fill[:,:,0] = OBJECT_TO_IDX[None]
                y_fill[:,:,1] = COLOR_TO_IDX[None]
                y_fill[:,:,2] = 0

                # Add to correct side of observation
                if (top_left[1] == 0):
                    obs_view = np.concatenate((y_fill,obs_view), axis = 1)

                else:
                    obs_view = np.concatenate((obs_view, y_fill), axis = 1)

            assert _tup_equal(obs_view.shape, (self.agent_view_size, self.agent_view_size)), "Error: Observation malformed"

        return obs_view

    def gen_obs(self):
        """Generate observation

        """
        obs = self.env.gen_obs()
        return self.get_obs_view(obs['image'])

    def step(self, action):

        #Get full observability first
        obs, reward, done, info = self.env.step(action)

        # Filter and correct observation
        info['env'] = copy.deepcopy(obs['image'])
        # info['frame'] = copy.deepcopy(self.env.grid.img)
        obs['image'] = self.get_obs_view(obs['image'])

        return obs, reward, done, info

#######################################################################
# One Hot Wrapper
#######################################################################


class CanniballsOneHotWrapper(gym.core.ObservationWrapper):

    """Observation wrapper to improve learning for Canniballs"""

    def __init__(self,env):
        super().__init__(env)
        self.observation_space.spaces["image"] = spaces.Box(
            low=0,
            high=1,
            shape=(self.env.width, self.env.height, 4),  # number of cells
            dtype='uint8'
        )

    def __str__(self):
        return "canniballs_one_hot"

    def build_one_hot(self, obs):
        """Build one-hot from observation

        :obs: np.ndarray:     Tensor encoding of env
        :returns: np.ndarray: One-hot encoding of env

        """
        # Build scores
        agent_pos = self.env.agent.cur_pos

        #Build array with agent score
        semantics =obs[:,:,0]
        scores =obs[:,:,2]
        is_agent = (semantics == OBJECT_TO_IDX['agent']).astype(np.uint8)

        agent_score = self.env.agent.score

        #Make food and opponents
        is_food_opponent = ((semantics == OBJECT_TO_IDX['food'])|
                            (semantics == OBJECT_TO_IDX['ball'])).astype(np.uint8)

        is_opponent = (is_food_opponent &
                      (scores >= agent_score)).astype(np.uint8)

        is_food = (is_food_opponent &
                   (scores < agent_score)).astype(np.uint8)

        #Identify obstacles
        is_obstacle = (semantics == OBJECT_TO_IDX['wall']).astype(np.uint8)

        # Remove agent score
        scores[agent_pos[0],agent_pos[1]] = 0

        obs =  np.stack((is_agent,
                               is_opponent,
                               is_food,
                               is_obstacle),
                        axis = -1)
        return obs

    def step(self, action):
            observation, reward, done, info = self.env.step(action)
            info['env'] = observation['image']
            info['frame'] = copy.deepcopy(self.env.grid.img)
            return self.observation(observation), reward, done, info

    def gen_obs(self):
        """Generate observation

        """
        obs = self.env.gen_obs()
        return self.observation(obs)

    def observation(self, obs):
        """Step function for canniballs

        :obs: Observation

        """

        # Filter and correct observation
        obs['image'] = self.build_one_hot(obs['image'])

        # Return information
        return obs

##################################################################################
##   Deprecated
##################################################################################


class EpisodeWriterWrapper(gym.core.Wrapper):

    """Wrapper to record environment and agent interactions
    for later use with CAML analysis with RANDOM AGENTS. This
    is NOT the preferred method of collecting rollouts - use
    callbacks with an early checkpoint instead

    """

    def __init__(self,env, args = None):
        super().__init__(env)
        self.outdir = args.writer_dir
        self.args = args
        # This needs to be -1 because of weird issues with
        # the parallel processing of rollouts
        self.episode_num = -1
        self.reward_total = 0
        self.rollout = {}
        self.obs_hash = None

        self.write_compressed = _write_hkl if args.use_hickle else _write_pkl
        self.read_compressed = _read_hkl if args.use_hickle else _read_pkl
        self.ext = "hkl" if args.use_hickle else "pkl"


    def write_episode(self):
        """Write the episode

        """

        # and len(self.rollout) > 0
        self.episode_id = "{}_ep0_s{}_r{}_pid{}-{}.{}".format(
                                         self.obs_hash,
                                         self.env.step_count,
                                         str(round(self.reward_total)).replace("-","n"),
                                         self.episode_num,
                                         os.getpid(),
                                         self.args.ext)

        if self.outdir and len(self.rollout) > 0:
            self.write_compressed(self.rollout,
                  self.outdir + self.episode_id)

    def reset_writer(self):
        """Reset writer for new episode

        """
        self.write_episode()
        self.episode_num += 1
        self.step_num = 0
        self.reward_total = 0

    def reset(self, **kwargs):
        """
        Record a new episode when last one terminates
        """

        self.reset_writer()
        self.rollout = {}

        obs = self.env.reset(**kwargs)
        self.obs_hash = hashlib.shake_256(str(obs).encode()).hexdigest(6)
        self.rollout[self.env.step_count] = {"observation":obs}
        return obs

    def step(self, action):
        """Action taken for the last state of
        the env

        :action: action_space.Action: Action taken by agent

        """

        #Get information
        obs, reward, done, info = self.env.step(action)
        # print("\n\n\n\n")
        # print(self.agent)
        # sys.exit(1)
        step = self.env.step_count
        self.reward_total += reward

        #Store rollout information
        self.rollout[step-1]["reward"] = reward
        self.rollout[step-1]["done"] = done
        self.rollout[step-1]["action"] = action
        self.rollout[step-1]["info"] = copy.deepcopy(info)

        # Store new observation if not done
        if not done:
            self.rollout[step] = {"observation":obs}

        # Return necessary information
        # No need to keep sending info around
        # can take that out of memory
        return obs, reward, done, info

#######################################################################
# Main
#######################################################################

