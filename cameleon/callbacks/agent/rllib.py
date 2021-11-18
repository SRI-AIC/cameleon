#################################################################################
#
#             Project Title:  Callbacks to Collect information from Rllib
#             Author:         Sam Showalter
#             Date:           07-21-2021
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################


from typing import Dict
import argparse
import sys
import copy
import numpy as np
import os
import hashlib

import tensorflow as tf
import torch

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

# Custom imports
from cameleon.policy_extractors.rllib import build_rllib_policy_extractor
from cameleon.utils.general import _write_pkl, _write_hkl,_read_pkl, _read_hkl

#################################################################################
# RLlib Callback Class for Cameleon
#################################################################################


class RLlibIxdrlCallbacks(DefaultCallbacks):
    """
    Callbacks to store information from Cameleon

    """

    def __init__(self,
                 args,
                 config):

        DefaultCallbacks.__init__(self)

        self.args = args
        self.config = config

        self.outdir = args.writer_dir
        self.model = args.model_name
        self.framework = config['framework']
        self.no_frame = args.no_frame
        # self.epochs_trained = self._extract_train_epochs()
        self.epochs_trained =  args.epochs_trained if args.epochs_trained is not None else None
        self.use_hickle = args.use_hickle
        self.write_compressed = _write_hkl if self.use_hickle else _write_pkl
        self.read_compressed = _read_hkl if self.use_hickle else _write_pkl
        self.ext = "hkl" if self.use_hickle else "pkl"

        # This needs to be -1 because of weird issues with
        # the parallel processing of rollouts
        self.episode_num = -1
        self.step_num = 0
        self.reward_total = 0
        self.last_done = False
        self.episode = {}

    def write_episode(self):
        """Write the episode

        """
        #Only keep a truncated version of the hash - still appears sufficient
        assert self.epochs_trained is not None,\
            "ERROR: epochs_trained variable was not set properly"

        obs_hash = hashlib.shake_256(str(self.first_obs).encode()).hexdigest(6)
        self.first_obs = None

        self.episode_id = "{}_cp{}_s{}_r{}_pid{}-{}.{}".format(
                                         obs_hash,
                                         self.epochs_trained,
                                         self.step_num,
                                         str(round(self.reward_total)).replace("-","n"),
                                         self.episode_num,
                                         os.getpid(),
                                         self.ext)

        if self.outdir and len(self.episode) > 0:
            self.write_compressed(self.episode,
                  self.outdir + self.episode_id)

    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, \
            "ERROR: `on_episode_start()` callback should be called right " \
            "after env reset!"


        self.episode = {}
        self.step_num = 0
        self.first_obs = None
        self.reward_total = 0
        self.episode_num += 1

        # Environment observation at start
        self.env = base_env.get_unwrapped()[0]
        obs = self.env.gen_obs()

        # Get the last observation
        self.first_obs = obs

        # Add information for the episode
        self.episode[self.step_num] = {
            "observation": obs,
            "agent_pos": self.env.agent.cur_pos
        }




    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       episode: MultiAgentEpisode, env_index: int, **kwargs):

        # Make sure this episode is ongoing.
        assert episode.length > 0, \
            "ERROR: `on_episode_step()` callback should not be called right " \
            "after env reset!"

        pe = build_rllib_policy_extractor(
                                  self.args.to_collect,
                                  self.model,
                                  episode,
                                  worker,
                                  framework = self.framework,
                                  env = self.env,
                                  episode_start = False)

        # Get reward for total reward and obs
        reward = pe.get_last_reward()
        self.reward_total += reward
        self.last_done = pe.get_last_done()
        info =pe.get_last_info()
        obs =self.episode[self.step_num]['observation']
        agent_pos =self.episode[self.step_num]['agent_pos']


        if self.no_frame:
            info['frame'] = None

        # Add information for the episode
        self.episode[self.step_num] = {
            "observation": obs,
            "agent_pos": agent_pos,
            "action":pe.get_last_action(),
            "reward":reward,
            "done": self.last_done,
            "info":info,
        }

        # Items to collect beyond standard info
        for item in self.args.to_collect:
            self.episode[self.step_num][item] = pe.call_method(item)

        # "pi_info":pe.get_last_pi_info(),
        # "value_function":pe.get_value_function_estimate(),
        # "action_dist" : pe.get_action_dist(),
        # "q_values": pe.get_q_values(),
        # "action_logits":pe.get_action_logits()

        self.step_num += 1

        # Add information for the next episode observation
        self.episode[self.step_num] = {
            "observation": episode.last_observation_for(),
            "agent_pos":self.env.agent.cur_pos
        }


    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # Write episode and delete last observation
        # since there was no action taken, roll back steps
        del self.episode[self.step_num]

        # self.step_num -=1
        self.write_episode()

#######################################################################
# Main method
#######################################################################

