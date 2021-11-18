#################################################################################
#
#             Project Title:  DQN Policy Extractor for RLlib - Cameleon Compatibility
#             Author:         Sam Showalter
#             Date:           2021-07-23
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys

import tensorflow as tf
import torch
from scipy.special import softmax
import numpy as np

from cameleon.policy_extractors.rllib import BaseRLlibPolicyExtractor

#################################################################################
#   RLlib PPO Policy Extractor
#################################################################################

class DQNExtractor(BaseRLlibPolicyExtractor):

    """
    Extracts artifacts from DQN Policy
    to be used in competency analysis
    """

    def __init__(self,
                 to_collect,
                 model,
                 episode,
                 worker,
                 framework,
                 env,
                 episode_start):

        BaseRLlibPolicyExtractor.__init__(self,
                                          to_collect,
                                          model,
                                          episode,
                                          worker,
                                          framework,
                                          env,
                                          episode_start)
        self.q_vals = None

    def get_action_logits(self):
        """Get logits from q function

        :returns: logits

        """
        if self.q_vals is not None:
            return self.q_vals

        self.get_q_function_dist()

        return self.q_vals


    def get_value_function_estimate(self):
        """Still not sure how this is different
        from .value_function, but it is specific to
        the DQN package so might as well keep it

        :returns: state_value_fxn, whatever this is

        """

        vf = self.policy.model.value_function()

        # Make sure it is correct shape
        for s in vf.shape:
            assert (s == 1),\
            "ERROR: Value Function estimate malformed"

        if (self.framework == "tf"):
            vf = tf.reshape(vf,[-1]).eval()[0]

        if (self.framework == "tf2"):
            vf = tf.reshape(vf,[-1]).numpy()[0]

        elif (self.framework == "torch"):
            vf = torch.flatten(vf).detach().cpu().numpy()[0]

        return vf


    def get_action_dist(self):
        """Get Q function distribution

        :returns: Q(a,s)

        """

        if (not self.q_vals):
            self.get_q_values()

        return [softmax(self.q_vals.flatten(),
                       axis = 0)]

    def get_q_values(self):
        """Get Q function distribution

        :returns: Q(a,s)

        """

        if self.q_vals is not None:
            return self.q_vals

        # Returns a list of three items:
        # - (action_scores, logits, dist) if num_atoms == 1, otherwise
        # - (action_scores, z, support_logits_per_action, logits, dist)
        q_vals, _,_ = self.policy.model.get_q_value_distributions(self.model_out)

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                q_vals = q_vals.eval()

        if (self.framework == "tf2"):
            q_vals = q_vals.numpy()

        elif (self.framework == "torch"):
            q_vals = q_vals.detach().cpu().numpy()

        self.q_vals = q_vals
        return q_vals


