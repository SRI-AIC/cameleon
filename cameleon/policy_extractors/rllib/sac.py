#################################################################################
#
#             Project Title:  SAC Policy Extractor for RLlib - Cameleon Compatibility
#             Author:         Sam Showalter
#             Date:           2021-07-23
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys

import numpy as np
import tensorflow as tf
import torch

from cameleon.policy_extractors.rllib import BaseRLlibPolicyExtractor

#################################################################################
#   RLlib PPO Policy Extractor
#################################################################################

class SACExtractor(BaseRLlibPolicyExtractor):

    """Docstring for SACExtractor. """

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
        self.twin_q_vals = None

    def get_action_logits(self):
        """Get logits from base policy prediction
        given the observation

        :returns: logits
        """
        if self.logits is not None:
            return self.logits

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                logits = tf.reshape(self.model_out,[-1]).eval()

        elif (self.framework == "tf2"):
            logits = tf.reshape(self.model_out,[-1]).numpy()

        elif (self.framework == "torch"):
            logits = torch.flatten(self.model_out).detach().cpu().numpy()

        self.logits = logits
        return logits


    def get_value_function_estimate(self):
        """Still not sure how this is different
        from .value_function, but it is specific to
        the DQN package so might as well keep it

        :returns: state_value_fxn, whatever this is

        """

        # Get q value dist if you don't have already
        if not self.q_vals:
            self.get_q_values()

        return np.max(self.q_vals)


    def get_action_dist(self):
        """Get action distribution for
        the current state pi(a|s). Should
        work with either torch or tf2

        (NOT with tf1 and lazy eval, just
        save yourself some grief and use
        eager execution)

        :returns: Current action dist. pi(a|s)

        """

        assert (self.model_out.shape[1] == self.env.action_space.n),\
            "ERROR: Action distribution output does not match action space"

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                logits = tf.reshape(self.model_out,[-1])
                action_dist = tf.nn.softmax(logits).eval()
                logits = logits.eval()

        elif (self.framework == "tf2"):
            logits = tf.reshape(self.model_out,[-1])
            action_dist = tf.nn.softmax(logits).numpy()
            logits = logits.numpy()

        elif (self.framework == "torch"):
            logits = torch.flatten(self.model_out)
            action_dist = torch.nn.functional.softmax(logits,dim = 0).detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

        self.logits = logits
        return [action_dist]

    def get_q_values(self):
        """Get value function estimate for
        current state V(s)

        :returns: Value function estimate V(s)

        """
        # Q values
        q_vals = self.policy.model.get_q_values()

        assert (q_vals.shape[1] == self.env.action_space.n),\
            "ERROR: Q values output does not match action space"

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                q_vals = q_vals.eval()

        if (self.framework == "tf2"):
            q_vals = q_vals.numpy()

        elif (self.framework == "torch"):
            q_vals = q_vals.detach().cpu().numpy()

        self.q_vals = q_vals
        return q_vals


    def get_twin_q_values(self):
        """Get value function estimate for
        current state V(s)

        :returns: Value function estimate V(s)

        """

        # Q values
        twin_q_vals = self.policy.model.get_twin_q_values()

        assert (twin_q_vals.shape[1] == self.env.action_space.n),\
            "ERROR: Twin Q values output does not match action space"

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                twin_q_vals = twin_q_vals.eval()

        if (self.framework == "tf2"):
            twin_q_vals = twin_q_vals.numpy()

        elif (self.framework == "torch"):
            twin_q_vals = twin_q_vals.detach().cpu().numpy()

        self.twin_q_vals = twin_q_vals
        return twin_q_vals


#################################################################################
#   Main Method
#################################################################################



