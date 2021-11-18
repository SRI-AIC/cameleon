#################################################################################
#
#             Project Title:  APPO Policy Extractor for RLlib - Cameleon Compatibility
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

from cameleon.policy_extractors.rllib import BaseRLlibPolicyExtractor

#################################################################################
#   RLlib PPO Policy Extractor
#################################################################################

class APPOExtractor(BaseRLlibPolicyExtractor):

    """Docstring for APPOExtractor. """

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

    def get_value_function_estimate(self):
        """Get value function estimate for
        current state V(s)

        :returns: Value function estimate V(s)

        """

        # Get value function
        vf = self.policy.model.value_function()

        # Make sure it is correct shape
        for s in vf.shape:
            assert (s == 1),\
            "ERROR: Value Function estimate malformed"

        if (self.framework == "tf"):
            with self.policy._sess.as_default():
                vf = tf.reshape(vf,[-1]).eval()[0]
        elif (self.framework == "tf2"):
            vf = tf.reshape(vf,[-1]).numpy()[0]
        elif (self.framework == "torch"):
            vf = torch.flatten(vf).detach().cpu().numpy()[0]

        return vf





#################################################################################
#   Main Method
#################################################################################



