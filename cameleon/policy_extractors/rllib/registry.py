#################################################################################
#
#             Project Title:  Policy Extraction General Class for Cameleon, Abstracts
#                             policies away just like RLlib (wrapper class)
#             Author:         Sam Showalter
#             Date:           2021-07-22
#
#################################################################################

# If you want up to date keys and values
# Also a safety check
from ray.rllib.agents.registry import ALGORITHMS

from cameleon.policy_extractors.rllib.ppo import PPOExtractor
from cameleon.policy_extractors.rllib.appo import APPOExtractor
from cameleon.policy_extractors.rllib.dqn import DQNExtractor
from cameleon.policy_extractors.rllib.a2c import A2CExtractor
from cameleon.policy_extractors.rllib.a3c import A3CExtractor
from cameleon.policy_extractors.rllib.pg import PGExtractor
from cameleon.policy_extractors.rllib.impala import IMPALAExtractor

#################################################################################
# Agent roster for objects
#################################################################################

# Taken from RLlib template
EXTRACTORS = {
    "A2C": A2CExtractor,
    "A3C": A3CExtractor,
    "APEX": DQNExtractor,
    # "APEX_DDPG": _import_apex_ddpg,
    "APPO": APPOExtractor,
    # "ARS": _import_ars,
    # "BC": _import_bc,
    # "CQL": _import_cql,
    # "ES": _import_es,
    # "DDPG": _import_ddpg,
    # "DDPPO": _import_ddppo,
    "DQN": DQNExtractor,
    # "SlateQ": _import_slate_q,
    # "DREAMER": _import_dreamer,
    "IMPALA": IMPALAExtractor,
    # "MAML": _import_maml,
    # "MARWIL": _import_marwil,
    # "MBMPO": _import_mbmpo,
    "PG": PGExtractor,
    "PPO": PPOExtractor,
    # "QMIX": _import_qmix,
    "R2D2": DQNExtractor,
    "DDPPO": PPOExtractor,
    # "SAC": _import_sac,
    # "SimpleQ": _import_simple_q,
    # "TD3": _import_td3,
}

# AGENT_CATEGORY_DICT = {"policy_gradient":{"PG","A2C","A3C","PPO","APPO","SAC","IMPALA"},
#                           "imitation_learning":{"BC","MARWIL"},
#                           "evolution":{"ES"},
#                           "q_learning":{"DQN","APEX","RAINBOW","SimpleQ", "SlateQ","CQL","R2D2"},
#                           "model_based":{"DREAMER","MBMPO"}}

# Make sure that all these keys match
def validate_algo_config():
    """Validate Cameleon config against
    what exists in RLlib. Must be a subset

    """

    for e in EXTRACTORS.keys():
        assert e in ALGORITHMS,\
            "ERROR: Extractor {} not found in RLlib registry".format(e)

# Validate configuration immediately
validate_algo_config()

