#################################################################################
#
#             Project Title:  Recording Engine for Collecting Episodes from Cameleon
#             Author:         Sam Showalter
#             Date:           2021-07-14
#
#             Source: This was taken from rollout.py in RLlib and altered slightly.
#                     Original file found here:
#                     https://github.com/ray-project/ray/blob/master/rllib/rollout.py
#
#################################################################################

# General stuff
import argparse
import collections
import copy
import shutil
import glob
import logging
import re
import os
import time
import sys
from datetime import datetime as dt
import numpy as np
from tqdm import tqdm

# Ray stuff
import ray
import ray.cloudpickle as cloudpickle
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.utils import merge_dicts
from ray.tune.registry import get_trainable_cls, _global_registry, ENV_CREATOR,register_env

# Gym stuff
import gym
import gym_minigrid.envs
from gym import wrappers as gym_wrappers

# Custom imports
import cameleon.envs
from cameleon.wrappers import EpisodeWriterWrapper
from cameleon.utils.env import load_env, wrap_env, cameleon_logger_creator
from cameleon.utils.parser import str2bool, str2model, str2int, str2list,\
    str2wrapper, str2str, str2dict, str2framework, str2log_level
from cameleon.utils.general import _read_hkl, _write_hkl, _read_pkl, _load_metadata, _save_metadata
from cameleon.callbacks.agent.rllib import RLlibIxdrlCallbacks

#######################################################################
# Create parser
#######################################################################

def create_parser(parser_creator=None):
    """
    Create argparse argument list
    """
    parser_creator = parser_creator or argparse.ArgumentParser

    parser = parser_creator(formatter_class=argparse.RawDescriptionHelpFormatter,
                            description="Roll out a reinforcement learning agent given a checkpoint.")

    # Required arguments
    required_named = parser.add_argument_group("required named arguments")
    required_named.add_argument("--checkpoint-path",type=str2str,required = True,
                        help="Checkpoint from which to roll out. MUST be provided to run competency analysis")
    required_named.add_argument("--env-name",type=str,required = True,
                        help="The environment specifier to use. This could be an openAI gym "
                        "specifier (e.g. `CartPole-v0`) or a full class-path (e.g. "
                        "`ray.rllib.examples.env.simple_corridor.SimpleCorridor`).")

    parser = create_optional_args(parser)

    return parser


def create_optional_args(parser):
    """Add optional args to parser

    """
    # Optional arguments
    parser.add_argument("--local-mode",default = False,action="store_true",
                        help="Run ray in local mode for easier debugging.")
    parser.add_argument("--model-name",type=str2str,default="",
                        help="Only needed if checkpoint not provided. User must define in this case")
    parser.add_argument("--no-render",default=True,type=str2bool,
                        help="Suppress rendering of the environment.")
    parser.add_argument("--use-hickle",default=False,type=str2bool,
                        help="Use gzip hickle over more standard pickle compression")
    parser.add_argument("--no-frame",default=True,type=str2bool,
                        help="Whether or not to store frames from rollouts. Can be a huge memory burden")
    parser.add_argument("--framework",default='torch',type=str2framework,
                        help="Framework for model in which rollouts given. This should be provided by checkpoint and not user in most cases. Config will overwrite what user provides.")
    parser.add_argument("--store-video",type=str2bool,default=True,
                        help="Specifies the directory into which videos of all episode "
                        "rollouts will be stored.")
    parser.add_argument("--num-timesteps",default=0,type = int,
                        help="Number of timesteps to roll out. Rollout will also stop if "
                        "`--num-episodes` limit is reached first. A value of 0 means no "
                        "limitation on the number of timesteps run.")
    parser.add_argument("--seed",default=42,type = str2int,
                        help="Random seed for rollout execution. If not provided, excecution"
                        "is not seeded and cannot be reproduced.")
    parser.add_argument("--num-episodes",default=0,type = int,
                        help="Number of complete episodes to roll out. Rollout will also stop "
                        "if `--num-timesteps` limit is reached first. A value of 0 means "
                        "no limitation on the number of episodes run.")
    parser.add_argument("--to-collect", default="action_dist,action_logits,value_function",type=str2list,
                        help="Items to collect from rollouts beyond obs, action, reward, done")
    parser.add_argument("--outdir", default='rollouts/',
                        help="Output directory for rollouts")
    parser.add_argument("--imago-dir", default="data/imago/",
                        help="Output filepath for imago data")
    parser.add_argument("--bundle-only", default=False,type=str2bool,
                        help="Option to just bundle existing rollouts. If chosen, a filepath needs to be specified")
    parser.add_argument("--bundle-only-dir", default=None,type=str2str,
                        help="If only bundling of existing rollouts needed, provide access to appropriate directory")
    parser.add_argument("--sync-bundles", default=False,type=str2bool,
                        help="Syncs up bundles across runs to data is complete")
    parser.add_argument("--imago-features",
                        default="observation,action_dist,action_logits,value_function",
                        type=str2list, help="Features to collect and store for imagination")
    parser.add_argument("--store-imago", default=False,type=str2bool,
                        help="Boolean on whether to collect and store imago dataset")
    parser.add_argument("--init-ray", default=True,type=str2bool,
                        help="Whether or not to init Ray (may already be running)")
    parser.add_argument("--config",default="{}",type=str2dict,
                        help="Algorithm-specific configuration (e.g. env, hyperparams). "
                        "Gets merged with loaded configuration from checkpoint file and "
                        "`evaluation_config` settings therein.")
    parser.add_argument('--num-workers', default = 4, type = int,
                        help="Number of rollout workers to utilize during training")
    parser.add_argument('--num-gpus', default = 1,type = int,
                        help="Number of GPUs to utilize during training. Generally this is not bottleneck, so 1 is often sufficient")
    parser.add_argument('--log-level', default = "info", type=str2log_level,
                        help = "Get logging level from input args: 'info' | 'warn','warning' | 'error' | 'critical' ")
    parser.add_argument('--wrappers', default="", type = str2wrapper,  help=
                        """
                            Wrappers to encode the environment observation in different ways. Wrappers will be executed left to right,
                                and the options are as follows (example: 'encoding_only,canniballs_one_hot'):
                        - partial_obs.{obs_size}:         Partial observability - must include size (odd int)
                        - encoding_only:                  Provides only the encoded representation of the environment
                        - rgb_only:                       Provides only the RGB screen of environment
                        - canniballs_one_hot              Canniballs specific one-hot
                        """)
    return parser

#######################################################################
# Helper Functions
#######################################################################

class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    return DEFAULT_POLICY_ID

def keep_going(steps, num_steps, episodes, num_episodes):
    """Determine whether we've collected enough data

    :steps: int:        Current step
    :num_steps: int:    Total number of steps
    :episodes: int:     Current episode
    :num_episodes: int: Total number of episodes

    """
    # If num_episodes is set, stop if limit reached.
    if num_episodes and episodes >= num_episodes:
        return False
    # If num_steps is set, stop if limit reached.
    elif num_steps and steps >= num_steps:
        return False
    # Otherwise, keep going.
    return True

def get_rollout_tags(writer_files,
                     writer_dict,
                     ext):
    """Get tags for making rollout tag
    folder names

    :writer_files: List: List of output rollout files
    :writer_dict: Dict:  Dictionary to store rollout files
    :ext: str:           File extension (pkl, hkl)

    """
    # Fill writer dict with tags
    for file in writer_files:
        # Get tag for video
        tag = file.split("/")[-1]\
                       .split("_")[-1]\
                       .replace(".{}".format(ext),"")

        writer_dict[tag] = file

def _update_bundle_configuration(args):
    """Update bundle configuration to reflect
    correct random seed if rollouts not run during
    the execution (e.g. bundle-only execution)

    :args: Argparse.Args: User-defined arguments

    """
    name_components = args.bundle_only_dir.split("/")[-1].split("_")
    try:
        num_workers = int(name_components[-2].replace("w",""))
        random_seed = int(name_components[-3].replace("rs",""))
        args.seed = random_seed
        args.num_workers = num_workers
    except:
        logging.warn("Random seed information could not be extracted"
              " from provided bundle-only filename. Be sure that"
              " your provided random seed and number of workers"
              " matches the rollout execution.\n")

    args.writer_dir = args.bundle_only_dir

def delete_dummy_video_files(pid_max,
                             subdirs):
    """OpenAI's video creator creates one dummy
    video file per worker. This removes it.
    The id is always the last (max)

    :pid_max: dict: Max pid filenames
    :subdirs: dict: Dictionary of subdirectories by (pid, ep)

    """

    # Remove all dummy files
    for pid,file in pid_max['files'].items():

        # Get episode
        ep = pid_max[pid]

        # Remove garbage subdir
        subdirs.remove((pid,ep))
        os.remove(file)

def build_writer_dir(args):
    """Build writer directory

    :args: Argparse args

    """
    # Add writer to environment
    args.writer_dir = "{}{}_{}_{}_ep{}_ts{}_rs{}_w{}_{}/".format(args.outdir,
                                                args.model_name,
                                                args.config['framework'],
                                                args.env_name,
                                                args.num_episodes,
                                                args.num_timesteps,
                                                args.seed,
                                                args.num_workers,
                                                dt.now().strftime("%Y.%m.%d"))

    if not os.path.exists(args.writer_dir) and not args.bundle_only:
        os.makedirs(args.writer_dir)


def bundle_rollouts(subdirs,
                    writer_dict,
                    writer_dir,
                    ext = "hkl",
                    monitor = True):
    """Bundle rollout pkl files and videos
    into different folders. Only runs if
    videos are created. Otherwise pickles
    stay loose in the directory

    :subdirs: Set:      Set of pid,episode combos
    :writer_dict: Dict: Dictionary to store rollout files
    :writer_dir: str:   Base path to write rollouts
    :ext: str:          File extension (pkl, hkl)
    :monitor: bool:     Whether or not video files were generated

    """

    # Bundle everything together
    rollout_subdirs = []
    for (pid,ep) in subdirs:

        # Suffix and filepath - I know, this is messy.
        writer_file = writer_dict["pid{}-{}".format(ep,pid)]
        writer_suffix = writer_file.split("/")[-1]

        # Make rollout subdirectory
        rollout_subdir = "{}/{}".format(writer_dir,
                                        writer_suffix.split("_")[0])
        #Make directory if it does not exist
        if not os.path.exists(rollout_subdir):
            os.mkdir(rollout_subdir)

        # Replace file and move to subdir
        os.replace(writer_file,
                "{}/{}".format(rollout_subdir,
                               writer_suffix))

        # Only move if they exist
        if monitor:
            os.replace("{}{}_ep{}_video.mp4".format(writer_dir,pid,ep),
                    "{}/{}_video.mp4".format(
                                            rollout_subdir,
                                            writer_suffix))


def rename_video_files(writer_dir,
                       video_files,
                       subdirs,
                       pid_max):
    """Rename video files to match writer
    output. OpenAI names the videos first

    :writer_dir: str:   Writer directory for files
    :video_files: List: List of video files
    :subdirs: Set:      Set of (pid, episode) tuples
    :pid_max: Dict:     Dictionary of dummy video ids

    """
    for v in video_files:
        # Only do this for non-processes files
        # Remove any dummy files at end
        try:

            # Get name components based on openai naming convention
            name_components = v.split("/")[-1].split(".")[-3:]

            # Get PID and episode
            pid = int(name_components[0])
            ep = int(name_components[1].replace('video',''))

            # Add video to registry
            subdirs.add((pid,ep))

            # Add new video name
            new_video_name = "{}{}_ep{}_video.mp4".format(
                                          writer_dir,
                                          pid,
                                          ep)

            #Keep track of which files to remove
            pid_max[pid] = max(ep,
                               pid_max.get(pid,0))
            if pid_max[pid] == ep:
                pid_max['files'][pid] = new_video_name

            os.rename(v, new_video_name)

        except Exception as e:
            # print("Process failed for filepath {}"\
            #       .format(v))
            pass

def check_for_saved_config(args):
    """Check for saved configuration

    :args: Argparse arguments
    :returns: Saved config file with merged updates

    """

    # Load configuration from checkpoint file.
    config_path = ""
    args.save_info = True
    config = None

    # If there is a checkpoint, find parameters
    if args.checkpoint_path:
        config_dir = os.path.dirname(args.checkpoint_path)
        config_path = os.path.join(config_dir, "params.pkl")
        # Try parent directory.
        if not os.path.exists(config_path):
            config_path = os.path.join(config_dir, "../params.pkl")

    # Load the config from pickled.
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)
    # If no pkl file found, require command line `--config`.
    else:
        # If no config in given checkpoint -> Error.
        if args.checkpoint_path:
            raise ValueError(
                "Could not find params.pkl in either the checkpoint dir or "
                "its parent directory AND no `--config` given on command "
                "line!")

        # Use default config for given agent.
        _, config = get_trainer_class(args.model_name, return_config=True)


    # Make sure worker 0 has an Env.
    config["create_env_on_driver"] = True

    # Merge with `evaluation_config` (first try from command line, then from
    # pkl file).
    evaluation_config = copy.deepcopy(
        args.config.get("evaluation_config", config.get(
            "evaluation_config", {})))
    config = merge_dicts(config, evaluation_config)
    # Merge with command line `--config` settings (if not already the same
    # anyways).

    # Adds any custom arguments here
    config = merge_dicts(config, args.config)

    if not args.env_name:
        if not config.get("env"):
            parser.error("the following arguments are required: --env")
        args.env_name = config.get("env")

    # Make sure we have evaluation workers.
    if not config.get("evaluation_num_workers"):
        config["evaluation_num_workers"] = config.get("num_workers", 0)
    if not config.get("evaluation_num_episodes"):
        config["evaluation_num_episodes"] = 1

    return config

#######################################################################
# Load config file
#######################################################################

def load_config(args):
    """Load configuration given all inputs

    :args: Argparse Args: User defined arguments
    :returns: Dict:       Config

    """

    # Set up config
    args.config = check_for_saved_config(args)
    config = args.config

    #Build writer directory
    build_writer_dir(args)

    # Set render environment and number of workers
    # as well as number of gpus
    config["render_env"] = not args.no_render
    config["num_workers"] = 0
    config["evaluation_num_workers"] = args.num_workers
    config["num_gpus"] = args.num_gpus
    args.ext ="hkl" if args.use_hickle else "pkl"
    args.read_compressed = _read_hkl if args.use_hickle else _read_pkl
    args.epochs_trained = None

    # Add random seed
    config["seed"] = args.seed

    # Only set if  provided explicitly and there is not checkpoint
    if args.framework:
        # Specifying framework here only if one is explicity provided
        config["framework"] = args.framework

    # Allow user to specify a video output path.
    # Determine the video output directory.
    if args.store_video:
        # Add writer to environment
        args.video_dir = os.path.abspath(args.writer_dir)
        config["record_env"] = args.video_dir

    else:
        args.video_dir = None
        config["monitor"] = False

    return config

#######################################################################
# Gather up relevant information for imagination, then save
#######################################################################

def _init_imago_dict(fields):
    """Initialize imago dictionary
    for collection

    :fields: List: List of fields for dict keys

    :returns: Dict: Imago collection dictionary

    """

    imago_dict = {}
    for f in fields:
        imago_dict[f] = []

    return imago_dict

def _add_imago_sample(sample,imago_dict):
    """Add sample to imago_dict

    :sample: Dict:     Rollout sample
    :imago_dict: Dict: Imago dictionary store

    """
    for k,store in imago_dict.items():
        # Only flatten artifacts that should be vectors
        if (isinstance(sample[k],np.ndarray) and
            ((len(sample[k].shape) == 2) and
             sample[k].shape[0] == 1)):
            sample[k] = sample[k].flatten()
        store.append(sample[k])

def _get_checkpoint_root(checkpoint):
    """Get root directory of checkpoint

    :checkpoint: str: Checkpoint filepath
    :returns: str: Checkpoint root directory

    """
    components = checkpoint.split("/")
    root = [c for c in components if not c.lower().startswith('checkpoint')]
    return "/".join(root)

def _bundle_imago_dict(imago_dict):
    """Bundle imago dictionary for better
    compression

    :imago_dict: Dictionary of imago information

    """
    for k,v in imago_dict.items():
        # 1D array
        if not isinstance(v[0],np.ndarray):
            imago_dict[k] = np.array(v)

        #2+D array
        else:
            imago_dict[k] = np.stack(v,axis=0)

def collect_imago_dataset(args):
    """Collect imago dataset information and store
    for training. Saves to data/imago

    :args: Argparse.Args: Argparse arguments

    """


    imago_dict = _init_imago_dict(args.imago_features)
    rollout_paths = glob.glob(args.writer_dir + "/**/*.{}"\
                                    .format(args.ext),recursive=True)

    # Rollout regex
    rollout_regex =r'(\w+)_cp(\d+)_s(\d+)_r*(.+).[ph]kl'

    # Rollout path file iteration
    num_rollouts = 0
    for i in tqdm(range(len(rollout_paths))):
        rollout_path = rollout_paths[i]
        episode_id = rollout_path.split("/")[-1]
        matches = re.match(rollout_regex, episode_id)
        if matches:
            num_rollouts += 1

            # Iterate through timesteps
            data = args.read_compressed(rollout_path)
            for i in range(len(data.keys())):
                _add_imago_sample(data[i],
                                  imago_dict)

    # Bundle imago information
    _bundle_imago_dict(imago_dict)


    # Store the dictionary in the correct folder
    imago_dir = "{}{}_{}_{}".format(args.imago_dir,
                                    args.model_name,
                                  args.framework,
                                  args.env_name)

    if not os.path.exists(imago_dir):
        os.makedirs(imago_dir)
    imago_filename = "{}/imago_rollouts-{}_rs{}_w{}.hkl".format(imago_dir,
                                                            num_rollouts,
                                                            args.seed,
                                                            args.num_workers)

    _write_hkl(imago_dict, imago_filename)

def sync_bundled_data(writer_dir,
                 ext = "hkl"):
    """Sync rollout dirs across runs to ensure
    that there is no missing data

    :rollout_dirs: List of final rollout directories
    :ext: Filename extension

    """
    # Get all directories
    rollout_dirs = [d for d in os.listdir(writer_dir) if os.path.isdir(writer_dir + d)]
    max_files =0

    # Find max data in all bundle folders
    for d in rollout_dirs:
        cur_files =len(list(glob.glob(writer_dir + d+ "/*.{}".format(ext))))
        max_files = max(max_files, cur_files)

    # Remove any folders with missing data
    for d in rollout_dirs:
        cur_files =len(list(glob.glob(writer_dir + d + "/*.{}".format(ext))))
        if cur_files < max_files:
            shutil.rmtree(writer_dir + d)

def cleanup(monitor,
            writer_dir,
            ext = "hkl",
            sync_bundles = False):
    """

    Clean up folders where artifacts saved. Lots of meaningless
    stuff gets generated by RLlib and clutters the space. This also
    renames everything more intuitively

    TODO: This is a bandaid for some artifacts I can't control easily.
    Probably not a great thing to have long-term

    :outdir: str:     Output directory for pickle artifact
    :writer_dir: str: Directory where artifacts saved
    :ext: str:        Filename extension (pkl, hkl)

    """

    # Relevant files
    video_files = glob.glob(writer_dir+"*.mp4")
    writer_files = glob.glob(writer_dir+"*.{}".format(ext))
    unneeded_jsons = glob.glob(writer_dir+"*.json")

    # Remove unneeded JSON files
    for f in unneeded_jsons:
        os.remove(f)

    # Useful storage objects
    pid_max = {}
    pid_max['files'] = {}
    writer_dict = {}
    subdirs = set()

    # Rename video files
    rename_video_files(writer_dir,
                       video_files,
                       subdirs,
                       pid_max)

    # Delete the dummy video files made by OpenAi monitor
    delete_dummy_video_files(pid_max,
                             subdirs)

    # Get folder names for all rollouts (by tag)
    get_rollout_tags(writer_files,
                     writer_dict,
                     ext = ext)

    # Bundle all of the artifacts together
    bundle_rollouts(subdirs,
                    writer_dict,
                    writer_dir,
                    monitor = monitor,
                    ext = ext)

    # Sync rollout bundles across runs
    if sync_bundles:
        sync_bundled_data(writer_dir,
                          ext = ext)



#######################################################################
# Rollout information
#######################################################################

def run_rollout(agent,
            env,
            env_name,
            num_steps,
            num_episodes=0,
            no_render=True,
            video_dir=None,
            args = None):
    """
    Rollout execution function. This was largely inherited from RLlib.

    :agent: Agent:        Rllib agent
    :env:   Env:          Gym environment
    :env_name: str:       Env id / name
    :num_steps: Int:      number of steps
    :num_episodes: Int:   Number of episodes
    :no_render: bool:     Whether to render environment for visual inspection
    :video_dir: str:      Video storage path
    :args: Argparse.Args: User defined arguments

    """

    policy_agent_mapping = default_policy_agent_mapping
    # Normal case: Agent was setup correctly with an evaluation WorkerSet,
    # which we will now use to rollout.
    if hasattr(agent, "evaluation_workers") and isinstance(
            agent.evaluation_workers, WorkerSet):
        steps = 0
        episodes = 0

        while keep_going(steps, num_steps, episodes, num_episodes):
            eval_result = agent.evaluate()["evaluation"]

            # Increase timestep and episode counters.
            eps = agent.config["evaluation_num_episodes"]
            episodes += eps
            steps += eps * eval_result["episode_len_mean"]
            # Print out results and continue.
            logging.info("Episode #{}: reward: {}".format(
                episodes, eval_result["episode_reward_mean"]))
        return

    # Agent has no evaluation workers, but RolloutWorkers.
    elif hasattr(agent, "workers") and isinstance(agent.workers, WorkerSet):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]
        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    logging.warn("""\nWARNING: You are either spinning up a random agent (untrained, no checkpoint)
         or you have a malformed checkpoint object with no evaluation workers.
         You may run rollouts in this way, but rollouts will be slow. Instead, read in a
         checkpoint from the very beginning of your model training for faster rollouts.""")

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    # If monitoring has been requested, manually wrap our environment with a
    # gym monitor, which is set to record every episode.
    if video_dir:
        env = gym_wrappers.Monitor(
            env=env,
            directory=video_dir,
            video_callable=lambda _: True,
            force=True)

    # Make episode writer
    env = EpisodeWriterWrapper(env, args = args)

    steps = 0
    episodes = 0
    while keep_going(steps, num_steps, episodes, num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and keep_going(steps, num_steps, episodes,
                                      num_episodes):
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                # action = agent.compute_action(a_obs)
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = flatten_to_single_ndarray(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, info = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(
                    r for r in reward.values() if r is not None)
            else:
                reward_total += reward
            if not no_render:
                env.render()
            steps += 1
            obs = next_obs

        logging.info("Episode #{}: reward: {}".format(episodes, reward_total))
        if done:
            episodes += 1


########################################################################
## Run method for rollout information
########################################################################
def run_rollouts(args,config):
    """Run rollouts (if not bundling
    existing rollouts)

    :args: Argparse.Args: User defined arguments
    :config: Dict: Execution Configuration

    """

    # Make sure configuration has the correct outpath
    config['callbacks'] = lambda: RLlibIxdrlCallbacks(
                                                        args = args,
                                                        config = config)

    #Spin up Ray only if it is not already running
    if args.init_ray:
        ray.init(local_mode=args.local_mode)

    # Set up environment
    env = gym.make(args.env_name)

    # Wrap environment
    env = wrap_env(env, args.wrappers)

    # Register environment with Ray
    register_env(args.env_name, lambda config: env)

    # Create the model Trainer from config.
    cls = get_trainable_cls(args.model_name)

    # Instantiate agent
    agent = cls(env=args.env_name, config=config,
                logger_creator = cameleon_logger_creator(
                                    args.writer_dir))

    # Restore agent if needed
    if args.checkpoint_path:

        # This is not ideal, but only way to guarantee
        # correct information about model. Add slight overhead
        # Need to restore the model for rollouts but, then
        # must restart to feed information to logger
        logging.info("Restoring agent twice to feed information correctly to logger")
        agent.restore(args.checkpoint_path)

        # Make sure configuration has the correct outpath
        args.epochs_trained = agent._iteration if agent._iteration is not None else 0

        # Make sure configuration has the correct outpath
        config['callbacks'] = lambda: RLlibIxdrlCallbacks(
                                                            args = args,
                                                            config = config)
        # Need to run setup again with new callbacks
        agent.setup(config)
        agent.restore(args.checkpoint_path)

    # Do the actual rollout.
    run_rollout(agent,env, args.env_name,
                args.num_timesteps, args.num_episodes,
                args.no_render, args.video_dir,
                args = args)

    # Stop the agent
    agent.stop()

    # Get the gross files out of there
    cleanup(config['monitor'],
            args.writer_dir,
            ext = args.ext,
            sync_bundles=args.sync_bundles)

def _get_args_from_metadata(args, metadata):
    """Get metadata from training configuration

    :args: Argparse.Args: User-defined arguments
    :metadata: Dict: Metadata for run

    """
    train_metadata = metadata['train']
    args.model_name = train_metadata['model_name']
    args.framework = train_metadata['framework']
    args.train_env_name = train_metadata['env_name']
    args.rollout_env_name = args.env_name

def rollout(args, parser=None):
    """
    Run rollouts with arguments and
    argparse parser

    """

    # Set logging level
    logging.basicConfig(level=args.log_level,
                        format='%(message)s')

    # Load up metadata if a checkpoint is provided
    metadata = {"rollout":None}
    if args.checkpoint_path:
        args.checkpoint_root = _get_checkpoint_root(args.checkpoint_path)
        metadata = _load_metadata(args.checkpoint_root)
        _get_args_from_metadata(args,metadata)

    # Load config from saved dictionary
    # Framework should match checkpoint
    config = load_config(args)

    # Check to make sure bundling only option is correctly formed
    assert ((args.bundle_only and args.bundle_only_dir) or
            (not args.bundle_only)),\
        "ERROR: bundle only set without accompanying filepath"

    # If not bundle only, run the rollouts
    if not args.bundle_only:
        run_rollouts(args,config)
    else:
        _update_bundle_configuration(args)

    # If we are saving for imago, do it now (when memory clear)
    if args.store_imago:
        collect_imago_dataset(args)

    # Save metadata from training
    args.config = config
    metadata["rollout"] = vars(args)
    _save_metadata(metadata, args.writer_dir)



#######################################################################
# Main - Run rollout parsing engine
#######################################################################

def main():
    """Main method for rollout argparse

    """
    parser = create_parser()
    args = parser.parse_args()

    # Run rollout
    rollout(args, parser)


#######################################################################
#  Run main method
#######################################################################

if __name__ == "__main__":
    main()
