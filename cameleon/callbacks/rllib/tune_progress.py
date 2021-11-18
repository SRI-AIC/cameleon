#################################################################################
#
#             Project Title:  Tune progressbar that also has an eta
#             Author:         Sam Showalter
#             Date:           2021-07-26
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import sys
from typing import Dict, List, Optional, Union
import logging
import datetime as dt
from ray.tune.progress_reporter import CLIReporter
from ray.tune.trial import Trial


#################################################################################
#   Function-Class Declaration
#################################################################################

class CameleonRLlibTuneReporter(CLIReporter):

    """CLI Reporter Instantiation for Cameleon RLlib Tune compatibility"""

    def __init__(self, args,**kwargs):
        CLIReporter.__init__(self,**kwargs)
        self.args = args

    def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
        # print(self._progress_str(trials, done, *sys_info))
        result = trials[0].last_result

        # Get total time passed so far and current iteration
        total_time = result.get('time_total_s','')
        current_iter = result.get('training_iteration','')

        # Start up condition
        if total_time =='':
            return

        # Get stopping criteria information
        stop_crit_dict = {"Epochs":[self.args.num_epochs,result['training_iteration']],
                        "Episodes":[self.args.num_episodes,result['episodes_total']],
                        "Timesteps":[self.args.num_timesteps,result['timesteps_total']]}
        sc = stop_crit_dict[self.args.stopping_crit]

        # Get metrics
        avg_per_training_iteration = round(total_time / current_iter,2)
        avg_per_iteration= total_time / sc[1]
        time_left_s = (sc[0] - sc[1])*avg_per_iteration
        percent_complete = round(sc[1]*100 / sc[0],2)


        time_status = "Model: {}\nEnv: {}\n\nEpoch {:2d} | ETA {} | {:6.2f}% complete | Avg. Epoch {:6.2f} sec.\n".format(
                        self.args.model_name,
                        self.args.env_name,
                        current_iter,
                        dt.timedelta(seconds = round(time_left_s)),
                        percent_complete,
                        avg_per_training_iteration)

        reward_status = " - Reward min|mean|max {:6.2f} | {:6.2f} | {:6.2f} - Mean length {:4.2f}"
        reward_status = reward_status.format(
                    result["episode_reward_min"],
                    result["episode_reward_mean"],
                    result["episode_reward_max"],
                    result["episode_len_mean"]
                        )

        curr_timestamp = dt.datetime.now()
        current_time = " - {} remaining: {}\n\n"\
                        " - Epochs total: {}\n"\
                        " - Episodes total: {}\n"\
                        " - Timesteps total: {}\n\n"\
                        " - Current time: {}\n"\
                        " - Est. datetime at finish: {}\n"\
                        " - Total elapsed {}"\
            .format(self.args.stopping_crit,
                    sc[0] - sc[1],
                    result['training_iteration'],
                    result['episodes_total'],
                    result['timesteps_total'],
                    curr_timestamp.strftime("%y-%m-%d %H:%M:%S"),
                    (curr_timestamp + dt.timedelta(seconds = round(time_left_s))).strftime("%y-%m-%d %H:%M:%S"),
                    dt.timedelta(seconds = round(result['time_total_s'])))

        logging.info(time_status)
        logging.info(current_time)
        logging.info(reward_status)
        logging.info("====="*15)
        sys.stdout.flush()



#################################################################################
#   Main Method
#################################################################################



