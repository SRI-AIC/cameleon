#################################################################################
#
#             Project Title:  Send emails to yourself when execution is done
#             Author:         Sam Showalter
#             Date:           2021-08-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import smtplib, ssl
import os
import logging
import sys
import base64
import datetime as dt
from getpass import getpass

#################################################################################
#   Dictionary template for all executions
#################################################################################

EMAIL_TEMPLATES = {
"train_finished": """\
Subject: {} agent finished training on {}

Hi! Cameleon Email Bot here. You training job has finished.
Please see details of the execution below:

    - Agent: {}
    - Environment: {}
    - Framework: {}

    Planned Execution:
    - Number of epochs: {}
    - Number of episodes: {}
    - Number of timesteps: {}

    Actual Execution:
    - Total epochs: {}
    - Total episodes: {}
    - Total timesteps: {}

    Time information:
    - Total time elapsed: {}
    - Current time: {}

    Other information:
    - Random seed: {}
    - Checkpoint epochs: {}
    - Output directory:
        + {}
""",
"failure":"""\
Subject: {} agent FAILED while {} on {}

    Hi! Cameleon Email Bot here. You training job has FAILED :(.
Please see details of the failure and execution progress below:

------------------------------------------------------------------
Failure Message:
---------------

        {}

Failure StackTrace:
------------------

{}

------------------------------------------------------------------
Training Progress:

    - Agent: {}
    - Environment: {}
    - Framework: {}

    Planned Execution:
    - Number of epochs: {}
    - Number of episodes: {}
    - Number of timesteps: {}

    Actual Execution:
    - Total epochs: {}
    - Total episodes: {}
    - Total timesteps: {}

    Time information:
    - Total time elapsed: {}
    - Current time: {}

    Other information:
    - Random seed: {}
    - Checkpoint epochs: {}
    - Output directory:
        + {}
"""
}

#################################################################################
#   Function-Class Declaration
#################################################################################

class CameleonEmailBot(object):

    """
    System to send you emails when training, rollouts,
    or other experiments are finished

    """

    def __init__(self,
                 email_sender,
                 email_receiver,
                 email_server = "smtp.mail.yahoo.com",
                 port = 465
                 ):
        self.email_server = email_server
        self.email_sender = email_sender
        self.email_receiver = email_receiver
        self.message_type = None
        self.port = port

        # Not that secure, but better than nothing
        pwd =getpass(prompt = "Please enter app password for email {}:"\
                                                 .format(self.email_sender))
        self.password = base64.b64encode(pwd.encode("utf-8"))

    def send_email(self, message_type, args):
        """Send email to user with updates on execution

        :message_type: str: Type of message (e.g. failure)
        :args: Argparse.Args: User-defined arguments

        """
        self.message_type = message_type
        email_router = {'train_finished':self._write_email_train_finished,
                        'failure':self._write_email_failure}

        # Send email
        message = email_router[self.message_type](args)
        logging.info(message)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(self.email_server, self.port, context=context) as server:
            server.login(self.email_sender,
                         base64.b64decode(self.password).decode('utf8'))
            server.sendmail(self.email_sender, self.email_receiver, message)

        logging.info("Email successfully sent!")

    def _write_email_train_finished(self, args):
        """Write email for training finishing

        :args: Argparse args
        :returns: Message for email

        """
        return EMAIL_TEMPLATES[self.message_type].format(
            args.model_name,
            args.env_name,

            args.model_name,
            args.env_name,
            args.framework,

            args.num_epochs,
            args.num_episodes,
            args.num_timesteps,

            args.epochs_total,
            args.episodes_total,
            args.timesteps_total,

            dt.timedelta(seconds = round(args.time_total_s)),
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            args.seed,
            args.checkpoint_epochs,
            args.outdir)


    def _write_email_failure(self,args):
        """Write email for training finishing

        :args: Argparse args
        :returns: Message for email

        """
        return EMAIL_TEMPLATES[self.message_type].format(

            args.model_name,
            args.execution_type,
            args.env_name,

            args.failure_message,
            args.failure_stacktrace,

            args.model_name,
            args.env_name,
            args.framework,

            args.num_epochs,
            args.num_episodes,
            args.num_timesteps,

            args.epochs_total,
            args.episodes_total,
            args.timesteps_total,

            dt.timedelta(seconds = round(args.time_total_s)),
            dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            args.seed,
            args.checkpoint_epochs,
            args.outdir)




#################################################################################
#   Main Method
#################################################################################




#################################################################################
#   Main Method
#################################################################################
