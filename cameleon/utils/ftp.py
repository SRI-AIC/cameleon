#################################################################################
#
#             Project Title:  Transfer files to remote server
#             Author:         Sam Showalter
#             Date:           2021-08-10
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################


import os
import re
import logging
import pathlib
import shutil
import sys

from getpass import getpass
import requests
from requests.auth import HTTPBasicAuth


#################################################################################
#   Function-Class Declaration
#################################################################################

CAMELEON_DIR_DICT =\
       {'models':r'(\S+)_(\S+)_(\S+)_rs(\d+)_w(\d+)(\d+).(\d+).(\d+)',
        'models/tune':r'(\S+)-(\S+)-v(\d+)',
        'rollouts':r'(\S+)_(\S+)_(\S+)_ep(\S+)_ts(\S+)_rs(\d+)_w(\d+)_(\d+).(\d+).(\d+)' ,}

class CameleonHttpFTP(object):

    """Transfer files to remote server"""

    def __init__(self,
                 username,
                 zip_only = False,
                 post_only = False,
                 remote_server_root="",
                 overwrite = False,
                 project_root ="../../../",
                 dirs = list(CAMELEON_DIR_DICT.keys()),
                 dir_dict = CAMELEON_DIR_DICT,
                 archive = 'archive',
                 log_level = logging.INFO,
                 ):

        # Set logging level
        logging.basicConfig(level=log_level,
                            format='%(message)s')
        self.remote_server = remote_server_root
        self.project_root = self._set_project_root(project_root)
        self.dir_dict = dir_dict
        self.dirs = dirs
        self.username = username
        self._auth = self._validate_auth()
        self.zip_only = zip_only
        self.post_only = post_only
        self.overwrite = overwrite
        self.archive = archive

    def _set_project_root(self,project_root):
        """Set project root relative to file

        :project_root: str: Project root string

        """
        root = pathlib.Path(os.path.join(
            os.path.abspath(__file__),
            project_root)).resolve()
        return root

    def _validate_auth(self):
        """Get password and validate

        """
        pwd = getpass("Password for {} to access:\n - {}\n\nPassword:"\
                      .format(self.username,
                              self.remote_server))
        auth = HTTPBasicAuth(self.username,pwd)
        response = requests.head(self.remote_server,
                            auth = auth)
        self._validate_response(response)
        return auth

    def _validate_response(self, response):
        """Validate Http response

        :response: Response object from requests

        """
        sc = response.status_code
        if (sc == 200):
            return
        elif (sc == 404):
            assert False,\
                "ERROR: Remote server root not found, though authentication succeeded."
        elif (sc == 401):
            assert False,\
                "ERROR: Authentication failed. Please check credentials."
        elif (sc == 301):
                logging.info("ERROR: Remote server root not a directory. Request redirected")

    def zip_dirs(self):
        """Zip listed directories that match provided regex

        """
        # Get all folders in the directory
        for k,v in self.dir_dict.items():

            if k not in self.dirs:
                continue
            logging.info('')
            logging.info('=========='*7)
            logging.info("Zipping dirs in {} project root".format(k))
            logging.info('=========='*7)
            items = [(i,os.path.join(self.project_root,k,i))
                     for i in os.listdir(os.path.join(self.project_root,k))
                     if os.path.isdir(os.path.join(self.project_root,k,i))
                     and re.match(v,i)]

            for i, i_path in items:
                logging.info('----------'*5)
                logging.info("Zipping {}".format(os.path.join(k,i)))
                final_dir = os.path.join(self.project_root,
                                        self.archive,k)
                final_path = os.path.join(self.project_root,
                                          self.archive,k,i)

                # Only if directory does not already exist
                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)

                # Zip up  file
                if ((not os.path.isfile("{}.zip".format(final_path)))
                    or (self.overwrite)):
                    shutil.make_archive(final_path, 'zip', i_path)
                else:
                    logging.warn("File already created:\n - {}.zip"\
                        .format(os.path.join(k,i)))
    def post_dirs(self):
        """Post zipped up directories in archive

        """
        for k,v in self.dir_dict.items():
            if ((k not in self.dirs)
                or not os.path.exists(os.path.join(self.project_root,self.archive,k))):
                continue
            logging.info('')
            logging.info('=========='*7)
            logging.info("Posting dirs in {} archive root".format(k))
            logging.info('=========='*7)

            items = [(i,os.path.join(self.project_root,self.archive,k,i)) for i
                     in os.listdir(os.path.join(self.project_root,self.archive,k))
                     if re.match(v,i)]

            for i, i_path in items:
                zipfile = open(i_path, 'rb')
                files = {i: zipfile}
                url = os.path.join(self.remote_server,k,i)
                r = requests.head(url,
                                  auth=self._auth)
                logging.info('----------'*5)
                if (r.status_code == 404) or ((r.status_code == 200) and self.overwrite):
                    logging.info("Posting {}".format(os.path.join(k,i)))
                    with zipfile as f:
                        requests.put(url,data=f,
                                    auth = self._auth)
                elif (r.status_code == 200):
                    logging.warn("Filename already exists, skipping:\n - {}"\
                          .format(url.split("/")[-1]))
                    continue

    def run(self):
        """Run full execution for zipping and
        pushing zipfile information

        """
        if self.zip_only:
            self.zip_dirs()
        elif self.post_only:
            self.post_dirs()
        else:
            self.zip_dirs()
            self.post_dirs()

#################################################################################
#   Main Method
#################################################################################



