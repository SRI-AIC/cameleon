###############################################################################
#
#             Project Title:  Environment and Argparse Utilities for Cameleon Env
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import pickle as pkl
import hickle as hkl
import logging
import json
from operator import add

#######################################################################
# Important Global Variables
#######################################################################

LOG_LEVEL_DICT = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warn': logging.WARNING,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

#################################################################################
#   Function-Class Declaration
#################################################################################

def _write_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pkl.dump(obj, file)

def _read_pkl(filename):
    with open(filename, 'rb') as file:
        return pkl.load(file)

def _write_hkl(obj, filename):
    # with open(filename, 'wb') as file:
    hkl.dump(obj, filename, mode = 'w',
             compression='gzip')

def _read_hkl(filename):
    return hkl.load(filename)

def _tup_equal(t1,t2):
    """Check to make sure to tuples are equal

    :t1: Tuple 1
    :t2: Tuple 2
    :returns: boolean equality

    """
    if t1 is None or t2 is None:
        return False
    return (t1[0] == t2[0]) and (t1[1] == t2[1])

def _tup_add(t1, t2):
    """Add two tuples

    :t1: Tuple 1
    :t2: Tuple 2

    :returns: Tuple sum

    """
    return tuple(map(add,t1,t2))


def _tup_subtract(t1, t2):
    """Add two tuples

    :t1: Tuple 1
    :t2: Tuple 2

    :returns: Tuple sum

    """
    t2 = (-t2[0],-t2[1])
    return tuple(map(add,t1,t2))

def _tup_mult(t1, t2):
    """Multiply tuples

    :t1: Tuple 1
    :t2: Tuple 2

    :returns: Tuple sum

    """
    return (t1[0]*t2[0],t1[1]*t2[1])

def _save_metadata(args,outdir):
    """Save metadata for execution

    :args: Argparse.Args: User-defined arguments
    :outdir: str: Ouput directory

    """
    with open("{}/metadata.json".format(outdir), 'wt') as f:
        json.dump(args, f, indent=4,default=str)


def _load_metadata(indir):
    """Load metadata for execution

    :indir: str: Input directory

    """
    with open("{}/metadata.json".format(indir)) as f:
        return json.load(f)
#######################################################################
# Main
#######################################################################

