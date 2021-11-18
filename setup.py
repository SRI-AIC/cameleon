#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='cameleon',
      version='1.0',
      description='CAMeLeon is a user-friendly research and development tool built to standardize RL competency assessment for custom agents and environments',
      url='https://github.com/SRI-AIC/cameleon',
      author='Sam Showalter',
      author_email='sam.showalter@sri.com',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'gym',
          'gym-minigrid',
          'numpy',
          'matplotlib',
          'tqdm',
          'ray[rllib,tune]==1.6.0',
          'torch',
          'tensorflow',
          'hickle',
          'argparse',
          'pathlib',
          'datetime',
      ],
      zip_safe=True)
