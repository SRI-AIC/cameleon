#################################################################################
#
#             Project Title:  CAML Minigrid Envs (test sample)
#             Author:         Sam Showalter
#             Date:           2021-07-06
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
from cameleon.register import register
from gym_minigrid.envs import DynamicObstaclesEnv

#################################################################################
#   Function-Class Declaration
#################################################################################

class DynamicObstaclesEnv50x50(DynamicObstaclesEnv):

    """Large minigrid environment"""

    def __init__(self):
        super().__init__(size = 50,
                         n_obstacles=12)


class DynamicObstaclesEnv42x42(DynamicObstaclesEnv):

    """Large minigrid environment"""

    def __init__(self):
        super().__init__(size = 42,
                         n_obstacles=12)

class DynamicObstaclesEnv20x20(DynamicObstaclesEnv):

    """Large minigrid environment"""

    def __init__(self):
        super().__init__(size = 20,
                         n_obstacles=8)

class DynamicObstaclesEnv100x100(DynamicObstaclesEnv):

    """Large minigrid environment"""

    def __init__(self):
        super().__init__(size = 100,
                         n_obstacles=45)



#################################################################################
#   Register envs
#################################################################################

register(
    id='Cameleon-Dynamic-Obstacles-50x50-v0',
    entry_point='cameleon.envs:DynamicObstaclesEnv50x50'
)

register(
    id='Cameleon-Dynamic-Obstacles-100x100-v0',
    entry_point='cameleon.envs:DynamicObstaclesEnv100x100'
)

register(
    id='Cameleon-Dynamic-Obstacles-20x20-v0',
    entry_point='cameleon.envs:DynamicObstaclesEnv20x20'
)

register(
    id='Cameleon-Dynamic-Obstacles-42x42-v0',
    entry_point='cameleon.envs:DynamicObstaclesEnv42x42'
)



















