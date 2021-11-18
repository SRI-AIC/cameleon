#################################################################################
#
#             Project Title:  Cameleon Base Objects (extend if useful)
#                             Originally based on minigrid objects
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#######################################################################
# Module Imports
#######################################################################

import os
import sys
import numpy as np

# Fill coordinates rendering for minigrid
from gym_minigrid.rendering import *

#######################################################################
# Tile Pixel Size
#######################################################################


# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

#######################################################################
# Color to IDX
#######################################################################


# Map of color names to RGB values
COLORS = {
    None    : np.array([0, 0, 0]),
    'red'   : np.array([255, 0, 0]),
    'orange': np.array([255, 165, 0]),
    'green' : np.array([0, 255, 0]),
    'pink' : np.array([255,51, 143]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100])
}

#Included color names
COLOR_NAMES = sorted([i for i in list(COLORS.keys()) if i is not None])
COLOR_NAMES.insert(0,None)

# Used to map colors to integers
COLOR_TO_IDX = {
    None    : 0,
    'red'   : 1,
    'green' : 2,
    'blue'  : 3,
    'purple': 4,
    'orange': 5,
    'yellow': 6,
    'grey'  : 7,
    'pink'  : 8
}

# Reverse dict for index to color
IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

#######################################################################
# Object to Index
#######################################################################


# Map of object type to integers
OBJECT_TO_IDX = {
    # Originally 'empty'
    None         : 0,
    'unseen'        : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'food'          : 7,
    'box'           : 8,
    'goal'          : 9,
    'lava'          : 10,
    'agent'         : 11,
}

#Reverse dict for idx to object
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))

#######################################################################
# State to IDX
#######################################################################


# Map of state names to integers
STATE_TO_IDX = {
    'open'  : 0,
    'closed': 1,
    'locked': 2,
}

#######################################################################
# WorldObj (base class for all objects)
#######################################################################


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.state = 'open'
        self.score = 0
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], STATE_TO_IDX[self.state])

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == None or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type in ['box','food']:
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError

#######################################################################
# Base Disruption
#######################################################################


class BaseDisruption(object):

    """
    Base disruption class for cameleon objects
    """

    def __init__(self,
                 start_fn,
                 end_fn,
                 grid):

        self.start_fn = start_fn
        self.end_fn = end_fn
        self.active = False
        self.active_timesteps = 0
        self.grid = grid

    def add_game(self, game):
        """Add game to disruption

        :game: Game for disruption

        """
        self.game = game
        self._init_disruption(game.width, game.height)

    def place_objs(self):
        """

        """
        raise NotImplementedError


    def _init_disruption(self,width,height):
        """Initialize disruption

        :width: int: Width of grid
        :height: int: height of grid

        """
        raise NotImplementedError
#######################################################################
#    Food
#######################################################################


class Food(WorldObj):
    def __init__(self, color = 'purple'):
        super().__init__('food', color)

    def can_overlap(self):
        return True

    def render(self,img):
        """how to render object

        :img: location of item

        """
        color = COLORS[self.color]
        tri = point_in_triangle((0.19, 0.19),
                                (0.50, 0.81),
                                (0.81, 0.19))
        fill_coords(img,tri,color)

#######################################################################
# Goal
#######################################################################


class Goal(WorldObj):
    def __init__(self):
        super().__init__('goal', 'green')

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

#######################################################################
# Floor
#######################################################################


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        color = COLORS[self.color] / 2
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

#######################################################################
# Lava
#######################################################################


class Lava(WorldObj):
    def __init__(self):
        super().__init__('lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

#######################################################################
# Wall
#######################################################################


class Wall(WorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

#######################################################################
# Door
#######################################################################


class Door(WorldObj):
    def __init__(self, color='green', is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

#######################################################################
# Key
#######################################################################


class Key(WorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

#######################################################################
# Ball
#######################################################################


class Ball(WorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

#######################################################################
# Box
#######################################################################


class Box(WorldObj):
    def __init__(self, color = 'yellow', contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

#######################################################################
# Registry for later rendering
#######################################################################


#Reverse dict for idx to object
IDX_TO_INIT_OBJECT = {
    # Originally 'empty'
    0:  None,
    1:  None,
    2:  Wall(),
    3:  Floor(),
    4:  Door(),
    5:  Key(),
    6:  Ball(),
    7:  Food(),
    8:  Box(),
    9:  Goal(),
    10: Lava(),
    11: Ball(),
}


