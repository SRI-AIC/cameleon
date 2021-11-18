#################################################################################
#
#             Project Title:  Cameleon Base Environment
#             Author:         Sam Showalter
#             Date:           2021-07-12
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import hashlib
from enum import IntEnum
import numpy as np
import copy

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_minigrid.rendering import *

from cameleon.utils.general import _tup_add, _tup_equal, _tup_mult
from cameleon.base_objects import *

#######################################################################
# Grid
#######################################################################


class Grid:
    """
    Represent a grid and operations on it
    for Cameleon. Base taken from Gym-minigrid
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.agent = None
        self.width = width
        self.height = height
        self.img = None
        self.last_img = None

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1  = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        try:
            assert i >= 0 and i < self.width
            assert j >= 0 and j < self.height
            return self.grid[j * self.width + i]
        except Exception as e:
            # Handles issues well enough
            # print("Error point",i,j)
            return 1

    def horz_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(self, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y+h-1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x+w-1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid

        :topX: Int:   Top x position of slice
        :topY: Int:   Top y position of slice
        :width: Int:  Slice width
        :height: Int: Slice height

        """

        # Build grid
        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(cls,
                    obj,
                    highlight=False,
                    tile_size=TILE_PIXELS,
                    subdivs=3):
        """
        Render a tile and cache the result

        :cls: NOT SURE YET
        :obj: WorldObj: Object to place
        :highlight: bool: Whether or not to highlight
        :tile_size: Int: Render tile size
        :subdivs: Int: Subdivisions
        """

        # Hash map lookup key for the cache
        # Need to look into this hash map

        key = (None, highlight, tile_size)
        if obj:
            # key = (obj.type, obj.color,highlight,tile_size)
            key = obj.encode() + key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size,
        highlight_mask=None):
        """
        Render this grid at a given scale

        :tile_size: Int:             Render tile size
        :highlight_mask: np.ndarray: Highlight mask

        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(self.agent.cur_pos, (i,j))
                if agent_here:
                    cell = self.agent

                tile_img = Grid.render_tile(
                    cell,
                    highlight=None,
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j+1) * tile_size
                xmin = i * tile_size
                xmax = (i+1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        # Just take difference in images
        self.img = img if self.last_img is None else img - self.last_img
        self.last_image = img

        return img

    def encode(self, agent, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid

        :agent:    Agent:      Agent object
        :vis_mask: np.ndarray: Visibility mask of agent

        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, 3), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):

                if vis_mask[i, j]:
                    if _tup_equal(agent.cur_pos, (i,j)):
                        array[j,i,:] = self.agent.encode()

                    else:
                        v = self.get(i, j)

                        if v is None:
                            array[j, i, 0] = OBJECT_TO_IDX[v]
                            array[j, i, 1] = COLOR_TO_IDX[v]
                            array[j, i, 2] = 0

                        else:
                            array[j, i, :] = v.encode()

        return array

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid

        :array: np.ndarray: Representation of grid

        """

        width, height, channels = array.shape
        assert channels == 3

        vis_mask = np.ones(shape=(width, height), dtype=np.bool)

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                type_idx, color_idx, state = array[i, j]
                v = WorldObj.decode(type_idx, color_idx, state)
                grid.set(i, j, v)
                vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

        return grid, vis_mask

#######################################################################
# Cameleon Env
#######################################################################


class CameleonEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        up = 2
        down = 3

        # Pick up an object
        pickup = 4
        # Drop an object
        drop = 5
        # Toggle/activate an object
        toggle = 6

        # Done completing task
        done = 7

    def __init__(
        self,
        grid_size=None,
        width=None,
        height=None,
        max_steps=1000,
        see_through_walls=False,
        reward_range = (-10,10),
        disruptions = [],
        seed=0,
        random_seed = True,):

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        self.mission = "Be more modular than MiniGrid"

        # Action enumeration for this environment
        self.actions = CameleonEnv.Actions

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        # Three channels -> object, color, state
        self.observation_space = spaces.Box(
                                    low=0,
                                    high=255,
                                    shape=(width, height, 3),
                                    dtype='uint8')
        self.observation_space = spaces.Dict({
            'image': self.observation_space})

        # Range of possible rewards
        self.reward_range = self.reward_range

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Current position and of the agent
        self.agent_pos = None
        self.agent = None
        self.grid = None

        # Set up disruptions
        self.base_disruptions = disruptions
        self.disruptions = []

        # Set random seed
        self.seed(seed=seed)
        self.reset_seed()

        # Initialize the state
        self.reset()


    def reset(self):

        # Get new random seed for identical envs
        self.reset_seed()

        # Current position and direction of the agent
        self.agent_pos = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        assert self.agent.cur_pos is not None

        # Item picked up, being carried, initially nothing
        self.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Instantiate all disruptions
        self._init_disruptions()

        # Return first observation
        obs = self.gen_obs()
        return obs


    def reset_seed(self):
        # Seed the random number generator
        # if random_seed:
        #     # Choose a random seed
        #     self.np_random = np.random
        # else:
        seed=self.old_np_random.randint(0,2**32)
        # print(seed)
        # print("Reset seed {}".format(int(a)*int(b)))
        self.np_random, _ = seeding.np_random(seed)

    def seed(self, seed = 1337):
        # Seed the random number generator
        # if random_seed:
        #     # Choose a random seed
        #     self.np_random = np.random
        # else:
        # print("Initial seed {}".format(seed))
        self.old_np_random, _ = seeding.np_random(int(seed))

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def _init_disruptions(self):
        """Initialize all disruptions

        """
        self.live_disruption = None
        self.disruptions = []
        disruptions = copy.deepcopy(self.base_disruptions)
        for d in disruptions:
            d.add_game(self)
            self.disruptions.append(d)

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall'          : 'W',
            'floor'         : 'L',
            'door'          : 'D',
            'key'           : 'K',
            'ball'          : 'A',
            'food'          : 'F',
            'box'           : 'B',
            'goal'          : 'G',
            'lava'          : 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent.cur_pos[0] and j == self.agent.cur_pos[1]:
                    str += 2 * '0'
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        return self.np_random.randint(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        return (self.np_random.randint(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        NOT USED
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """

        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.cur_pos = (i, j)

    def step(self, action):
        """Step function

        IMPLEMENT FOR YOUR OWN ENVS

        :action: Int: Encoded agent action

        """
        # Obs, reward, done, notes
        return None,None,None,None

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        # Encode the fully observable view into a numpy array
        image = self.grid.encode(agent = self.agent,vis_mask = None)

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass) - not used here
        # - a textual mission string (instructions for the agent)

        obs = {
            'image': image,
            'mission': self.mission
        }

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS//2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            #No need to highlight with full observability
            highlight_mask=None
        )

        return img

    def render(self, mode='human',
               close=False,
               highlight=False,
               agent = None,
               tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        highlight_mask = None
        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('Cameleon Grid - ')
            self.window.show(block=False)


        # Render the whole grid
        img = self.grid.render(
            tile_size,
            # agent = agent,
            highlight_mask=highlight_mask
        )

        if mode == 'human':
            self.window.set_caption(self.mission)
            self.window.show_img(img)

        return img

    def close(self):
        if self.window:
            self.window.close()
        return

#######################################################################
# Main
#######################################################################


