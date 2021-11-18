################################################################################
#
#             Project Title:  Canniballs Game, Includes stochasticity
#                             and subgoals
#             Author:         Sam Showalter
#             Date:           2021-07-08
#
#################################################################################


#################################################################################
#   Module Imports
#################################################################################

import os
import sys
import numpy as np
from functools import partial
import logging
from operator import add

from cameleon.register import register
from cameleon.grid import *
from cameleon.utils.general import _tup_add, _tup_equal, _tup_mult, _tup_subtract

#######################################################################
# Base canniball
#######################################################################


class BaseCanniball(Ball):

    """Base Canniball. Core functionality for all non-agent,
       live objects in environment
    """

    def __init__(self,game,color = "blue",score = 0):
        """Reference to game itself important to maintain state"""

        Ball.__init__(self,color)
        self.game = game
        self.alive = True
        self.score = score

        #Make sure all objects governed by same seed
        self.np_random = game.np_random if self.game else np.random
        self.new_pos = None
        self.prev_pos = None

        #Base object for priority moves
        self.p_moves = []
        self.type = 'ball'

    def can_pickup(self):
        """Cannot pick up opponents

        """
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.score)

    def is_valid(self,pos,
                 is_disruption = False,
                 reject_fn = None):
        """Determine if placement is valid

        :pos: tuple:          Position to check
        :reject_fn: function: Reject function

        """
        # Don't place the object on top of another object
        # Don't place the object where the agent is
        # Check if there is a filtering criterion
        if is_disruption:
            return True
        elif ((self.game.grid.get(*pos) != None) or
            _tup_equal(pos, self.game.agent_pos) or
            (reject_fn and reject_fn(self,pos))):
            return False

        return True

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high]
        with correct seed
        """

        return self.np_random.randint(low, high)

    def reset(self):
        """Reset object

        """
        self.init_pos = None
        self.priorities = []

    def close_to_agent(self,pos,agent_loc,
                       close = None):
        """
        :returns: If object not too close to agent

        """
        if not close:
            close = self.game.not_close_to_agent

        agent_pos  =self.game.agent.cur_pos

        pos_dif = (abs(agent_pos[0] - pos[0]),
                   abs(agent_pos[1] - pos[1]))

        close = (pos_dif[0] <= close) and (pos_dif[1] <= close)

        return close



    def find_pos(self,
                 priorities = [],
                 size = (3,3),
                 reject_fn  = None,
                 agent_loc = None,
                 is_disruption = False,
                 close = None,
                 max_tries = 100):
        """Core functionality for determining a position for an object.
        Used by all non-agent objects

        :priorities: List:    List of positions to attempt first
        :size: (int, int):    Size of box to consider for placing object
        :reject_fn: function: Rejection function
        :agent_loc: tuple:    Current location of agent (only used on init)
        :max_tries: Int:      Maximum attempts to place object

        """

        # Check priorities first
        for p in priorities:
            if (self.is_valid(p, is_disruption = is_disruption)
                and (not agent_loc
                or (agent_loc and not
                    self.close_to_agent(p, agent_loc, close = close)))):
                self.new_pos = p
                # print("----------")
                return

        tries = 0

        # Top corner of box to explore
        shift = (size[0] // 2)*-1

        # If agent needs to be initialized
        # place it anywhere on the grid
        if self.init_pos is None:
            top = (0,0)
            size = (self.game.grid.width, self.game.grid.height)

        # Otherwise, find top corner of placement box
        else:
            top = _tup_add(self.cur_pos, (shift,shift))

        #Look for positions
        while tries < max_tries:
            tries += 1

            # Candidate position
            pos = np.array((
                    self._rand_int(max(top[0],0), min(top[0] + size[0], self.game.grid.width)),
                    self._rand_int(max(top[1],0), min(top[1] + size[1], self.game.grid.height))
                ))

            #Make sure it is not no the agent at init (only time last bit is called)
            if (self.is_valid(pos,reject_fn)
                and (not agent_loc
                or (agent_loc and not self.close_to_agent(pos, agent_loc)))):
                self.new_pos = pos
                return

        # Not possible to move, emit a warning. This has never happened in testing
        logging.warning("ERROR: {} object not able to move".format(self.type))
        self.new_pos = pos
        self.prev_pos = self.cur_pos


#######################################################################
# Random Walker Caniball class
#######################################################################

class RandomWalker(BaseCanniball):

    """
    Random walking ball opponent
    - Weakest element - score = 1
    - Mostly stands still (easy target)
    - With random probability, walks around aimlessly

    """

    def __init__(self,game,color = "green",score =1,
                 stay_put_prob = 0.9):
        BaseCanniball.__init__(self,game,color,score)
        self.opponent_type = 'random_walker'
        self.stay_put_prob = stay_put_prob


    def move(self,
             update_pos = None):
        """Random walker chooses a random new available position

        :update_pos: Potential update position, not used for this obj

        """

        self.prev_pos = self.cur_pos

        #Item does not move
        if self.np_random.rand() < self.stay_put_prob:
            self.new_pos = self.cur_pos

        # Item shuffles around 3x3 box
        else:
            # No priority moves here
            self.find_pos(priorities=self.p_moves,
                        size = (3,3),
                        reject_fn = None)

#######################################################################
# Benign follower - for disruptions
#######################################################################


class Follower(BaseCanniball):

    """
    Random walking ball opponent
    - Score 1 greater than opponent always
    - Mostly stands still (easy target)
    - With random probability, walks around aimlessly

    """

    def __init__(self,game,color = "pink",score =1,
                 change_relation_prob = 0.2):
        BaseCanniball.__init__(self,game,color,score)
        self.opponent_type = 'follower'
        self.agent_relation = None
        self.change_relation_prob = change_relation_prob


    def move(self,
             update_pos = None):
        """Random walker chooses a random new available position

        :update_pos: Potential update position, not used for this obj

        """

        if not self.agent_relation:
            self.agent_relation = _tup_subtract(self.init_pos,
                                                self.game.agent.init_pos)

        agent_pos = self.game.agent.cur_pos
        ax,ay = agent_pos[0],agent_pos[1]
        pot_new_pos = _tup_add(agent_pos,
                                self.agent_relation)
        cant_move = ((pot_new_pos[0] < self.game.agent_range_squeeze) or
                     (pot_new_pos[0] >= self.game.width - self.game.agent_range_squeeze) or
                     (pot_new_pos[1] < self.game.agent_range_squeeze) or
                     (pot_new_pos[1] >= self.game.height - self.game.agent_range_squeeze))

        # The agent is at a range border and the open position is valid
        must_move_list = [((ax == self.game.agent_range_squeeze)
                      and (self.is_valid((ax-1,ay)))) ,
                     ((ay == self.game.agent_range_squeeze)
                      and (self.is_valid((ax,ay-1)))) ,
                     ((ax == self.game.width - self.game.agent_range_squeeze-1)
                      and (self.is_valid((ax+1,ay)))) ,
                     ((ay == self.game.height - self.game.agent_range_squeeze -1)
                      and (self.is_valid((ax,ay+1))))]

        must_move = True if True in must_move_list else False

        if cant_move:
            # print("Can't move")
            self.prev_pos = self.cur_pos
            self.new_pos = pot_new_pos

        elif must_move:
            # print("Must move")
            move_pos = [(ax-1,ay),(ax,ay-1),(ax+1,ay),(ax,ay+1)]
            for i in range(len(must_move_list)):
                if must_move_list[i]:
                    self.prev_pos = self.cur_pos
                    self.new_pos = move_pos[i]
                    # print(self.prev_pos)
                    # print(move_pos[i])
                    return

            assert False, "ERROR: No valid position found during must move"


        elif ((self.game.np_random.rand() < self.change_relation_prob)
            or (not self.is_valid(pot_new_pos))):
            relations =[(1,0),(-1,0),(0,1),(0,-1)]
            self.np_random.shuffle(relations)
            for r in relations:
                pos = _tup_add(agent_pos,r)
                if self.is_valid(pos):
                    # print("old rel {} - new rel {}".format(self.agent_relation,r))
                    self.agent_relation = r
                    self.prev_pos = self.cur_pos
                    self.new_pos = _tup_add(agent_pos,
                                            self.agent_relation)
                    return

            assert False, "ERROR: No valid position found during random explore"

        else:

            self.prev_pos = self.cur_pos
            self.new_pos = pot_new_pos



#######################################################################
# Bouncer Canniball Class
#######################################################################

class Bouncer(BaseCanniball):

    """
    Caniball that bounces according to basic physics
    with rare but occasional random movements. Also
    has a somewhat common tendency to pause (to make
    catching it easier)

    Moderately strong, with score = 2
    """

    def __init__(self,game,color  = "yellow",
                 score = 2,
                 rand_move_prob=0.02,
                 stay_put_prob = 0.3):

        BaseCanniball.__init__(self,game,color,score)

        # Can bounce diagonally, horizontally, or vertically
        # Movement sign flips when the agent finds itself colliding
        # with a different opponent
        movement_options =[(0,1),(1,0),(1,1)]
        self.movement = movement_options[self._rand_int(0,3)]
        self.move_sign = self.np_random.choice([1,-1])
        self.move_sign = (self.move_sign, self.move_sign)

        # Random movmement probability (low)
        # and staying put prob (moderate)
        self.rand_move_prob = rand_move_prob
        self.stay_put_prob = stay_put_prob

    def _determine_next_move_sign(self,pos):
        """Determine next movement sign based on
        where it collided

        :pos: potential next position

        """

        # Get correct sign for horizontal and vertical pieces
        sign = (-1,-1)

        # Get correct bouncing physics
        if self.movement[0] == self.movement[1]:
            if (((pos[1]) < 2) or
                (self.game.height - pos[1] < 2)):
                sign = (1,-1)
            else:
                sign = (-1,1)

        #Return movement sign
        return sign

    def move(self,
             update_pos = None):
        """Move object if possible, otherwise
        do nothing.

        :update_pos: tuple: Potential update position

        """

        self.prev_pos = self.cur_pos

        #Agent is staying put
        if (self.np_random.rand() < self.stay_put_prob):
            self.new_pos = self.cur_pos

        #Movement not random
        elif (self.np_random.rand() > self.rand_move_prob):

            # Two options: Current direction or bounce
            for i in range(2):
                shift = _tup_mult(self.movement, self.move_sign)
                pot_next_move = _tup_add(self.cur_pos,shift)

                # If direction chosen is valid
                if self.is_valid(pot_next_move):
                    self.new_pos = pot_next_move
                    return

                else:
                    #Flip the direction the agent is going if contact is made
                    sign = self._determine_next_move_sign(pot_next_move)
                    self.move_sign = _tup_mult(self.move_sign, sign)

            # Both directions blocked for bouncer, happens occasionally
            # In this case, bouncer stays put
            # print("Bouncer is trapped")
            self.new_pos = self.cur_pos

        else:
            # Choose a random position
            # Priorities are empty
            self.find_pos(priorities=self.p_moves,
                            size = (3,3),
                            reject_fn = None)

#######################################################################
# Chaser Canniball class
#######################################################################

class Chaser(BaseCanniball):

    """
    Chaser ball triggered by agent proximity
    - Strongest element - score = 3
    - Waits motionless until agent comes within a box range
    - Then, it chases with optimal movement with
      some probability of random action (makes escape easier

    """

    def __init__(self,
                 game,
                 score = 3,
                 #Trigger range should be an odd number
                 #Represents the box in which the piece is centered
                 trigger_range = 7,
                 color = "red"):

        BaseCanniball.__init__(self,game,color)
        self.score = score
        self.opponent_type = 'chaser'
        self.triggered = False
        self.trigger_range = trigger_range

        # Trigger is distance from the chaser
        # in any direction
        self.trigger = trigger_range//2
        self.rand_move_prob = 0.25


    def can_move(self):
        """Determine if ball can move
        i.e. is the agent close enough
        to trigger a response

        """
        i,j = self.cur_pos
        k,l = self.game.agent.cur_pos

        #Manhattan distance optimal movement options
        self.p_moves = np.array([(i + np.sign(k - i) ,j),
                             (i,j + np.sign(l - j))])

        #Shuffle the options so it is random
        self.np_random.shuffle(self.p_moves)

        # Have option to move optimally
        if self.triggered:
            return True

        # First move after being triggered
        elif (((np.abs(i - k) <= self.trigger) and
            (np.abs(j -l) <= self.trigger))):
            self.triggered = True

        # Return state of trigger
        return self.triggered

    def move(self,
             update_pos = None):
        """Move object if possible to chase agent
        - If chaser not active, do nothing
        - If chaser active but random move taken, do a random move
        - Otherwise, move Manhattan-distance style towards agent

        :update_pos: Potential update position, not used for this obj

        """

        # If agent cannot move
        self.prev_pos = self.cur_pos
        if not self.can_move():
            self.new_pos = self.cur_pos

        #Movement not random
        elif (self.np_random.rand() > self.rand_move_prob):

            for pos in self.p_moves:
                pos = tuple(pos)

                # Make sure position is valid and that
                # it is different from current position
                if ((self.is_valid(pos) and
                    (not _tup_equal(pos, self.cur_pos)))):
                    self.new_pos = pos

        else:
            # Choose a random position if either agent is trapped
            # or agent takes a random action
            self.find_pos(priorities=[],
                            size = (3,3),
                            reject_fn = None)



#######################################################################
# CanniballFood Class
#######################################################################

class CanniballFood(BaseCanniball):

    """Food triangle for Canniballs"""

    def __init__(self,game):
        BaseCanniball.__init__(self,game,color = "purple")
        self.alive = True
        self.type = "food"
        self.score = 0
        self.game = game
        self.new_pos = None
        self.cur_pos = None
        # In the case this is a dummy entity, remove random seed
        self.np_random = self.game.np_random if self.game else np.random

    def can_overlap(self):
        """Agent can overlap with food

        """
        return True

    def reset(self):
        """Reset the food

        """
        BaseCanniball.reset(self)
        self.alive = True

    def move(self,
             update_pos =None):
        """Move food to random new position only if
        it has been eaten

        :update_pos: Potential update position, not used for this obj

        """

        self.new_pos = self.cur_pos
        if not self.alive:
            self.find_pos()

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
# Agent
#######################################################################

class CanniballAgent():

    """Agent for Canniballs game. Another ball
       with a starting score of 1. Goal is to eat
       enough food to consume opponents as fast as possible

    """

    def __init__(self, game):
        """

        :init_pos: starting position of agent

        """
        self.game = game
        self.color = 'blue'
        self.type = 'agent'
        self.score = 1
        self.state = 'open'
        self.init_pos = None
        self.cur_pos = None
        self.prev_pos = None
        self.range_squeeze = game.agent_range_squeeze
        self.alive = True

    def find_pos(self,
                 agent_loc = None,
                 max_tries = 200):
        """
        Find position for the agent (only used at start,
        the game dynamics move the agent for remainder)
        """

        dims = (self.game.width, self.game.height)


        for i in range(max_tries):
            self.init_pos = (self.game.np_random.randint(self.range_squeeze,
                                                         self.game.width - self.range_squeeze),
                            self.game.np_random.randint(self.range_squeeze,
                                                        self.game.height - self.range_squeeze))

            if not (self.game.grid.get(*self.init_pos)):
                self.new_pos = self.init_pos
                self.cur_pos = self.init_pos
                return

        assert False, "ERROR: Cannot place agent"

    def can_overlap(self):
        """Nothing can overlap with agent

        """
        return False

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.score)

    def move(self, obj, update_pos=None):
        """Update agent position

        :obj: Potential object that the agent is overlapping with
        :update_pos: Position agent will be moving to

        """
        #Update position information
        self.prev_pos = self.cur_pos
        self.cur_pos = update_pos

        # No collisions
        if obj == None:
            return

        #If collision with canniball
        if obj.type == 'ball':
            # If stronger, then eat
            if self.score > obj.score:
                obj.alive = False
                self.score += 1

            # Otherwise, you die
            else:
                self.cur_pos = self.prev_pos
                self.alive = False

        # If it is food, eat the food
        elif obj.type == 'food':
            obj.alive = False
            self.score += 1

        #Otherwise, don't move since you can't overlap
        elif not obj.can_overlap():
            self.cur_pos = self.prev_pos

    def render(self, img):
        """ How to render the CanniballAgent """
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


#######################################################################
# Disruptions
#######################################################################


class CanniballsNstepAvoidanceDisruption(BaseDisruption):

    """Adds a disruption to the game that is built
    to elicit a specific reponse in the agent"""

    def __init__(self,
                 n,
                 start_fn = lambda game: game.step_count >= 0,
                 end_fn = lambda self: self.active_timesteps >= self.n-1,
                 grid = Grid,
                 num_objs = 3,
                 squeeze = 0,
                 obj_type = "follower"):


        BaseDisruption.__init__(self,
                               start_fn,
                               end_fn,
                               grid = Grid)
        self.n = n
        self.num_objs = num_objs
        self.obj_type = obj_type
        self.canniballs = []

    def add_game(self, game):
        """Add game to disruption

        :game: Game for disruption

        """
        self.game = game
        self._init_disruption(game.width, game.height)

    def place_objs(self):
        """

        """
        obj_dict = {"food": CanniballFood,
                    "chaser":Chaser,
                    "bouncer": Bouncer,
                    "random_walker":RandomWalker,
                    "follower": Follower}

        assert self.obj_type in obj_dict,\
            "ERROR: Object type {} not found in registry: {}"\
            .format(self.obj_type,
                    list(obj_dict.keys()))

        obj = obj_dict[self.obj_type]

        ax,ay = self.game.agent.cur_pos

        self.positions = []
        # dirs left, right, top, bottom
        dir_dists = np.array([ax - 1,
                               # 2 to resolve for 0 indexing
                               self.game.width - 2 - ax,
                               ay - 1,
                               self.game.height - 2 - ay])

        missing_dir = self.game.np_random.choice(
                        np.argwhere(dir_dists >= 2).flatten())

        # Dirs
        self.positions = [(ax-1,ay),
                        (ax+1,ay),
                        (ax,ay-1),
                        (ax,ay+1)]
        del self.positions[missing_dir]

        for i in range(self.num_objs):
            pos = self.positions[i]
            active_obj = obj(game=self.game,
                             score=self.game.agent.score+1)

            active_obj.find_pos(priorities = [pos],
                    agent_loc = self.game.agent.cur_pos,
                    is_disruption=True)
            active_obj.init_pos = active_obj.new_pos
            active_obj.cur_pos = active_obj.init_pos
            self.canniballs.append(active_obj)

            # Set the objects
            self.grid.set(*active_obj.cur_pos,
                        active_obj)


    def _init_disruption(self,width,height):
        """Initialize disruption

        :width: int: Width of grid
        :height: int: height of grid

        """
        # Generate the surrounding walls
        self.canniballs = []
        self.grid = self.grid(width,height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.agent = self.game.agent

#######################################################################
# Food Distance consumption disruption
#######################################################################

class CanniballsNstepConsumptionDisruption(BaseDisruption):

    """Adds a disruption to the game that is built
    to elicit a specific reponse in the agent"""

    def __init__(self,
                 start_fn = lambda game: game.step_count >= 0,
                 end_fn = lambda self: self.active_timesteps >= self.max_steps,
                 grid = Grid,
                 num_objs = 3,
                 slack_steps =0,
                 squeeze = 0,
                 obj_type = "food"):


        BaseDisruption.__init__(self,
                               start_fn,
                               end_fn,
                               grid = Grid)

        self.num_objs = num_objs
        self.slack_steps = slack_steps
        self.max_steps = num_objs + slack_steps
        self.obj_type = obj_type
        self.canniballs = []

    def add_game(self, game):
        """Add game to disruption

        :game: Game for disruption

        """
        self.game = game
        self._init_disruption(game.width, game.height)

    def place_objs(self):
        """

        """
        obj_dict = {"food": CanniballFood,
                    "chaser":Chaser,
                    "bouncer": Bouncer,
                    "random_walker":RandomWalker,
                    "follower": Follower}

        assert self.obj_type in obj_dict,\
            "ERROR: Object type {} not found in registry: {}"\
            .format(self.obj_type,
                    list(obj_dict.keys()))

        obj = obj_dict[self.obj_type]

        # Start near agent
        ax,ay = self.game.agent.cur_pos

        # Dirs
        for i in range(self.num_objs):

            positions = [(ax-1,ay),
                            (ax+1,ay),
                            (ax,ay-1),
                            (ax,ay+1)]

            self.game.np_random.shuffle(positions)
            active_obj = obj(game=self.game)

            active_obj.find_pos(priorities = positions,
                    agent_loc = self.game.agent.cur_pos)
            active_obj.init_pos = active_obj.new_pos
            active_obj.cur_pos = active_obj.init_pos
            self.canniballs.append(active_obj)
            ax,ay = active_obj.cur_pos[0], active_obj.cur_pos[1]

            # Set the objects
            self.grid.set(*active_obj.cur_pos,
                        active_obj)


    def _init_disruption(self,width,height):
        """Initialize disruption

        :width: int: Width of grid
        :height: int: height of grid

        """
        # Generate the surrounding walls
        self.canniballs = []
        self.grid = self.grid(width,height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.agent = self.game.agent


#######################################################################
# Corners Disruption
#######################################################################


class CanniballsCornersDisruption(BaseDisruption):

    """Adds a disruption to the game that is built
    to elicit a specific reponse in the agent"""

    def __init__(self,
                 start_fn = lambda game: game.step_count >= 10,
                 end_fn = lambda self: self.active_timesteps >= 8,
                 grid = Grid,
                 squeeze = 0,
                 obj_type = "food"):


        BaseDisruption.__init__(self,
                               start_fn,
                               end_fn,
                               grid = Grid)
        self.squeeze = squeeze
        self.obj_type = obj_type
        self.canniballs = []

    def add_game(self, game):
        """Add game to disruption

        :game: Game for disruption

        """
        self.game = game
        self._init_disruption(game.width, game.height)

    def place_objs(self):
        """

        """
        obj_dict = {"food": CanniballFood,
                    "chaser":Chaser,
                    "bouncer": Bouncer,
                    "random_walker":RandomWalker}

        assert self.obj_type in obj_dict,\
            "ERROR: Object type {} not found in registry: {}"\
            .format(self.obj_type,
                    list(obj_dict.keys()))

        obj = obj_dict[self.obj_type]

        for corner in self.corners:
            active_obj = obj(game=self.game)

            active_obj.find_pos(priorities = [corner],
                    agent_loc = self.game.agent.cur_pos,
                    is_disruption=True)
            active_obj.init_pos = active_obj.new_pos
            active_obj.cur_pos = active_obj.init_pos
            self.canniballs.append(active_obj)

            # Set the objects
            self.grid.set(*active_obj.cur_pos,
                        active_obj)


    def _init_disruption(self,width,height):
        """Initialize disruption

        :width: int: Width of grid
        :height: int: height of grid

        """
        # Generate the surrounding walls
        self.canniballs = []
        self.grid = self.grid(width,height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.agent = self.game.agent
        s = self.squeeze
        self.corners = [(1+s,1+s),
                        (1+s,width-2-s),
                        (height -2-s,1+s),
                        (width-2-s,height-2-s)]


#######################################################################
# Cannibals game
#######################################################################

class Canniballs(CameleonEnv):

    """Canniball game for Cameleon RL Experiments"""

    # Custom set of Actions relative to Base Cameleon Grid
    class Actions(IntEnum):
        left = 0
        right = 1
        up = 2
        down = 3
        # done = 4

    def __init__(self,
                 n_random_walkers = 4,
                 n_bouncers = 3,
                 n_chasers = 2,
                 n_food = 3,
                 size = 42,
                 max_steps = 300,
                 agent_init_space = 1,
                 replenish_food = True,
                 agent_range_squeeze = 1,
                 disruptions = [],
                 disruptions_only = False,
                 score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":1,
                                 "bouncer":2,
                                 "chaser":3}
                 ):

        self.n_random_walkers = n_random_walkers
        self.n_bouncers = n_bouncers
        self.n_chasers = n_chasers
        self.not_close_to_agent = agent_init_space
        self.n_food = n_food
        self.alive_canniballs = 0
        self.score_roster = score_roster
        self.replenish_food = replenish_food
        self.agent_range_squeeze = agent_range_squeeze
        self.disruptions_only = disruptions_only

        # Initialize everything
        super().__init__(grid_size = size,disruptions=disruptions)

        # Standard grid when disruption taking place
        self.live_disruption = None
        self.dormant_grid = None
        self.dormant_canniballs = None

        # Action enumeration for this environment
        self.actions = Canniballs.Actions
        self.max_steps = max_steps

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))
        self.mission = "Defeat all enemy balls by eating food\n and growing more powerful than them before attack"


    def _init_place_obj(self, obj,
                        agent = False,
                        agent_loc = None):
        """Place object at start of game

        :obj: tuple:       Object to be placed
        :agent: bool:      Is the object the agent
        :agent_loc: tuple: Location of agent, used for all objects so no overlap

        """
        obj.find_pos(agent_loc = agent_loc)
        obj.init_pos = obj.new_pos
        obj.cur_pos = obj.init_pos
        if not agent:
            self.grid.set(*obj.cur_pos, obj)

    def disruptions_active(self):
        """Check if there are any active disruptions

        :returns:

        """

        for d in self.disruptions:
            if d.start_fn(self):
                return d

        return None

    def check_disruptions(self):
        """
        Check to see if a disruption has become active.
        If it has, then swap grids. If disruption is
        not inactive, remove it and reset standard
        grid (return agent to original spot).
        """

        if self.live_disruption and not self.live_disruption.end_fn(self.live_disruption):
            self.live_disruption.active_timesteps += 1
            return None

        elif self.live_disruption:
            self.live_disruption.alive = False
            self.agent.cur_pos = self.live_disruption.agent_init_pos
            self.live_disruption = None
            self.grid = self.dormant_grid
            self.canniballs = self.dormant_canniballs

        else:
            disruption = self.disruptions_active()
            if disruption:
                self.live_disruption = disruption
                self.live_disruption.agent_init_pos = self.agent.cur_pos
                self.dormant_grid = self.grid
                self.grid = self.live_disruption.grid
                disruption.place_objs()
                self.disruptions.remove(disruption)
                self.dormant_canniballs = self.canniballs
                self.canniballs = self.live_disruption.canniballs

        return ((not self.live_disruption) and len(self.disruptions) == 0)


    def _place_obj(self, obj,
                   update_pos = None):
        """Place object after game has started
        and nullify previous cell

        :obj: Object to be moved
        :update_pos: Potential position to update

        """
        prev_pos = obj.cur_pos
        obj.move(update_pos = update_pos)
        obj.cur_pos = obj.new_pos
        self.grid.set(*obj.cur_pos, obj)

        #Don't remove object unless it has moved to new location
        if not _tup_equal(obj.cur_pos, prev_pos):
            self.grid.set(*prev_pos,None)

    def _init_obj(self, num,obj,score, store):
        """Instantiate objects and store them
        as well as place them

        :num: Int:      Number of objects
        :obj: WorldObj: Object class to instantiate and place
        :score: Int:    Score object should start with
        :store: List:   Where they should be stored

        """

        for i in range(num):

            #Instantiate object
            active_obj = obj(game = self)
            active_obj.score = score

            #Add to canniballs
            store.append(active_obj)

            # If the object is a ball, add to count
            if active_obj.type == 'ball':
                self.alive_canniballs += 1

            # Randomly place objects for game start
            self._init_place_obj(active_obj,
                                 agent_loc = self.agent.cur_pos)



    def _gen_grid(self, width, height):
        """
        Generate the grid of the environment for Canniballs

        :width: Width of grid
        :height: Height of grid

        """

        #Reset alive canniballs (will be filled)
        self.alive_canniballs = 0

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place obstacles
        self.dormant_canniballs = []
        self.canniballs = []
        canniball_roster =  {CanniballFood: (self.n_food,
                                             self.score_roster['food']),
                             RandomWalker:(self.n_random_walkers,
                                           self.score_roster['random_walker']),
                             Bouncer:(self.n_bouncers,
                                      self.score_roster['bouncer']),
                             Chaser:(self.n_chasers,
                                     self.score_roster['chaser'])}


        #Init Canniball Agent
        self.agent = CanniballAgent(self)
        self.agent.score = self.score_roster['agent']
        self.grid.agent = self.agent

        # Place the agent
        self._init_place_obj(self.agent,
                             agent = True)

        # Initialize all of the objects
        for obj,(num,score) in canniball_roster.items():
            self._init_obj(num,obj,score,self.canniballs)

    def move_objects(self):
        """Move all non-agent objects

        """

        # Food is also placed with this functionality
        for i,item in enumerate(self.canniballs):

            # Remove eaten Canniballs
            if item.type == 'ball':
                #Check if ball has been killed
                if not item.alive:
                    self.alive_canniballs -= 1
                    self.grid.set(*item.cur_pos,None)
                    del self.canniballs[i]

            # If you are not replenishing food,
            # Then delete it
            elif ((item.type == 'food') and
                  (not item.alive) and
                  (not self.replenish_food)):
                    self.grid.set(*item.cur_pos,None)
                    del self.canniballs[i]

            # Only view if item is alive
            if item.alive:
                self._place_obj(item)


    def reward_update(self,
                      food = False,
                      ate = False,
                      mult = 1,
                      food_reward = 0.5,
                      base = 0.00):
        """Update reward for agent

        :food: Boolean on if food eaten
        :ate: Boolean on if agent ate ball or was eaten
        :mult: Reward multiplier
        :food_reward: Reward for food
        :base: Base reward for being alive

        """
        reward = 0
        # Extra negative reward if agent gets stuck
        if _tup_equal(self.agent.cur_pos,
                      self.agent.prev_pos):
            base = 0.05
        # Small reward for food
        if food:
            reward += food_reward
        # Ate opponent or was eaten
        elif ate:
            if (mult < 0):
                reward += mult
            else:
                reward += mult*self.agent.score*((self.max_steps - self.step_count)
                                                 /self.max_steps)
        # Win the game
        if self.alive_canniballs == 0:
            reward += 50

        # Base reward to keep agent moving
        # In this case, we set it to nothing
        return reward - base

    def crossover(self,pos_update):
        """Check if an object crossed over the
        agent. This means the balls crossed over
        each other but did not collide in same cell.
        This is also a death for the agent in some cases
        or a successful canniballism by the agent

        :pos_update: Update position to check for crossover

        """
        # Get object at agent's current position
        # (in case something moved over agent)
        obj = self.grid.get(*self.agent.cur_pos)

        if (obj and
            (obj.type == 'ball') and
            # Don't do this on init, check there is a previous position first
            self.agent.prev_pos):
            return True

        return False

    def is_valid(self, pos):
        """Check if position is valid

        :pos:

        """
        return (((pos[0] >=0) and (pos[0] < self.width)) and
                (pos[1] >= 0) and (pos[1] < self.height))

    def reset(self):
        """Reset, inherits from parent

        """
        super().reset()
        self.check_disruptions()
        return self.gen_obs()


    def step(self,action):
        """Step function to step environment

        :action: action_space.Action: Action by agent

        """
        done = False
        reward = 0

        # Update position of agent
        pos_update_dict = {
            self.actions.left : (-1,0),
            self.actions.right : (1,0),
            self.actions.up : (0,-1),
            self.actions.down : (0,1),
            # self.actions.done : (0,0),
        }

        self.step_count += 1

        if (self.alive_canniballs == 0):
            reward = self.reward_update()
            done = True

        # Update the state of the agent after action taken
        pos_update = _tup_add(self.agent.cur_pos,
                            pos_update_dict[action])

        if not self.is_valid(pos_update):
            pos_update = self.agent.cur_pos

        # Conditions if new position is not empty
        new_pos_obj = self.grid.get(*pos_update)

        #Crossover, where agent and ball don't directly collide in the same cell
        # but overlap, which is considered a collision
        if (not done) and self.crossover(pos_update):
            old_score = self.agent.score
            obj = self.grid.get(*self.agent.cur_pos)
            self.agent.move(obj, pos_update)
            if not self.agent.alive:
                done = True
                reward = self.reward_update(ate = True,
                               mult = old_score - obj.score - 1)

            # Remove object if it was eaten
            else:
                self.grid.set(*obj.cur_pos,None)
                reward = self.reward_update(ate = True,
                               mult = obj.score)

        # No new position
        elif (not done) and (not new_pos_obj):
            self.agent.move(new_pos_obj, pos_update)
            reward = self.reward_update()

        # If new position has something there. Can happen with crossover
        if (not done) and new_pos_obj:
            old_score = self.agent.score
            self.agent.move(new_pos_obj,pos_update)
            if new_pos_obj.type == 'food':

                # Set the current position to none
                self.grid.set(*new_pos_obj.cur_pos,None)

                #Respawn food to new location
                if self.replenish_food:
                    new_pos_obj.reset()
                    self._init_place_obj(new_pos_obj,
                                         agent_loc = self.agent.cur_pos)

                reward = self.reward_update(food=True)

            elif new_pos_obj.type == 'ball':
                if self.agent.alive:
                    # Positive reward for eating another opponent
                    reward = self.reward_update(ate = True,mult = new_pos_obj.score)
                else:
                    #Large negative reward for losing
                    reward = self.reward_update(ate = True, mult = (old_score - new_pos_obj.score -1))
                    done = True

            else:
                reward = self.reward_update()


        #Move objects so agent has a chance
        self.move_objects()

        if self.step_count >= self.max_steps:
            done = True

        # Check for an updated state on disruptions
        # This may also revert things back to normal
        if self.disruptions_only:
            done = (self.check_disruptions() or done)

        obs = self.gen_obs()

        return obs, reward, done, {}

#################################################################################
#   Subclass Cameleon Environments for Canniballs
#################################################################################


class CanniballsEnv42x42(Canniballs):

    """Large minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 42)


class CanniballsEnv22x22(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 22)


class CanniballsEnvEasy12x12(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1)

class CanniballsEnvEasyCornerDisruption12x12(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1,
                         disruptions=[CanniballsCornersDisruption(
                             start_fn = lambda game: game.step_count >= 5,
                             end_fn = lambda self:self.active_timesteps >=20,
                             obj_type = "food")])


class CanniballsEnvEasyFakeCornerDisruption12x12(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1,
                         disruptions=[CanniballsCornersDisruption(
                             start_fn = lambda game: False,
                             end_fn = lambda self:self.active_timesteps >=10,
                             squeeze = 2,
                             obj_type = "chaser")]
                         )


class CanniballsEnvHardCornerDisruption12x12(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1,
                         replenish_food=False,
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5},
                         disruptions=[CanniballsCornersDisruption(
                             start_fn = lambda game: game.step_count >= 5,
                             end_fn = lambda self:self.active_timesteps >=10,
                             squeeze = 2,
                             obj_type = "chaser")])


class CanniballsEnvHardFakeCornerDisruption12x12(Canniballs):

    """Medium minigrid environment
    for canniballs"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1,
                         replenish_food=False,
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5},
                         disruptions=[CanniballsCornersDisruption(
                             start_fn = lambda game: False,
                             end_fn = lambda self:self.active_timesteps >=10,
                             squeeze = 2,
                             obj_type = "chaser")])


class CanniballsEnvMedium12x12(Canniballs):

    """Small minigrid environment
    for canniballs with hard env dynamics"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 4,
                         n_random_walkers=3,
                         n_bouncers=2,
                         n_chasers=1,
                         replenish_food=False,
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5}
                         )


class CanniballsEnvMediumNstepAvoidance12x12(Canniballs):

    """Small minigrid environment
    for canniballs with med env dynamics
    and avoidance disruption"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 4,
                         n_random_walkers=3,
                         n_bouncers=2,
                         n_chasers=1,
                         replenish_food=False,
                         agent_init_space = 0,
                         agent_range_squeeze=2,
                         disruptions_only=True,
                         disruptions=[CanniballsNstepAvoidanceDisruption(
                                        n=500,
                                        num_objs=3)
                                      ],
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5}
                         )


class CanniballsEnvMediumNstepConsumption12x12(Canniballs):

    """Small minigrid environment
    for canniballs with med env dynamics
    and avoidance disruption"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 4,
                         n_random_walkers=3,
                         n_bouncers=2,
                         n_chasers=1,
                         replenish_food=False,
                         agent_init_space = 0,
                         disruptions_only=True,
                         disruptions=[CanniballsNstepConsumptionDisruption(
                                        num_objs=5)
                                      ],
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5}
                         )

class CanniballsEnvHard12x12(Canniballs):

    """Small minigrid environment
    for canniballs with hard env dynamics"""

    def __init__(self):
        super().__init__(size = 12,
                         n_food = 2,
                         n_random_walkers=2,
                         n_bouncers=1,
                         n_chasers=1,
                         replenish_food=False,
                         score_roster = {"agent":1,
                                 "food":0,
                                 "random_walker":2,
                                 "bouncer":4,
                                 "chaser":5}
                         )

#######################################################################
# Registery Cameleon Environments
#######################################################################


register(
    id='Cameleon-Canniballs-Easy-42x42-v0',
    entry_point='cameleon.envs:CanniballsEnv42x42'
)


register(
    id='Cameleon-Canniballs-Easy-22x22-v0',
    entry_point='cameleon.envs:CanniballsEnv22x22'
)

register(
    id='Cameleon-Canniballs-Easy-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvEasy12x12'
)

register(
    id='Cameleon-Canniballs-Easy-Corner-Disruption-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvEasyCornerDisruption12x12'
)

register(
    id='Cameleon-Canniballs-Easy-Fake-Corner-Disruption-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvEasyFakeCornerDisruption12x12'
)


register(
    id='Cameleon-Canniballs-Hard-Corner-Disruption-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvHardCornerDisruption12x12'
)

register(
    id='Cameleon-Canniballs-Medium-NStep-Avoidance-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvMediumNstepAvoidance12x12'
)


register(
    id='Cameleon-Canniballs-Medium-NStep-Consumption-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvMediumNstepConsumption12x12'
)

register(
    id='Cameleon-Canniballs-Hard-Fake-Corner-Disruption-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvHardFakeCornerDisruption12x12'
)

register(
    id='Cameleon-Canniballs-Medium-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvMedium12x12'
)

register(
    id='Cameleon-Canniballs-Hard-12x12-v0',
    entry_point='cameleon.envs:CanniballsEnvHard12x12'
)









