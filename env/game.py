"""
This is the main game loop and logic. 
"""

import env.constants as C
from env.entities import Player, Platform, Goal
from math import sqrt


class Game:
    """Other than init, ill need a function to call the player reset(), a step()
    to move it forwards a frame, function to call check_goal_reached(), and a 
    function to update the rewards? like -.01 every turn"""
    def __init__(self):
        # this loop SHOULD work, C.PLATFORMS is the list defined in constants.
        # the Platform init() needs 4 args, so the * should separate them
        self.platforms = [Platform(*p) for p in C.PLATFORMS]
        # same extrapolation for this?
        self.goal = Goal(*C.GOAL)
        self.player = Player()
        self.steps_elapsed = 0
        self.done = False
        self.reset()

    
    def reset(self):
        """resets the game"""
        self.player.reset()
        self.done = False
        self.steps_elapsed = 0
        # leading _ is convention for variables that arent meant to be used outside of class
        self._prev_dist_to_goal = self._distance_to_goal()

    
    def step(self, action):
        """
        loads the next frame of the game based on what the player does.
        I just need to call the code in the player class in entities to 
        apply action, apply physics, etc.
        """
        if self.done:
            raise RuntimeError("Step() called after episode ended. Call reset() again first")

        self.player.apply_action(action)
        self.player.apply_physics()
        # need to give Game platforms info so it can pass it here
        self.player.resolve_platform_collisions(self.platforms)
        self.player.check_pit_fall(C.PIT_X_RANGE)

        self.steps_elapsed += 1

        return self._compute_reward(), self._check_done()


    def _compute_reward(self):
        """
        Just like with varibales, leading _ means this func isnt called outside
        of this class. This func just calculates distance and progress to get the current reward.
        Its called once per step/frame, so i just add one -.01 at the start
        """
        reward = C.REWARD_STEP

        dist = self._distance_to_goal()
        progress = self._prev_dist_to_goal - dist
        reward += progress * C.REWARD_PROGRESS_SCALE
        self._prev_dist_to_goal = dist

        if not self.player.alive:
            reward += C.REWARD_DEATH
        elif self.player.check_goal_reached(self.goal):
            reward += C.REWARD_GOAL
        
        return reward


    def _check_done(self):
        """
        basically all the cases in which the agent / player needs to reset
        """
        if not self.player.alive:
            return True
        if self.player.check_goal_reached(self.goal):
            return True
        if self.steps_elapsed >= C.MAX_EPISODE_STEPS:
            return True
        return False


    # not correct, fix later
    def _distance_to_goal(self):
        """
        check the distance to the nearest edge. Theres 8 cases:
        Above, below, left, and right. Then the 4 diagonals.
        Check left, mid, and right thirds with bools?
        Then check top, middle, and bottom thirds.
        Then do a bunch of if stmts comparing the bools
        """
        # short player position variables, based on the center. x and y are for top left corner of player
        px = self.player.x + self.player.width / 2
        py = self.player.y + self.player.height / 2
        # top and bottom of the goal
        topGY = self.goal.y
        botGY = self.goal.y + self.goal.height
        # left and right bounds of the goal
        leftGX = self.goal.x
        rightGX = self.goal.x + self.goal.width

        if px < leftGX:
            if py < topGY:
                # up and left
                return sqrt((px-leftGX)**2 + (py-topGY)**2)
            if py > botGY:
                # down and left
                return sqrt((px-leftGX)**2 + (py-botGY)**2)
            else:
                # just left of goal bounds
                return leftGX - px
        
        if px > rightGX:
            if py < topGY:
                # up and right
                return sqrt((px-rightGX)**2 + (py-topGY)**2)
            if py > botGY:
                # down and right
                return sqrt((px-rightGX)**2 + (py-botGY)**2)
            else:
                # just right of goal bounds
                return px - rightGX
            
        else:
            # this means its right above, below, or inside the goal
            if py < topGY:
                return topGY - py
            if py > botGY:
                return py - botGY
        
        return 0


    def get_state(self):
        """
        maybe useful later. use this to print out all the game information
        should i print? or return as a list or dict?
        dict is better because i can actually use it as a multi helper function!
        """
        return{
            "player_x": self.player.x,
            "player_y": self.player.y,
            "player_vx": self.player.vx,
            "player_vy": self.player.vy,
            "on_ground": self.player.on_ground,
            "alive": self.player.alive,
            "steps_elapsed": self.steps_elapsed,
            "platforms": [(p.x,p.y,p.width,p.height) for p in self.platforms],
            "goal": (self.goal.x, self.goal.y, self.goal.width, self.goal.height),
        }
