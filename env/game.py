"""
This is the main game loop and logic. 
"""

import env.constants as C
from env.entities import Player, Platform, Goal


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

        self.player.apply_action(action)
        self.player.apply_physics()
        # need to give Game platforms info so it can pass it here
        self.player.resolve_platform_collisions()


    def _compute_reward(self):
        """
        Just like with varibales, leading _ means this func isnt called outside
        of this class. This func just calculates distance and progress to get the current reward.
        Its called once per step/frame, so i just add one -.01 at the start
        """
        reward = C.REWARD_STEP

        dist = self._distance_to_goal()


    def _distance_to_goal():
        

