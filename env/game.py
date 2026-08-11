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

    
    def reset(self):
        """resets the game"""
        self.player.reset()

    
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
