"""
A gymnasium wrapper around game.py. RL algorithms expect reset() and step(), so
thats what this file aims to provide. It lets me swap out other algorithms without
needing to change game code.
"""

import numpy as np
import env.constants as C
from env.game import Game
from math import sqrt



class PlatformerEnv:
    """
    Gymlike interface:
    reset() returns an observation
    step(action) returns (observation, reward, done, info)
    observation is our normalized array in numpy
    """

    OBS_SIZE = 6 # player_x, player_y, vx, vy, on_ground, dist_to_goal

    def __init__(self):
        self.game=Game()

    def reset(self):
        self.game.reset()
        return self._build_observation()


    def step(self, action):
        reward, done = self.game.step(action)
        observation = self._build_observation()
        info = {} # might need later to pass extra info

        return observation, reward, done, info


    def _build_observation(self):
        """
        Normalization of traits.

        Float 32 dtype is probably the best, i think pytorch expects it. 
        Python / numpy might default to 64bit, which is less efficient in training
        """
        state = self.game.get_state()

        #normalize here. Every trait/feature becomes a decimal in [0,1], or [-1,1] for velocity
        norm_x = state["player_x"] / C.SCREEN_WIDTH
        norm_y = state["player_y"] / C.SCREEN_HEIGHT
        norm_vx = state["player_vx"] / C.MOVE_SPEED     # the constant LR speed
        norm_vy = state["player_vy"] / C.TERMINAL_VELOCITY

        if state["on_ground"]:
            on_ground = 1.0
        else:
            on_ground = 0.0

        # for goal distance, ill normalize against the max diagonal of the screen, just in case...
        max_dist = sqrt((C.SCREEN_HEIGHT)**2 + (C.SCREEN_WIDTH)**2)
        norm_dist_to_goal = state["distance_to_goal"] / max_dist

        return np.array([
            norm_x,
            norm_y,
            norm_vx,
            norm_vy,
            on_ground,
            norm_dist_to_goal,
        ], dtype=np.float32) 