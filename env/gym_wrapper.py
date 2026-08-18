"""
A gymnasium wrapper around game.py. RL algorithms expect reset() and step(), so
thats what this file aims to provide. It lets me swap out other algorithms without
needing to change game code.
"""

import numpy as np
import env.constants as C
from env.game import Game


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

        return observation, reward, done, info


    def _build_observation(self):
        state = self.game.get_state()

        #normalize here
        # 

        return np.array([], dtype=np.float32) 