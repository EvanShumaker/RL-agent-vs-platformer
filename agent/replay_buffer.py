"""
Replay buffer for DQN. stores past (state, action, reward, next_state, done)
experiences and supports random batch sampling for training.
Uses a circular buffer.
"""

import random
from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """ store an experience"""
        self.buffer.append((state,action,reward,next_state, done))


    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences that gets returned as separate
        numpy arrays. Pytorch will convert to tensors later for training
        """

        batch = random.sample(...)

        return(np.array)


    def __len__(self):
        return len(self.buffer)