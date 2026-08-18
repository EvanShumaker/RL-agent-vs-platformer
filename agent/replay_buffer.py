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
        # once it hits/passes capacity, deque automatically pops oldest element and replaces it
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """ store an experience"""
        self.buffer.append((state,action,reward,next_state, done))


    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences that gets returned as separate
        numpy arrays. Pytorch will convert to tensors later for training
        """

        # random.sample is better than .choices, because it doesnt allow for dupes
        batch = random.sample(self.buffer, batch_size)

        #batch is a list of 5 element tuples. Zipping it combines it all together. * isolates each col/feature, so i can 
        # assign one to each variable
        states, actions, rewards, next_states, dones = zip(*batch)

        return(
            np.array(states, dtype=np.float32),
            #pytorch expects floats for continuous vals, but ints for actions
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            # dones are float bc they need to be float later. No point putting bool here just to convert it later
            np.array(dones, dtype=np.float32),
            )


    def __len__(self):
        return len(self.buffer)