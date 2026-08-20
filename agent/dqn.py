"""
This is the DQN agent. It uses Q network architecture, epsilon greedy action
selection, and a training step to update the network from the random sampled experience
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import env.constants as C
from env.gym_wrapper import PlatformerEnv


class QNetwork(nn.Module):
    """
    This network takes the observation vector, outputs a Q val for each
    possible action.

    If the training target used the same network the code actively updates, it 
    causes issues for the code and causes instability. 'target_network' is a snapshot
    that periodically updates to give the training process a stable target to aim for.

    QNetwork inherets from nn.Module class.
    """

    def __init__(self, obs_size, num_acitons):
        super().__init__()
        self.net = nn.Sequential() # add parameters to this


    def forward(self, x):
        """
        Passes the observation x thru the network
        """
        return self.net(x)


class DQNAgent:
    def __init__():
        """Setup target network here"""


    def get_epsilon(self):
        """Linear decay epsilon from start to end"""


    def select_action(self, observation):
        """
        This is epsilon greedy: With probability epsilon, pick random action (observation)
        Otherwise, pick action with highest predicted Q val (exploitation)
        """


    def train_step(self, replay_buffer, batch_size):
        """
        Sample a batch from buffer and do gradient descent update.
        Returns loss value for logging, and None if buffer doesnt have 
        enough experience
        """


    def update_target_network(self):
        """
        Copy main networks weights into target network
        """


    def save(self):


    def load(self):