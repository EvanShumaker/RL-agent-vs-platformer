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
        self.net = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,num_acitons),
        ) # add parameters to this


    def forward(self, x):
        """
        Passes the observation x thru the network.
        """
        return self.net(x)


class DQNAgent:
    def __init__(
            self,
            obs_size=PlatformerEnv.OBS_SIZE,
            num_actions=C.NUM_ACTIONS,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=50_000,
    ):
        """Setup target network here."""
        self.num_actions = num_actions
        self.gamma = gamma # how much future reward matters compared to immediate

        #main network. Train this and use it to pick actions
        self.q_network = QNetwork(obs_size, num_actions)

        #Target network. An occasionally updated copy used to compute good training targets
        #Starts as exact copy of qnetwork
        self.target_network = QNetwork(obs_size, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval() # never trained directly. only copied into

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # epsilon greedy exploration schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0


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
        """Save state"""

    def load(self):
        """Load state"""