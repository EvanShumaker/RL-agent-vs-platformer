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
        fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


    def select_action(self, observation, training=True):
        """
        This is epsilon greedy: With probability epsilon, pick random action (observation)
        Otherwise, pick action with highest predicted Q val (exploitation)
        """
        if training:
            epsilon = self.get_epsilon()
            self.steps_done += 1
            if (random.random() < epsilon):
                return random.randrange(self.num_actions)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0) #to add batch dimension
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()


    def train_step(self, replay_buffer, batch_size):
        """
        Sample a batch from buffer and do gradient descent update.
        Returns loss value for logging, and None if buffer doesnt have 
        enough experience
        """
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        rewards = torch.from_numpy(rewards)
        next_states = torch.from_numpy(next_states)
        dones = torch.from_numpy(dones)

        # current qvals for actions taken
        q_values = self.q_network(states)
        q_values_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        #target q vals: reward + discounted best future val (based on target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            # If done there is no future. zero out the future term
            targets = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values_taken, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def update_target_network(self):
        """
        Copy main networks weights into target network.
        doesnt return
        """
        self.target_network.load_state_dict(self.q_network.state_dict())


    def save(self, path):
        """Save state"""
        torch.save(self.q_network.state_dict(), path)

    def load(self, path):
        """Load state"""
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())