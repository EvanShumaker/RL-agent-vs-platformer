"""
Training loop for the DQN agent. Runs episodes against PlatformerEnv,
stores experience in the replay buffer, periodically trains the network
and updates the target network, logs progress to CSV, and saves checkpoints.
"""

import csv
import os
import numpy as np
from env.gym_wrapper import PlatformerEnv
from agent.dqn import DQNAgent
from agent.replay_buffer import ReplayBuffer

# training hyperparameters
NUM_EPISODES = 1200
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 50_000
TARGET_UPDATE_FREQUENCY = 10  # episodes between target network syncs
MIN_BUFFER_SIZE_TO_TRAIN = 1000 # don't start training until buffer has this many experiences
CHECKPOINT_FREQUENCY = 100  # episodes between saving model checkpoints

