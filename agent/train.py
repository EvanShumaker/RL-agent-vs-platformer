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

LOG_PATH = "logs/training_log.csv"
CHECKPOINT_DIR = "checkpoints"


def train():
    env = PlatformerEnv()
    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)

    # prog is ran multiple times, so obv the file is going to exist after the first run, and thats fine
    # .dirname gets just logs/ from the whole logpath
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # I could do log_file = open(...), but id have to do .close() later. And if theres
    #an exception thrown somewhere and .close() isnt ran, the file stays open
    with open(LOG_PATH, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["episode","total_reward","steps","episode","avg_loss"])

        # cant do 0, NUM_EPS
        for episode in range(1, NUM_EPISODES+1)