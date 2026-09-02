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
    # these 3 print statements are for testing
    print("Starting train()...")
    env = PlatformerEnv()
    print("Environment created.")
    agent = DQNAgent()
    print("Agent created.")
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
        for episode in range(1, NUM_EPISODES+1):
            observation = env.reset()
            total_reward = 0.0
            steps = 0
            losses = []
            done = False

            while not done:
                action = agent.select_action(observation, training=True)
                next_observation, reward, done, info = env.step(action)

                replay_buffer.push(observation, action, reward, next_observation, done)

                observation = next_observation
                total_reward += reward
                steps += 1

                if len(replay_buffer) >= MIN_BUFFER_SIZE_TO_TRAIN:
                    loss = agent.train_step(replay_buffer, BATCH_SIZE)
                    if loss is not None:
                        losses.append(loss)

            if episode % TARGET_UPDATE_FREQUENCY == 0:
                agent.update_target_network()

            avg_loss = np.mean(losses) if losses else 0.0
            current_epsilon = agent.get_epsilon()

            log_writer.writerow([episode, total_reward, steps, current_epsilon, avg_loss])
            log_file.flush()  # write to disk immediately. logs are readable mid training

            print(
                f"Episode {episode}/{NUM_EPISODES} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"Epsilon: {current_epsilon:.3f} | "
                f"Avg Loss: {avg_loss:.4f}"
            )

            if episode % CHECKPOINT_FREQUENCY == 0:
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode}.pt")
                agent.save(checkpoint_path)
                print(f"  Saved checkpoint: {checkpoint_path}")

        # always save a final checkpoint at the end, regardless of frequency alignment
        final_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
        agent.save(final_path)
        print(f"Training complete. Final checkpoint saved: {final_path}")


    if __name__ == "__main__":
        train()