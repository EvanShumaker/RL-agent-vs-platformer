"""
Standalone script to plot the training learning curve from the CSV log.
this should be run after or during training process
"""

import csv
import matplotlib.pyplot as plt

LOG_PATH = "logs/training_log.csv"


def load_log(path):
    """ load info from csvs into variables """
    episodes, rewards, steps, epsilons, losses = [], [], [], [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            steps.append(int(row["steps"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["avg_loss"]))

    return episodes, rewards, steps, epsilons, losses


def moving_average(values, window=20):
    """Smooth data so it looks cleaner visually/clearer to see trends? idk if ill need"""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
    return smoothed


def plot(episodes, rewards, steps, epsilons, losses):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reward over time: the main signal you care about
    axes[0, 0].plot(episodes, rewards, alpha=0.3, label="raw")
    axes[0, 0].plot(episodes, moving_average(rewards), label="smoothed")
    axes[0, 0].set_title("Total Reward per Episode")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    # Episode length: should generally trend toward "reaches goal faster"
    axes[0, 1].plot(episodes, steps, alpha=0.3, label="raw")
    axes[0, 1].plot(episodes, moving_average(steps), label="smoothed")
    axes[0, 1].set_title("Steps per Episode")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Steps")
    axes[0, 1].legend()

    # Epsilon decay: sanity check it's decaying as expected
    axes[1, 0].plot(episodes, epsilons)
    axes[1, 0].set_title("Epsilon (Exploration Rate)")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Epsilon")

    # Loss: should generally stabilize/decrease, though DQN loss is noisy by nature
    axes[1, 1].plot(episodes, losses, alpha=0.3, label="raw")
    axes[1, 1].plot(episodes, moving_average(losses), label="smoothed")
    axes[1, 1].set_title("Average Loss per Episode")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()

    # helps keep them spread out
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    episodes, rewards, steps, epsilons, losses = load_log(LOG_PATH)
    plot(episodes, rewards, steps, epsilons, losses)