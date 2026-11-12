"""
Standalone script to plot the training learning curve from the CSV log.
this should be run after or during training process
"""

import csv
import matplotlib.pyplot as plt

LOG_PATH = "logs/training_log.csv"


def load_log(path):
    """ load info from csvs into variables """
    episodes, rewards, steps, epsilons, losses, wins = [], [], [], [], [], []

    with open(path, "r") as f:
        reader = csv.DictReader(f)
        has_wins = "reached_goal" in (reader.fieldnames or [])
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            steps.append(int(row["steps"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["avg_loss"]))
            if has_wins:
                wins.append(int(row["reached_goal"]))

    return episodes, rewards, steps, epsilons, losses, (wins if has_wins else None)

def moving_average(values, window=20):
    """Smooth data so it looks cleaner visually/clearer to see trends? idk if ill need"""
    if len(values) < window:
        return values
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(sum(values[start:i + 1]) / (i - start + 1))
    return smoothed


def plot(episodes, rewards, steps, epsilons, losses, wins):
    rows = 3 if wins is not None else 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4 * rows))

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

    if wins is not None:
        axes[2, 0].plot(episodes, moving_average(wins, window=50))
        axes[2, 0].set_title("Win Rate (50-episode rolling)")
        axes[2, 0].set_xlabel("Episode")
        axes[2, 0].set_ylabel("Fraction reaching goal")
        axes[2, 0].set_ylim(0, 1)
        axes[2, 1].axis("off")  # unused slot

    # helps keep them spread out
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    episodes, rewards, steps, epsilons, losses, wins = load_log(LOG_PATH)
    plot(episodes, rewards, steps, epsilons, losses, wins)