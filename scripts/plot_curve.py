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


def plot():


if __name__ == "__main__":