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


def moving_average():
    """Smooth data so it looks cleaner visually/clearer to see trends? idk if ill need"""


def plot():


if __name__ == "__main__":