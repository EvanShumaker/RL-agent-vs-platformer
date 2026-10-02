"""
Plays back recorded ghost trajectories all at once. many
"ghost" characters moving simultaneously along their recorded paths,
colored by training stage (blue is early, red is later). The top-N
highest-reward runs should stand out
"""

import json
import pygame
import env.constants as C

TRAJECTORY_PATH = "logs/ghost_trajectories.json"
TOP_N_HIGHLIGHT = 10
FPS = 60


def load_runs():
    with open(TRAJECTORY_PATH, "r") as f:
        return json.load(f)

def get_color_for_checkpoint(checkpoint_episode, min_ep, max_ep):
    """Based on how high the reward was, the color will be closer to
    blue than red. This will probably get called for every run in main()"""

    # min and max ep used to base the colors off of. Fraction math to get what the gradient should be
    if max_ep == min_ep:
            fraction = 1.0
    else:
        fraction = (checkpoint_episode - min_ep) / (max_ep - min_ep)
    r = int(255 * fraction)
    b = int(255 * (1 - fraction))
    return (r, 80, b)

def draw_static_level(screen):
    """self explanatory, recreating the level based on what was used in training"""


def main():
    """Big func: get all the runs as json, sort by reward, init pygame, then
     add all the run records with loop. """

if __name__ == "__main__":
    main()