"""
Runs the trained agent at each saved checkpoint (training stage) for several
episodes, recording the full player trajectory (x, y at every step) so it
can be visualized later as overlapping replays.
"""

import os
import re
import json

from env.gym_wrapper import PlatformerEnv
from agent.dqn import DQNAgent

# TODO: Maybe come back and change
CHECKPOINT_DIR = "checkpoints"
OUTPUT_PATH = "logs/ghost_trajectories.json"
EPISODES_PER_CHECKPOINT = 8
EXPLORATION_EPSILON = 0.15  # small randomness so repeated runs on the same checkpoint vary.



def get_checkpoint_files():
    """Find checkpoint files based on name and then sort them bu number"""
    files = []
    return files

def record_episode(env, agent):
    """Run a single episode, recording the x,y at every step"""
    trajectory = 0
    state = env.game.get_state()
    return trajectory, [], state["x"]


def main():
    env = PlatformerEnv()
    agent = DQNAgent()

    checkpoint_files = get_checkpoint_files()
    if not checkpoint_files:
        print(f"No checkpoints found in {CHECKPOINT_DIR}/")
        return

    all_runs = []

    for episode_num, path in checkpoint_files:
        agent.load(path)
        for i in range(EPISODES_PER_CHECKPOINT):
            trajectory, total_reward, reached_goal = record_episode(env, agent)
            all_runs.append({
                "checkpoint_episode": episode_num,
                "total_reward": total_reward,
                "reached_goal": reached_goal,
                "trajectory": trajectory,
            })
            print(f"Checkpoint ep{episode_num}, run {i}: reward={total_reward:.2f}, steps={len(trajectory)}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_runs, f)

    print(f"\nSaved {len(all_runs)} runs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()