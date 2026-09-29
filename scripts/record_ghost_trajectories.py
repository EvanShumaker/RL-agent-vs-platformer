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
    """Find checkpoint files named like checkpoint_ep100.pt, sorted by episode number."""
    files = []
    for name in os.listdir(CHECKPOINT_DIR):
        match = re.match(r"checkpoint_ep(\d+)\.pt", name)
        if match:
            episode_num = int(match.group(1))
            files.append((episode_num, os.path.join(CHECKPOINT_DIR, name)))
    files.sort(key=lambda pair: pair[0])
    return files

def record_episode(env, agent):
    """Run one episode, recording (x, y) at every step."""
    observation = env.reset()
    state = env.game.get_state()
    trajectory = [(state["player_x"], state["player_y"])]
    total_reward = 0.0
    done = False

    while not done:
        action = agent.select_action(observation, training=True, epsilon_override=EXPLORATION_EPSILON)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        state = env.game.get_state()
        trajectory.append((state["player_x"], state["player_y"]))

    return trajectory, total_reward, state["goal_reached"]


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