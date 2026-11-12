"""
Evaluates every saved checkpoint: one fully greedy run (epsilon 0) plus
several runs with a little randomness (matching the end-of-training
exploration rate) to estimate win rate and robustness.
"""

import os
import re

from env.gym_wrapper import PlatformerEnv
from agent.dqn import DQNAgent

CHECKPOINT_DIR = "checkpoints"
NOISY_EPISODES = 20
NOISY_EPSILON = 0.05


def run_episode(env, agent, epsilon=None):
    """Run one episode. epsilon=None means fully greedy."""
    observation = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        if epsilon is None:
            action = agent.select_action(observation, training=False)
        else:
            action = agent.select_action(observation, training=True, epsilon_override=epsilon)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    reached_goal = env.game.get_state()["goal_reached"]
    return reached_goal, total_reward, steps


def get_checkpoints():
    """Return [(label, sort_key, path)] for every checkpoint, in training order."""
    found = []
    for name in os.listdir(CHECKPOINT_DIR):
        path = os.path.join(CHECKPOINT_DIR, name)
        match = re.match(r"checkpoint_ep(\d+)\.pt", name)
        if match:
            found.append((f"ep{match.group(1)}", int(match.group(1)), path))
        elif name == "checkpoint_final.pt":
            found.append(("final", float("inf"), path))
    found.sort(key=lambda item: item[1])
    return found


def main():
    env = PlatformerEnv()
    agent = DQNAgent()

    checkpoints = get_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {CHECKPOINT_DIR}/")
        return

    print(f"{'checkpoint':<12}{'greedy':<10}{'steps':<8}{'win rate (eps=0.05)':<22}{'avg reward':<10}")
    print("-" * 62)

    best = None  # (win_rate, greedy_won, label)

    for label, _, path in checkpoints:
        agent.load(path)

        greedy_won, _, greedy_steps = run_episode(env, agent, epsilon=None)

        wins = 0
        reward_sum = 0.0
        for _ in range(NOISY_EPISODES):
            won, reward, _ = run_episode(env, agent, epsilon=NOISY_EPSILON)
            wins += int(won)
            reward_sum += reward

        win_rate = wins / NOISY_EPISODES
        print(
            f"{label:<12}{'WIN' if greedy_won else 'lose':<10}{greedy_steps:<8}"
            f"{win_rate:<22.2f}{reward_sum / NOISY_EPISODES:<10.1f}"
        )

        candidate = (win_rate, greedy_won, label)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
            
    if best is None:
        return    

    print(f"\nBest checkpoint: {best[2]} (win rate {best[0]:.2f}, greedy {'WIN' if best[1] else 'lose'})")


if __name__ == "__main__":
    main()