"""
quick check... how often do brand new, completely untrained networks "win"
under the same noisy evaluation? Tells us whether noisy win rate measures learning.
"""

from env.gym_wrapper import PlatformerEnv
from agent.dqn import DQNAgent
from scripts.evaluate_checkpoint import run_episode

env = PlatformerEnv()
for trial in range(5):
    agent = DQNAgent()   # fresh random weights, nothing loaded
    wins = sum(run_episode(env, agent, epsilon=0.05)[0] for _ in range(20))
    greedy_won = run_episode(env, agent, epsilon=None)[0]
    print(f"untrained network {trial}: noisy win rate {wins / 20:.2f}, greedy {'WIN' if greedy_won else 'lose'}")