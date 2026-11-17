"""
Training loop for the DQN agent. Runs episodes against PlatformerEnv,
stores experience in the replay buffer, periodically trains the network
and updates the target network, logs progress to CSV, and saves checkpoints.
Evaluates so best checkpoint is always saved.
"""

import csv
import os
import numpy as np
from env.gym_wrapper import PlatformerEnv
from agent.dqn import DQNAgent
from agent.replay_buffer import ReplayBuffer

# another test print
# print("MODULE LOADED", flush=True)

# training hyperparameters
# NUM_EPISODES = 1200
NUM_EPISODES = 1200
BATCH_SIZE = 64
REPLAY_BUFFER_CAPACITY = 200_000 #50_000
#TARGET_UPDATE_FREQUENCY = 10  # episodes between target network syncs
TARGET_UPDATE_STEPS = 2000         # env steps between target network syncs
MIN_BUFFER_SIZE_TO_TRAIN = 1000 # don't start training until buffer has this many experiences
CHECKPOINT_FREQUENCY = 100  # episodes between saving model checkpoints

# Periodic evaluation (decides when to save checkpoint_best.pt) 
EVAL_FREQUENCY = 50  # episodes between evaluations
EVAL_EPISODES = 20
EVAL_EPSILON = 0.05

LOG_PATH = "logs/training_log.csv"
CHECKPOINT_DIR = "checkpoints"

def evaluate(agent, eval_env):
    """
    Returns (noisy_win_rate, greedy_won).
    This function measures how good the agent currently is, so we know when to save
    a chackpoint, 
    The game is deterministic, so a fully greedy run is identical every time;
    the noisy runs (small epsilon) measure how robust the policy is.
    eval_env is our test env, we want separate so it doesnt share state with a
    training env.
    """
    def run_episode(epsilon):
        """
        epsilon is how much randomness to allow. 'None' is no randomness
        """
        # fresh episode
        observation = eval_env.reset()
        done = False
        while not done: #keep going till episode ends or hits 500 step limit
            if epsilon is None: #greedy mode: always pick highest q val option
                action = agent.select_action(observation, training=False)
            else: #noisy mode with randomness
                action = agent.select_action(
                    observation, training=True, epsilon_override=epsilon
                )
            #step() returns obervation, reward, done, info. Use _ for the stuff i dont need
            observation, _, done, _ = eval_env.step(action)

        #returns true if episode reached the goal, false otherwise
        return eval_env.game.get_state()["goal_reached"]

    greedy_won = run_episode(None)
    # run 20 noisy eps and count the wins
    wins = sum(run_episode(EVAL_EPSILON) for _ in range(EVAL_EPISODES))
    return wins / EVAL_EPISODES, greedy_won # return win% and whether greedy won

def train():
    env = PlatformerEnv()

    # separate env (only for the periodic evaluations) so evaluation never disturbs training
    eval_env = PlatformerEnv()   

    agent = DQNAgent()
    replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_CAPACITY)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    total_steps = 0 #counts every environment step across all episodes
    # score to beat, starts -1 so any real score can beat it
    best_score = (-1.0, False)   
    # NOTE: Python compares tuples left to right, so the true/false is a tiebreaker

    with open(LOG_PATH, "w", newline="") as log_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(
            ["episode", "total_reward", "steps", "epsilon", "avg_loss", "reached_goal"]
        )

        # each of these lps is one attempt at the level
        for episode in range(1, NUM_EPISODES + 1):
            observation = env.reset()
            total_reward = 0.0
            steps = 0
            losses = []
            done = False

            while not done: #each pass here is one step
                action = agent.select_action(observation, training=True)
                next_observation, reward, done, info = env.step(action)
                # 'done' means they died, won, or timed out.

                # Timeouts aren't real endings (truncated implies timeout), so true terminal = real ending
                terminal = done and not info["truncated"]
                replay_buffer.push(observation, action, reward, next_observation, terminal)

                observation = next_observation
                total_reward += reward
                steps += 1
                total_steps += 1

                # a buffer has 1000 experiences, every env step also triggers a training step
                #w/ a random batch of 64 old experiences
                if len(replay_buffer) >= MIN_BUFFER_SIZE_TO_TRAIN:
                    loss = agent.train_step(replay_buffer, BATCH_SIZE)
                    if loss is not None:
                        losses.append(loss)

                # every 2k steps overall, copy the main networks weights into the target network
                if total_steps % TARGET_UPDATE_STEPS == 0:
                    agent.update_target_network()

            # checks if episode ended in a win or loss (1 or 0)
            reached_goal = int(env.game.get_state()["goal_reached"])
            avg_loss = np.mean(losses) if losses else 0.0
            current_epsilon = agent.get_epsilon()

            log_writer.writerow(
                [episode, total_reward, steps, current_epsilon, avg_loss, reached_goal]
            )
            log_file.flush()

            print(
                f"Episode {episode}/{NUM_EPISODES} | "
                f"Reward: {total_reward:.2f} | "
                f"Steps: {steps} | "
                f"Epsilon: {current_epsilon:.3f} | "
                f"Avg Loss: {avg_loss:.4f}"
            )

            # saving the weights every 100eps in a checkpoint
            if episode % CHECKPOINT_FREQUENCY == 0:
                path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode}.pt")
                agent.save(path)
                print(f"  Saved checkpoint: {path}")

            # evaluate every 50 episodes
            if episode % EVAL_FREQUENCY == 0:
                win_rate, greedy_won = evaluate(agent, eval_env)
                score = (greedy_won, win_rate) #score = (win_rate, greedy_won)
                print(
                    f"  EVAL ep{episode}: win rate {win_rate:.2f}, "
                    f"greedy {'WIN' if greedy_won else 'lose'}"
                )
                # if the eval is the best so far, overwrite the best checkpoint we have
                if score > best_score:
                    best_score = score
                    best_path = os.path.join(CHECKPOINT_DIR, "checkpoint_best.pt")
                    agent.save(best_path)
                    print(f"  New best! Saved {best_path}")

    final_path = os.path.join(CHECKPOINT_DIR, "checkpoint_final.pt")
    agent.save(final_path)
    print(f"Training complete. Final checkpoint saved: {final_path}")
    #print(f"Best evaluation: win rate {best_score[0]:.2f}, greedy {'WIN' if best_score[1] else 'lose'}")
    print(f"Best evaluation: greedy {'WIN' if best_score[0] else 'lose'}, win rate {best_score[1]:.2f}")


if __name__ == "__main__":
    train()