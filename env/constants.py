"""
This file is for physics and game constants, like player dimensions
and screen size.
"""

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 450

GRAVITY = 0.8
JUMP_VELOCITY = -15
MOVE_SPEED = 5
TERMINAL VELOCITY = 20      # need max fall speed or physics get jankyy

PLAYER_WIDTH = 30
PLAYER_HEIGHT = 40
PLAYER_START_X = 50
PLAYER_START_Y = SCREEN_HEIGHT - 100    # near ground platform, higher pixel number is lower on screen i think


# Discrete actions for DQN
ACTION_NOOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_JUMP = 3
NUM_ACTIONS = 4


# --- Level Layout ---
# a simple level, with ground and two platforms over a pit
# the format is (x, y, width, height)
PLATFORMS = [
    (0, SCREEN_HEIGHT - 40, 300, 40),  # remember, down is positive.
    (400, SCREEN_HEIGHT - 40, 400, 40),     # This is the ground after the pit
    (300, SCREEN_HEIGHT - 150, 100, 20),
    (550, SCREE_HEIGH - 230, 100, 20),
]

# this is a hazard between 2 x values, basically saying the pit only exists there
PIT_X_RANGE = (300, 400)

# getting here is a win
GOAL = (720, SCREEN_HEIGHT - 270, 40, 40)

# this sets a limit on the number of actions an agent can use. prevents endless bs
MAX_EPISODE_STEPS = 500


# Reward values
REWARD_STEP = -0.01             # i wanna encourage efficiency
REWARD_GOAL = 100.0
REWARD_DEATH = -50.0
REWARD_PROGRESS_SCALE = 0.1     # multiplier for reward based on x progress towards goal