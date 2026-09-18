"""
Global Constants and Physical Parameters for Pong Studio
---------------------------------------------------------
Unifies court dimensions, velocities, and action spaces
to guarantee exact consistency between the training simulator (train.py),
production inference (model.py), and the graphical interface (PongCanvas.jsx).
"""

# ==============================================================================
# Paddle Actions
# ==============================================================================
ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
NUM_ACTIONS = 3

ACTION_NAMES = {
    ACTION_STAY: "STAY",
    ACTION_UP: "UP",
    ACTION_DOWN: "DOWN"
}

# ==============================================================================
# Physical Field Dimensions (Matching PongCanvas.jsx)
# ==============================================================================
FIELD_WIDTH = 800.0
FIELD_HEIGHT = 500.0

PADDLE_WIDTH = 12.0
PADDLE_HEIGHT = 86.0
PADDLE_OFFSET_X = 30.0  # Distance between the rear wall and the paddle center

BALL_SIZE = 10.0
AI_PADDLE_SPEED = 5.5
INITIAL_BALL_SPEED = 4.5
MAX_BALL_SPEED = 15.0

# Maximum normalization boundaries for DQN continuous state
MAX_DX = FIELD_WIDTH - PADDLE_OFFSET_X
MAX_DY = FIELD_HEIGHT - (PADDLE_HEIGHT / 2.0) - (BALL_SIZE / 2.0)
