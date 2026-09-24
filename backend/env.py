"""
Simulation environment and discretizer for Pong Studio
"""

import math
import random
from typing import Tuple, Dict, Any, Optional
import numpy as np

from constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    PADDLE_WIDTH,
    PADDLE_HEIGHT,
    PADDLE_OFFSET_X,
    BALL_SIZE,
    AI_PADDLE_SPEED,
    INITIAL_BALL_SPEED,
    MAX_BALL_SPEED,
    MAX_DX,
    MAX_DY
)
from rust_bridge import is_rust_available, rust_step, rust_discretize


class PongStateDiscretizer:
    """
    Convert game's continuous variables into a discrete integer index `state_id`.
    """
    def __init__(
        self,
        n_bins_dx: int = 20,
        n_bins_dy: int = 24,
        n_bins_vx: int = 2,
        n_bins_vy: int = 2,
        n_bins_speed: int = 3
    ):
        self.n_bins_dx = n_bins_dx
        self.n_bins_dy = n_bins_dy
        self.n_bins_vx = n_bins_vx
        self.n_bins_vy = n_bins_vy
        self.n_bins_speed = n_bins_speed

        self.max_dx = MAX_DX
        self.max_dy = MAX_DY
        self.total_states = n_bins_dx * n_bins_dy * n_bins_vx * n_bins_vy * n_bins_speed

    def discretize(
        self,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
        paddle_x: float,
        paddle_y: float,
        field_width: float = FIELD_WIDTH,
        field_height: float = FIELD_HEIGHT
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Compute the bin for each component and flatten them into a single linear state_id.
        """
        dx = abs(paddle_x - ball_x)
        dy = ball_y - paddle_y

        dx_clamped = max(0.0, min(self.max_dx, dx))
        bin_dx = min(self.n_bins_dx - 1, int((dx_clamped / self.max_dx) * self.n_bins_dx))

        dy_clamped = max(-self.max_dy, min(self.max_dy, dy))
        dy_norm = (dy_clamped + self.max_dy) / (2.0 * self.max_dy)
        bin_dy = min(self.n_bins_dy - 1, int(dy_norm * self.n_bins_dy))

        is_approaching = (ball_vx > 0 and paddle_x > ball_x) or (ball_vx < 0 and paddle_x < ball_x)
        bin_vx = 0 if is_approaching else 1
        bin_vy = 0 if ball_vy >= 0 else 1

        # Speed tiers: 0: Low (<= 7.0 px/f), 1: Medium (7.0 - 11.0 px/f), 2: High (> 11.0 px/f)
        speed = math.hypot(ball_vx, ball_vy)
        if speed <= 7.0:
            bin_speed = 0
        elif speed <= 11.0:
            bin_speed = 1
        else:
            bin_speed = 2

        r_state_id = rust_discretize(ball_x, ball_y, ball_vx, ball_vy, paddle_x, paddle_y)
        if r_state_id is not None:
            state_id = r_state_id
        else:
            state_id = (
                bin_dx * (self.n_bins_dy * self.n_bins_vx * self.n_bins_vy * self.n_bins_speed) +
                bin_dy * (self.n_bins_vx * self.n_bins_vy * self.n_bins_speed) +
                bin_vx * (self.n_bins_vy * self.n_bins_speed) +
                bin_vy * self.n_bins_speed +
                bin_speed
            )

        info = {
            "state_id": state_id,
            "bin_dx": bin_dx,
            "bin_dy": bin_dy,
            "bin_vx": bin_vx,
            "bin_vy": bin_vy,
            "bin_speed": bin_speed,
            "speed_tier": "low" if bin_speed == 0 else ("medium" if bin_speed == 1 else "high"),
            "speed_raw": float(speed),
            "dx_raw": float(dx),
            "dy_raw": float(dy)
        }
        return state_id, info

    def decode_state(self, state_id: int) -> Dict[str, int]:
        """
        Decode state_id int to his bins components.
        """
        speed = state_id % self.n_bins_speed
        rem = state_id // self.n_bins_speed
        vy = rem % self.n_bins_vy
        rem = rem // self.n_bins_vy
        vx = rem % self.n_bins_vx
        rem = rem // self.n_bins_vx
        dy = rem % self.n_bins_dy
        dx = rem // self.n_bins_dy
        return {"bin_dx": dx, "bin_dy": dy, "bin_vx": vx, "bin_vy": vy, "bin_speed": speed}


class FastPongEnv:
    """
    Simulator healess for Pong with fast physics engine
    Supports two modes:
      1. 'fronton': The left wall bounces the ball (accelerated defensive training).
      2. 'heuristic': An opponent paddle on the left with progressive speed
         for the AI to learn to score goals and win matches.
    """
    def __init__(
        self,
        discretizer: Optional[PongStateDiscretizer] = None,
        opponent: str = "fronton",
        opp_speed_ratio: float = 0.50
    ):
        self.discretizer = discretizer if discretizer is not None else PongStateDiscretizer()
        self.opponent = opponent  # 'fronton' or 'heuristic'
        self.opp_speed_ratio = opp_speed_ratio
        self.ai_x = FIELD_WIDTH - PADDLE_OFFSET_X
        self.opponent_x = PADDLE_OFFSET_X
        self.opponent_y = FIELD_HEIGHT / 2.0
        self.ai_vy = 0.0
        self.opp_vy = 0.0
        self.rally_hits = 0

        # Pre-allocated observation buffer for continuous states
        self._state_buffer = np.zeros(4, dtype=np.float32)
        self.reset()

    def set_opponent_speed_ratio(self, ratio: float):
        """
        Adjust opponent speed for Curriculum Learning (0.30 to 1.0).
        """
        self.opp_speed_ratio = float(np.clip(ratio, 0.30, 1.0))

    def get_continuous_state(self, dim: int = 4) -> np.ndarray:
        """
        Return the continuous state vector for DQN:
          - dim=4 (classic): [dx_norm, dy_norm, vx_norm, vy_norm]
          - dim=5 (no-blind): [dx_norm, dy_norm, vx_norm, vy_norm, opp_y_norm]
        """
        dx = abs(self.ai_x - self.ball_x)
        dy = self.ball_y - self.ai_y

        if dim == 5:
            opp_y_norm = float(np.clip((self.opponent_y - FIELD_HEIGHT / 2.0) / (FIELD_HEIGHT / 2.0), -1.0, 1.0))
            return np.array([
                float(np.clip(dx / MAX_DX, 0.0, 1.0)),
                float(np.clip(dy / MAX_DY, -1.0, 1.0)),
                float(np.clip(self.ball_vx / MAX_BALL_SPEED, -1.0, 1.0)),
                float(np.clip(self.ball_vy / MAX_BALL_SPEED, -1.0, 1.0)),
                opp_y_norm
            ], dtype=np.float32)

        self._state_buffer[0] = np.clip(dx / MAX_DX, 0.0, 1.0)
        self._state_buffer[1] = np.clip(dy / MAX_DY, -1.0, 1.0)
        self._state_buffer[2] = np.clip(self.ball_vx / MAX_BALL_SPEED, -1.0, 1.0)
        self._state_buffer[3] = np.clip(self.ball_vy / MAX_BALL_SPEED, -1.0, 1.0)
        return self._state_buffer.copy()

    def get_continuous_state_5d(self) -> np.ndarray:
        """Helper for 5-dimensional continuous state vector."""
        return self.get_continuous_state(dim=5)

    def reset(self) -> Tuple[int, Dict[str, Any]]:
        """
        Reset ball and paddle positions with aggressive domain randomization:
        - Wider launch angles (up to ~43 degrees) targeting court corners.
        - Dynamic serve speed scaling (between 4.3 and 6.1 px/step).
        - Cross-court stress-testing: Spawns the AI paddle at the opposite extreme
          in ~35% of serves to train rapid intercept reaction from disadvantageous positions.
        """
        min_paddle_y = PADDLE_HEIGHT / 2.0
        max_paddle_y = FIELD_HEIGHT - PADDLE_HEIGHT / 2.0

        self.rally_hits = 0
        self.ball_x = FIELD_WIDTH / 2.0
        self.ball_y = np.random.uniform(BALL_SIZE * 3, FIELD_HEIGHT - BALL_SIZE * 3)

        # Launch angles spanning up to +/- 43 degrees for aggressive corner trajectories
        angle = np.random.uniform(-0.75, 0.75)
        serve_speed = INITIAL_BALL_SPEED * np.random.uniform(0.95, 1.35)
        dir_x = 1.0 if random.random() < 0.70 else -1.0
        self.ball_vx = dir_x * serve_speed * math.cos(angle)
        self.ball_vy = serve_speed * math.sin(angle)

        # Cross-court stress test (35% of serves): paddle starts opposite to incoming ball trajectory
        if dir_x > 0 and random.random() < 0.35:
            if self.ball_vy < 0:
                # Ball ascending -> place paddle at lower boundary
                self.ai_y = float(np.random.uniform(max_paddle_y - 45.0, max_paddle_y))
            else:
                # Ball descending -> place paddle at upper boundary
                self.ai_y = float(np.random.uniform(min_paddle_y, min_paddle_y + 45.0))
        else:
            self.ai_y = float(np.random.uniform(min_paddle_y, max_paddle_y))

        if dir_x < 0:
            # When serving towards opponent, align opponent with ball so it reliably initiates a rally
            self.opponent_y = float(np.clip(self.ball_y + np.random.uniform(-15.0, 15.0), min_paddle_y, max_paddle_y))
        else:
            self.opponent_y = float(np.random.uniform(min_paddle_y, max_paddle_y))

        state_id, info = self.discretizer.discretize(
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_vx=self.ball_vx,
            ball_vy=self.ball_vy,
            paddle_x=self.ai_x,
            paddle_y=self.ai_y
        )
        info["continuous_state"] = self.get_continuous_state(dim=4)
        info["continuous_state_5d"] = self.get_continuous_state(dim=5)
        return state_id, info

    def reset_to_shot(self, shot: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """
        Reset environment to the exact physical snapshot of a recorded failure shot.
        Supports symmetric projection so that shots recorded on P1 (left) or P2 (right)
        are properly oriented towards the AI paddle (defending right wall).
        """
        min_paddle_y = PADDLE_HEIGHT / 2.0
        max_paddle_y = FIELD_HEIGHT - PADDLE_HEIGHT / 2.0

        origin = shot.get("shot_origin", {})
        miss = shot.get("miss_impact", {})
        side = shot.get("agent_side", "p1")

        orig_bx = origin.get("ball_x", FIELD_WIDTH / 2.0)
        orig_by = origin.get("ball_y", FIELD_HEIGHT / 2.0)
        orig_vx = origin.get("ball_vx", 5.0)
        orig_vy = origin.get("ball_vy", 0.0)

        # Authentic pre-shot paddle coordinate: use defender_paddle_y from shot_origin if available;
        # otherwise gracefully fall back to miss_impact paddle coordinate for legacy datasets.
        initial_ai_y = origin.get("defender_paddle_y")
        if initial_ai_y is None:
            initial_ai_y = miss.get("paddle_y", FIELD_HEIGHT / 2.0)

        if side == "p1" or orig_vx < 0:
            # Shot was incoming to left paddle. Mirror horizontally to right paddle (AI side)
            self.ball_x = float(np.clip(FIELD_WIDTH - orig_bx, 30.0, FIELD_WIDTH - 40.0))
            self.ball_y = float(np.clip(orig_by, BALL_SIZE, FIELD_HEIGHT - BALL_SIZE))
            self.ball_vx = float(abs(orig_vx))  # moving right towards AI paddle
            self.ball_vy = float(orig_vy)
            self.opponent_y = float(np.clip(origin.get("paddle_y", FIELD_HEIGHT / 2.0), min_paddle_y, max_paddle_y))
            self.ai_y = float(np.clip(initial_ai_y, min_paddle_y, max_paddle_y))
        else:
            # Shot was incoming to right paddle.
            self.ball_x = float(np.clip(orig_bx, 30.0, FIELD_WIDTH - 40.0))
            self.ball_y = float(np.clip(orig_by, BALL_SIZE, FIELD_HEIGHT - BALL_SIZE))
            self.ball_vx = float(abs(orig_vx))  # moving right towards AI paddle
            self.ball_vy = float(orig_vy)
            self.opponent_y = float(np.clip(origin.get("paddle_y", FIELD_HEIGHT / 2.0), min_paddle_y, max_paddle_y))
            self.ai_y = float(np.clip(initial_ai_y, min_paddle_y, max_paddle_y))

        self.ai_vy = 0.0
        self.opp_vy = 0.0
        self.rally_hits = 0

        state_id, info = self.discretizer.discretize(
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_vx=self.ball_vx,
            ball_vy=self.ball_vy,
            paddle_x=self.ai_x,
            paddle_y=self.ai_y
        )
        info["continuous_state"] = self.get_continuous_state()
        return state_id, info

    def _move_ai_paddle(self, ai_action: int):
        """Move AI paddle within field boundaries."""
        if ai_action == ACTION_UP:
            self.ai_y = max(PADDLE_HEIGHT / 2.0, self.ai_y - AI_PADDLE_SPEED)
        elif ai_action == ACTION_DOWN:
            self.ai_y = min(FIELD_HEIGHT - PADDLE_HEIGHT / 2.0, self.ai_y + AI_PADDLE_SPEED)

    def _move_opponent_paddle(self):
        """Move heuristic opponent paddle tracking the ball with speed ratio."""
        if self.opponent == "heuristic":
            dy_opp = self.ball_y - self.opponent_y
            opp_speed = AI_PADDLE_SPEED * self.opp_speed_ratio
            if dy_opp < -8:
                self.opponent_y = max(PADDLE_HEIGHT / 2.0, self.opponent_y - opp_speed)
            elif dy_opp > 8:
                self.opponent_y = min(FIELD_HEIGHT - PADDLE_HEIGHT / 2.0, self.opponent_y + opp_speed)
        elif self.opponent == "fronton":
            # In fronton mode, the left paddle acts as an impenetrable wall that always returns the ball
            self.opponent_y = self.ball_y



    def _compute_reward(
        self,
        hit_ai_paddle: bool,
        edge_bonus: float,
        scored_goal: bool,
        missed_ball: bool,
        ai_action: int
    ) -> float:
        """Centralized reward calculation with shaping and anti-jitter bonuses."""
        reward = 0.0

        if hit_ai_paddle:
            reward += 2.0 + edge_bonus

        if scored_goal:
            # Massive offensive reward: scoring against the opponent is the primary objective
            reward += 18.0

        if missed_ball:
            reward -= 5.0

        # Anticipatory active alignment reward when ball approaches
        if self.ball_vx > 0:
            dy = self.ball_y - self.ai_y
            dist_y = abs(dy)
            # General alignment gradient toward trajectory
            reward += 0.12 * (1.0 - min(1.0, dist_y / (FIELD_HEIGHT / 2.0)))

            # Zone A: Sweet spot (inner 22 px) - staying still is safe and rewarded
            if dist_y <= 22.0:
                reward += 0.08
                if ai_action == ACTION_STAY:
                    reward += 0.08

            # Zone B: Edge hazard zone (22 px to 48 px) - ball threatens to slip past!
            # Actively moving towards the ball is heavily rewarded; staying still is discouraged.
            elif dist_y <= (PADDLE_HEIGHT / 2.0 + BALL_SIZE / 2.0):
                is_correcting = (dy > 0 and ai_action == ACTION_DOWN) or (dy < 0 and ai_action == ACTION_UP)
                if is_correcting:
                    reward += 0.14
                elif ai_action == ACTION_STAY:
                    reward -= 0.06

            # Zone C: Outer pursuit zone (> 48 px) - reward closing distance
            else:
                is_moving_towards = (dy > 0 and ai_action == ACTION_DOWN) or (dy < 0 and ai_action == ACTION_UP)
                if is_moving_towards:
                    reward += 0.10

        return reward

    def step(self, ai_action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Advance one simulation step.
        """
        # 1. Update paddle positions
        prev_ai_y = self.ai_y
        prev_opp_y = self.opponent_y

        self._move_ai_paddle(ai_action)
        self._move_opponent_paddle()

        self.ai_vy = self.ai_y - prev_ai_y
        self.opp_vy = self.opponent_y - prev_opp_y

        # 2 & 3. Advance ball position and resolve collisions via Rust physics engine
        state, outcome = rust_step(
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_vx=self.ball_vx,
            ball_vy=self.ball_vy,
            p1_y=self.opponent_y,
            p2_y=self.ai_y,
            p1_vy=self.opp_vy,
            p2_vy=self.ai_vy,
            sub_steps=1
        )
        self.ball_x = state.ball_x
        self.ball_y = state.ball_y
        self.ball_vx = state.ball_vx
        self.ball_vy = state.ball_vy
        # In Rust: goal_p2 = true when ball_x < 0 (P2/AI scores goal on left)
        scored_goal = outcome.goal_p2
        hit_ai_paddle = outcome.hit_p2
        edge_bonus = 0.0
        if hit_ai_paddle:
            self.rally_hits += 1
            abs_offset = abs(outcome.hit_offset_p2)
            
            # Recompensa basada en la posición de impacto en la paleta:
            # 1. Centro (0.0 a 0.2): Máxima recompensa por tiro seguro
            # 2. Casi esquinas (0.2 a 0.85): Menor recompensa
            # 3. Extremos (> 0.85): Recompensa media para incitar tiros con rebote
            if abs_offset < 0.2:
                edge_bonus += 2.0
            elif abs_offset > 0.85:
                edge_bonus += 1.0
            else:
                edge_bonus += 0.0

            if abs(self.ball_vy) > 3.0:
                edge_bonus += (abs(self.ball_vy) / MAX_BALL_SPEED) * 1.5

            # Tactical cross-court placement bonus: reward directing ball into open court away from opponent
            opp_norm = (self.opponent_y - FIELD_HEIGHT / 2.0) / (FIELD_HEIGHT / 2.0)
            ball_dir = self.ball_vy / MAX_BALL_SPEED
            if (opp_norm > 0.2 and ball_dir < -0.15) or (opp_norm < -0.2 and ball_dir > 0.15):
                edge_bonus += 2.0

        # In Rust: goal_p1 = true when ball_x > FIELD_WIDTH (P1 scores on right, so AI missed ball)
        missed_ball = outcome.goal_p1
        done = scored_goal or missed_ball

        # 5. Compute shaped reward
        reward = self._compute_reward(hit_ai_paddle, edge_bonus, scored_goal, missed_ball, ai_action)

        # 6. Discretize next state
        next_state_id, info = self.discretizer.discretize(
            ball_x=self.ball_x,
            ball_y=self.ball_y,
            ball_vx=self.ball_vx,
            ball_vy=self.ball_vy,
            paddle_x=self.ai_x,
            paddle_y=self.ai_y
        )
        info["continuous_state"] = self.get_continuous_state(dim=4)
        info["continuous_state_5d"] = self.get_continuous_state(dim=5)
        info["hit_ai_paddle"] = hit_ai_paddle
        info["scored_goal"] = scored_goal
        info["missed_ball"] = missed_ball
        return next_state_id, reward, done, info
