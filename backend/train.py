"""
================================================================================
Pong RL Studio - Main Training Module (Tabular & DQN)
================================================================================
Orchestrates training of Reinforcement Learning agents for Pong:
  1. TabularTrainer: Discrete tabular Q-Learning using Bellman TD equations.
  2. DQNTrainer: Deep Q-Network with PyTorch, N-Step Replay Buffer, Double Q-Learning,
     Curriculum Learning, and automated ONNX export.

Typical usage:
  python3 pong/backend/train.py --algo dqn --opponent fronton
  python3 pong/backend/train.py --algo q_learning --episodes 50000
================================================================================
"""

import os
import sys
import copy
import json
import random
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from constants import NUM_ACTIONS
from env import FastPongEnv, PongStateDiscretizer
from buffer import NStepReplayBuffer


import torch
import torch.nn as nn
import torch.optim as optim



# ==============================================================================
# Definition of PongDQN
# ==============================================================================
class PongDQN(nn.Module):
    """
    Perceptron Multilayer (MLP) for Deep Q-Learning in Pong.
    Input: 4 normalized continuous dimensions [dx, dy, vx, vy] in [-1, 1].
    Output: 3 Q-values for actions [0: STAY, 1: UP, 2: DOWN].
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



# ==============================================================================
# Training Configuration
# ==============================================================================
@dataclass
class TabularConfig:
    total_episodes: int = 50000
    alpha: float = 0.15         # Learning rate
    gamma: float = 0.98         # Bellman TD discount factor
    epsilon_start: float = 1.0  # Initial exploration
    epsilon_min: float = 0.02   # Minimum exploration
    epsilon_decay: float = 0.9998
    jitter_penalty: float = 0.05 # Anti-jitter penalty
    max_steps: int = 3500       # Max steps per episode
    opponent: str = "fronton"   # 'fronton' or 'heuristic'
    pretrained_path: Optional[str] = None
    save_path: str = "models/q_table.npy"
    boost_errors: bool = True   # Boost learning on failure episodes via TD error scaling & backward rollback
    best_score_baseline: float = 0.0
    skip_baseline_eval: bool = False


@dataclass
class DQNConfig:
    total_episodes: int = 1200
    batch_size: int = 256
    gamma: float = 0.99         # Temporal horizon extended (100 effective steps)
    lr: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9985
    n_step: int = 5             # N-Step returns to reduce temporal myopia
    tau: float = 0.005          # Soft target update
    jitter_penalty: float = 0.05
    warmup_steps: int = 1000
    train_freq: int = 4         # Train every 4 simulation steps
    max_steps: int = 3500       # Max steps per episode
    buffer_capacity: int = 50000
    opponent: str = "fronton"   # 'fronton' or 'heuristic'
    pretrained_path: Optional[str] = None
    save_path: str = "models/dqn_pong.pth"
    best_score_baseline: float = 0.0
    skip_baseline_eval: bool = False


# ==============================================================================
# Q-Learning Trainer Class
# ==============================================================================
class TabularTrainer:
    """
    Manages the training cycle for discrete Q-learning on Q(s, a) matrix
    - Uses FastPongEnv physical engine with anticipatory reward gradient
    - Supports Transfer Learning / Warm-Restart reusing previous matrices
    - Safe checkpointing that immediately saves the best Avg Hits record
    - Ctrl+C capture to always preserve the best matrix obtained
    """
    def __init__(self, config: TabularConfig):
        self.config = config
        self.discretizer = PongStateDiscretizer()
        self.env = FastPongEnv(discretizer=self.discretizer, opponent=self.config.opponent)
        self.Q = np.zeros((self.discretizer.total_states, NUM_ACTIONS), dtype=np.float32)
        self.best_Q = self.Q.copy()
        self.best_avg_hits = 0.0
        self.best_score = self.config.best_score_baseline

        if self.config.pretrained_path:
            self._load_pretrained(self.config.pretrained_path)

    def _load_pretrained(self, path: str):
        """
        Load a pre-trained matrix for Warm-Restart.
        """
        if not os.path.isfile(path):
            path = os.path.join(os.path.dirname(__file__), "..", path)

        if not os.path.isfile(path):
            print(f" Matrix pre-trained not found at: {path}")
            return

        try:
            loaded = np.load(path).astype(np.float32)
            if loaded.shape != self.Q.shape:
                if loaded.shape[0] == 1920 and self.Q.shape[0] == 5760:
                    print(f"Warm-Restart Transfer: Expanding 1,920-state matrix into 5,760-state velocity-tiered matrix...")
                    for s_old in range(1920):
                        for tier in range(3):
                            self.Q[s_old * 3 + tier] = loaded[s_old]
                    self.best_Q = self.Q.copy()
                else:
                    print(f"Shape incompatible ({loaded.shape} vs {self.Q.shape}), initializing from scratch.")
                    return
            else:
                self.Q = loaded
                self.best_Q = loaded.copy()
            
            if self.config.epsilon_start == 1.0:
                self.config.epsilon_start = 0.20

            print(f"Warm-Restart Tabular: Matrix loaded from {path} (Shape: {loaded.shape})")
            if not self.config.skip_baseline_eval:
                print("Evaluating pre-trained baseline performance (200 episodes, pure greedy)...")
                base_avg, base_goals, base_score, base_max = self.evaluate(n_episodes=200)
                self.best_score = base_score
                self.best_avg_hits = base_avg
                print(
                    f"Pre-trained Baseline Skill | "
                    f"Avg Hits: {base_avg:.2f} | Goals: {base_goals:.2f} ({int(base_goals*100)}%) | "
                    f"Robust Score: {base_score:.2f} | Max: {base_max} hits"
                )
                print(f"Baseline record protected: New checkpoints must exceed {self.best_score:.2f} to overwrite.")
            else:
                self.best_score = self.config.best_score_baseline
        
        except Exception as e:
            print(f"Error loading matrix {path}: {e}")

    def evaluate(self, n_episodes: int = 200, show_progress: bool = False) -> Tuple[float, float, float, int]:
        """
        Evaluate current Q-table in pure greedy mode (epsilon=0) without updating weights.
        Returns: (avg_hits, avg_goals, score, max_hits)
        """
        eval_hits = []
        eval_goals = []

        # Standardize evaluation difficulty to maximum (100% speed) to ensure fair comparisons
        prev_ratio = self.env.opp_speed_ratio
        if self.config.opponent == "heuristic":
            self.env.set_opponent_speed_ratio(1.0)

        ep_iter = tqdm(range(n_episodes), desc="Evaluating", unit="ep") if show_progress else range(n_episodes)
        for _ in ep_iter:
            state, _ = self.env.reset()
            done = False
            steps = 0
            hits = 0
            goals = 0

            while not done and steps < self.config.max_steps:
                steps += 1
                action = int(np.argmax(self.Q[state]))
                next_state, _, done, info = self.env.step(action)
                if info.get("hit_ai_paddle"):
                    hits += 1
                if info.get("scored_goal"):
                    goals += 1
                state = next_state

            eval_hits.append(hits)
            eval_goals.append(goals)

        # Restore training difficulty
        if self.config.opponent == "heuristic":
            self.env.set_opponent_speed_ratio(prev_ratio)

        avg_hits = float(np.mean(eval_hits)) if eval_hits else 0.0
        p25_hits = float(np.percentile(eval_hits, 25)) if eval_hits else 0.0
        p50_hits = float(np.median(eval_hits)) if eval_hits else 0.0
        p75_hits = float(np.percentile(eval_hits, 75)) if eval_hits else 0.0
        iqr_hits = p75_hits - p25_hits
        avg_goals = float(np.mean(eval_goals)) if eval_goals else 0.0
        max_h = int(max(eval_hits)) if eval_hits else 0

        # Distribution-based robust quality score:
        hit_score = (avg_hits + p50_hits + p25_hits) / 3.0 - (iqr_hits / 4.0)
        score = (hit_score + avg_goals * 5.0) if self.config.opponent == "heuristic" else hit_score
        return avg_hits, avg_goals, score, max_h

    def train(self) -> np.ndarray:
        cfg = self.config
        total_states = self.discretizer.total_states
        resolved_save_path = os.path.join(os.path.dirname(__file__), "..", cfg.save_path)
        os.makedirs(os.path.dirname(resolved_save_path), exist_ok=True)

        print("=" * 65)
        print(f"Starting Q-Learning Tabular Training (Mode {cfg.opponent.capitalize()})")
        print(f"   - State Space: {total_states} discrete states (Fine resolution)")
        print(f"   - Actions: {NUM_ACTIONS} (0: STAY, 1: UP, 2: DOWN)")
        print(f"   - Episodes: {cfg.total_episodes}")
        print(f"   - Steps per Episode: {cfg.max_steps}")
        print(f"   - Hyperparameters: alpha={cfg.alpha}, gamma={cfg.gamma}, eps_decay={cfg.epsilon_decay}, boost_errors={cfg.boost_errors}")
        print("=" * 65)

        # Initialize epsilon with a decay factor applied over the number of episodes
        epsilon = cfg.epsilon_start
        # Track rally hits distribution per episode batch
        batch_hits = []
        batch_goals = []
        # Max rally is the maximum number of hits in a single episode
        max_rally_in_batch = 0
        current_opp_ratio = 0.45

        # Track total simulation steps across all episodes
        total_steps = 0
        report_interval = 50 if cfg.total_episodes <= 500 else (100 if cfg.total_episodes <= 2000 else (1000 if cfg.total_episodes <= 10000 else 2000))

        pbar = tqdm(range(1, cfg.total_episodes + 1), desc=f"Q-Learning ({cfg.opponent})", unit="ep")
        try:
            for episode in pbar:
                if cfg.opponent == "heuristic":
                    # Curriculum Learning: Gradually ramp up opponent difficulty up to 100%
                    progress = (episode - 1) / max(1, cfg.total_episodes - 1)
                    current_opp_ratio = 0.45 + 0.55 * progress
                    self.env.set_opponent_speed_ratio(current_opp_ratio)

                # The state is a discrete representation of the continuous state space
                state, _ = self.env.reset()
                # done is a boolean that indicates if the episode is over
                done = False
                # steps is the number of steps taken in the current episode
                steps = 0
                # episode_hits is the number of times the ball hits the AI paddle in the current episode
                episode_hits = 0
                episode_goals = 0
                # prev_action is the previous action taken by the AI
                prev_action = 0
                # Recent trajectory for retrospective error credit assignment (boosting on bad episodes)
                recent_history = []

                while not done and steps < cfg.max_steps:
                    steps += 1
                    total_steps += 1

                    # Epsilon-greedy policy
                    # Random action selection with probability epsilon
                    if np.random.rand() < epsilon:
                        action = np.random.randint(NUM_ACTIONS)
                    else:
                        # Action selection based on the Q-table
                        action = int(np.argmax(self.Q[state]))

                    # environment executes the action and returns the next state, reward, done and info
                    next_state, reward, done, info = self.env.step(action)

                    # if the ball hits the AI paddle
                    if info.get("hit_ai_paddle"):
                        # increment the number of hits in the current episode
                        episode_hits += 1

                    if info.get("scored_goal"):
                        episode_goals += 1

                    # Anti-jitter penalty: penalize rapid action switching between consecutive steps
                    # to encourage smooth trajectories and prevent erratic paddle oscillations
                    if action != prev_action and steps > 1:
                        reward -= cfg.jitter_penalty
                    
                    # update the previous action
                    prev_action = action

                    # Bellman TD update
                    # If episode is done, the target is the immediate reward
                    # Otherwise, it's the immediate reward plus the discounted maximum Q-value of the next state
                    td_target = reward if done else reward + cfg.gamma * np.max(self.Q[next_state])
                    td_error = td_target - self.Q[state, action]

                    # Residual Error Boosting: Scale learning rate dynamically with the magnitude of the TD error,
                    # prioritizing unexpected outcomes and large penalties (analogous to gradient boosting)
                    if cfg.boost_errors:
                        effective_alpha = cfg.alpha * (1.0 + 0.5 * min(2.0, abs(td_error)))
                        recent_history.append((state, action))
                        if len(recent_history) > 12:
                            recent_history.pop(0)
                    else:
                        effective_alpha = cfg.alpha

                    # Update the Q-value using the TD target and the effective learning rate
                    self.Q[state, action] += effective_alpha * td_error
                    # Update the state
                    state = next_state

                # Backward Failure Credit Assignment (Error Boosting on Missed Balls)
                # When an episode ends in failure (unforced miss), retrospectively penalize the sequence
                # of preceding actions leading up to the goal to accelerate corrective policy adjustments.
                if cfg.boost_errors and (reward <= -4.0 or info.get("missed_ball")) and len(recent_history) > 1:
                    discounted_penalty = reward
                    for past_state, past_action in reversed(recent_history[:-1]):
                        discounted_penalty *= 0.85
                        self.Q[past_state, past_action] += (cfg.alpha * 0.5) * (discounted_penalty - self.Q[past_state, past_action])

                # Record episode hits and goals for batch statistics
                batch_hits.append(episode_hits)
                batch_goals.append(episode_goals)

                # if the number of hits in the current episode is greater than the maximum number of hits in the batch
                if episode_hits > max_rally_in_batch:
                    # update the maximum number of hits in the batch
                    max_rally_in_batch = episode_hits

                epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

                # Update live tqdm progress bar postfix every 50 episodes
                if episode % 50 == 0:
                    pbar.set_postfix({
                        "Steps": total_steps,
                        "Eps": f"{epsilon:.3f}",
                        "Max": max_rally_in_batch,
                        "Record": f"{self.best_score:.2f}"
                    })

                # Print statistics periodically and evaluate against 100% boss
                if episode % report_interval == 0:
                    avg_hits = float(np.mean(batch_hits)) if batch_hits else 0.0
                    p25_hits = float(np.percentile(batch_hits, 25)) if batch_hits else 0.0
                    p50_hits = float(np.median(batch_hits)) if batch_hits else 0.0
                    p75_hits = float(np.percentile(batch_hits, 75)) if batch_hits else 0.0
                    iqr_hits = p75_hits - p25_hits
                    avg_goals = float(np.mean(batch_goals)) if batch_goals else 0.0

                    # Fast pure greedy evaluation (40 episodes) against 100% boss
                    eval_hits, eval_goals, eval_score, eval_max = self.evaluate(n_episodes=40)
                    is_new_best = False
                    if eval_score > self.best_score:
                        pbar.write(f" -> Record achieved! ({eval_score:.2f} > {self.best_score:.2f}). Saving...")
                        self.best_score = eval_score
                        is_new_best = True
                        self.best_Q = self.Q.copy()
                        np.save(resolved_save_path, self.best_Q)

                    best_marker = f" [RECORD: {self.best_score:.2f}]" if is_new_best else ""
                    rival_info = f" | Opponent: {int(current_opp_ratio * 100)}%" if cfg.opponent == "heuristic" else ""
                    goals_text = f" | Goals: {avg_goals:.2f} ({int(avg_goals*100)}%)" if cfg.opponent == "heuristic" else ""

                    pbar.write(
                        f"Episode [{episode:5d}/{cfg.total_episodes}]{rival_info} | "
                        f"Train Avg: {avg_hits:5.2f} | P25: {p25_hits:4.1f} | Med: {p50_hits:4.1f} | IQR: {iqr_hits:4.1f}{goals_text} | "
                        f"Eval Score (eps=0): {eval_score:5.2f} | Max: {max_rally_in_batch:3d} | "
                        f"Eps: {epsilon:.3f}{best_marker}"
                    )
                    batch_hits = []
                    batch_goals = []
                    max_rally_in_batch = 0

        except KeyboardInterrupt:
            print("\nTraining interrupted manually with Ctrl+C.")

        # Restore the best matrix obtained and run final evaluation
        print("\nEvaluating final policy in pure greedy mode (60 episodes, epsilon=0)...")
        self.Q = self.best_Q.copy()
        final_hits, final_goals, final_score, final_max = self.evaluate(n_episodes=60)
        np.save(resolved_save_path, self.best_Q)

        print("=" * 65)
        print("Tabular training completed successfully!")
        print(f"Best Q-Matrix saved at: {os.path.abspath(resolved_save_path)}")
        print(f"Historical record achieved: {self.best_score:.2f} robust score ({final_hits:.2f} avg hits)")
        if cfg.opponent == "heuristic":
            print(f"Final Goals per Episode: {final_goals:.2f} ({int(final_goals * 100)}% win rate)")
        print("=" * 65)
        return self.best_Q


# ==============================================================================
# Deep Q-Network Trainer (DQNTrainer)
# ==============================================================================
class DQNTrainer:
    """
    High-performance Deep Q-Network (DQN) trainer:
    - Replay Buffer with N-Step returns.
    - Soft Target Updates (Polyak Averaging with tau).
    - Dual atomic export: PyTorch checkpoint (.pth) and ONNX model (.onnx).
    - Support for Transfer Learning / Warm-Restart and Curriculum Learning.
    - Safe Ctrl+C capture to always preserve the best record.
    """
    def __init__(self, config: Optional[DQNConfig] = None):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = FastPongEnv(opponent=self.cfg.opponent)

        # Main policy network and target network
        self.policy_net = PongDQN(input_dim=4, hidden_dim=64, output_dim=NUM_ACTIONS).to(self.device)
        self.target_net = PongDQN(input_dim=4, hidden_dim=64, output_dim=NUM_ACTIONS).to(self.device)

        # Storage paths
        self.resolved_save_path = os.path.join(os.path.dirname(__file__), "..", self.cfg.save_path)
        os.makedirs(os.path.dirname(self.resolved_save_path), exist_ok=True)
        self.onnx_save_path = os.path.splitext(self.resolved_save_path)[0] + ".onnx"

        # Buffer and Optimizer
        self.replay_buffer = NStepReplayBuffer(
            capacity=self.cfg.buffer_capacity,
            n_step=self.cfg.n_step,
            gamma=self.cfg.gamma
        )
        # Optimizer for training the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)
        self.criterion = nn.SmoothL1Loss()

        self.best_score = self.cfg.best_score_baseline
        self.best_weights = copy.deepcopy(self.policy_net.state_dict())
        self.is_finetuning = False

        self._setup_transfer_learning()

    def _setup_transfer_learning(self):
        """
        Load previous weights if --pretrained is specified and adjust fine-tuning hyperparameters.
        """
        if not self.cfg.pretrained_path:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()
            return

        resolved = os.path.join(os.path.dirname(__file__), "..", self.cfg.pretrained_path)
        candidates = [resolved, self.cfg.pretrained_path]
        loaded = False

        for candidate in candidates:
            if os.path.isfile(candidate):
                self.policy_net.load_state_dict(torch.load(candidate, map_location=self.device))
                print(f"Warm-Restart: Weights successfully loaded from {candidate}")
                loaded = True
                self.is_finetuning = True
                if not self.cfg.skip_baseline_eval:
                    print("Evaluating pre-trained baseline performance (200 episodes, pure greedy)...")
                    b_hits, b_goals, b_score, b_max = self.evaluate(n_episodes=200)
                    self.best_score = max(self.cfg.best_score_baseline, b_score)
                    rival_info = f" | Goals: {b_goals:.2f} ({int(b_goals*100)}%)" if self.cfg.opponent == "heuristic" else ""
                    print(
                        f"Pre-trained Baseline Skill | "
                        f"Avg Hits: {b_hits:.2f}{rival_info} | Robust Score: {b_score:.2f} | Max: {b_max} hits"
                    )
                    print(f"Baseline record protected: New checkpoints must exceed {self.best_score:.2f} to overwrite.")
                else:
                    self.best_score = self.cfg.best_score_baseline
                break

        if loaded:
            # In fine-tuning, start with bounded exploration and gentle learning rate
            if self.cfg.epsilon_start == 1.0:
                self.cfg.epsilon_start = 0.20
            if self.cfg.lr == 1e-3:
                self.cfg.lr = 1.5e-4
                self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr)

            self.cfg.warmup_steps = min(self.cfg.warmup_steps, self.cfg.batch_size * 2)
            print(f"Fine-Tuning mode active | Initial Epsilon: {self.cfg.epsilon_start} | LR: {self.cfg.lr} | Baseline: {self.best_score:.2f}")
        else:
            print(f"Pretrained model not found at {self.cfg.pretrained_path}, starting from scratch.")

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.best_weights = copy.deepcopy(self.policy_net.state_dict())

    def evaluate(self, n_episodes: int = 200, show_progress: bool = False) -> Tuple[float, float, float, int]:
        """
        Evaluate current DQN policy in pure greedy mode without exploration or training.
        Returns: (avg_hits, avg_goals, score, max_hits)
        """
        self.policy_net.eval()
        hits_list = []
        goals_list = []

        # Standardize evaluation difficulty to maximum (100% speed) to ensure fair comparisons
        prev_ratio = self.env.opp_speed_ratio
        if self.cfg.opponent == "heuristic":
            self.env.set_opponent_speed_ratio(1.0)

        ep_iter = tqdm(range(n_episodes), desc="Evaluating", unit="ep") if show_progress else range(n_episodes)
        for _ in ep_iter:
            self.env.reset()
            state = self.env.get_continuous_state()
            done = False
            steps = 0
            ep_hits = 0
            ep_goals = 0

            while not done and steps < self.cfg.max_steps:
                steps += 1
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                    q_vals = self.policy_net(state_t)
                    action = int(q_vals.argmax(dim=1).item())

                _, _, done, info = self.env.step(action)
                if info.get("hit_ai_paddle"):
                    ep_hits += 1
                if info.get("scored_goal"):
                    ep_goals += 1
                state = self.env.get_continuous_state()

            hits_list.append(ep_hits)
            goals_list.append(ep_goals)

        # Restore training difficulty
        if self.cfg.opponent == "heuristic":
            self.env.set_opponent_speed_ratio(prev_ratio)

        self.policy_net.train()
        avg_hits = float(np.mean(hits_list)) if hits_list else 0.0
        p25_hits = float(np.percentile(hits_list, 25)) if hits_list else 0.0
        p50_hits = float(np.median(hits_list)) if hits_list else 0.0
        p75_hits = float(np.percentile(hits_list, 75)) if hits_list else 0.0
        iqr_hits = p75_hits - p25_hits
        avg_goals = float(np.mean(goals_list)) if goals_list else 0.0
        max_hits = int(max(hits_list)) if hits_list else 0

        # Distribution-based robust quality score:
        # Rewards overall mean, median, and 25th percentile (guaranteed skill floor)
        # while penalizing interquartile dispersion (erratic play).
        hit_score = (avg_hits + p50_hits + p25_hits) / 3.0 - (iqr_hits / 4.0)
        score = (hit_score + avg_goals * 5.0) if self.cfg.opponent == "heuristic" else hit_score
        return avg_hits, avg_goals, score, max_hits

    def save_model_to_disk(self):
        """
        Save PyTorch weights and export the ONNX graph for real-time inference.
        """
        torch.save(self.policy_net.state_dict(), self.resolved_save_path)
        try:
            self.policy_net.eval()
            dummy_input = torch.zeros(1, 4, dtype=torch.float32, device=self.device)
            torch.onnx.export(
                self.policy_net,
                dummy_input,
                self.onnx_save_path,
                input_names=["state"],
                output_names=["q_values"],
                dynamic_axes={"state": {0: "batch_size"}, "q_values": {0: "batch_size"}},
                opset_version=14
            )
        except Exception as e:
            print(f"ONNX export warning: {e}")

    def train_step(self, batch_size: Optional[int] = None):
        """
        Execute a gradient update on a mini-batch from the replay buffer.
        """
        bs = batch_size if batch_size is not None else self.cfg.batch_size
        if len(self.replay_buffer) < bs:
            return

        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample(bs)

        b_s = torch.tensor(b_states, dtype=torch.float32, device=self.device)
        b_a = torch.tensor(b_actions, dtype=torch.long, device=self.device).unsqueeze(1)
        b_r = torch.tensor(b_rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        b_next_s = torch.tensor(b_next_states, dtype=torch.float32, device=self.device)
        b_d = torch.tensor(b_dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a) predicted by the policy network
        current_q = self.policy_net(b_s).gather(1, b_a)

        # Double DQN target calculation:
        # Decouples action selection from action evaluation to mitigate Q-value overestimation bias.
        # 1. Action selection: Use policy_net to choose the best greedy action for next state: a* = argmax_a Q(s', a; theta)
        # 2. Action evaluation: Use target_net to evaluate the value of that chosen action: Q(s', a*; theta_target)
        # 3. Bellman target: Compute n-step discounted target Y = r + (gamma^n) * Q_target * (1 - done)
        with torch.no_grad():
            best_actions = self.policy_net(b_next_s).argmax(dim=1, keepdim=True)
            max_next_q = self.target_net(b_next_s).gather(1, best_actions)
            target_q = b_r + (self.cfg.gamma ** self.cfg.n_step) * max_next_q * (1.0 - b_d)

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Soft target network update (Polyak / soft-target averaging)
        for param_target, param_policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param_target.data.copy_(
                self.cfg.tau * param_policy.data + (1.0 - self.cfg.tau) * param_target.data
            )

    def train(self) -> PongDQN:
        """
        Main DQN training loop.
        """
        cfg = self.cfg

        print("=" * 65)
        print(f" Starting DQN Training | Opponent: {cfg.opponent.upper()}")
        print(f"   - Device: {self.device}")
        print(f"   - Mode: {'Fronton' if cfg.opponent == 'fronton' else 'Vs Heuristic Opponent'}")
        print(f"   - Episodes: {cfg.total_episodes}")
        print(f"   - Max steps per episode: {cfg.max_steps}")
        print(f"   - Batch Size: {cfg.batch_size} | Train Freq: every {cfg.train_freq} steps")
        print(f"   - LR: {cfg.lr} | Gamma: {cfg.gamma} | Epsilon Start: {cfg.epsilon_start}")
        print(f"   - N-Step: {cfg.n_step} | Reward Shaping Active")
        print("=" * 65)

        epsilon = cfg.epsilon_start
        total_steps = 0
        batch_hits = []
        batch_goals = []
        max_rally_in_batch = 0
        current_opp_ratio = 0.45

        report_interval = 50 if cfg.total_episodes <= 500 else 100
        pbar = tqdm(range(1, cfg.total_episodes + 1), desc=f"DQN ({cfg.opponent})", unit="ep")

        try:
            for episode in pbar:

                if cfg.opponent == "heuristic":
                    # Curriculum Learning: Gradually ramp up opponent difficulty as training progresses.
                    # Starting at 45% speed allows the agent to learn ball intercept basics easily.
                    # As episodes advance, linear scaling increases speed up to 100%, preventing
                    # early exploration collapse and preparing the policy for high-speed rallies.
                    progress = (episode - 1) / max(1, cfg.total_episodes - 1)
                    current_opp_ratio = 0.45 + 0.55 * progress
                    self.env.set_opponent_speed_ratio(current_opp_ratio)

                self.env.reset()
                # State: Normalize paddle y, ball x, ball y, ball vx, ball vy
                state = self.env.get_continuous_state()
                # done flag for episode termination
                done = False
                # Number of steps in current episode
                steps = 0
                # Number of hits in current episode
                episode_hits = 0
                # Previous action taken
                prev_action = 0

                episode_goals = 0

                while not done and steps < cfg.max_steps:
                    steps += 1
                    total_steps += 1

                    # Epsilon-greedy action selection
                    if random.random() < epsilon:
                        action = random.randint(0, NUM_ACTIONS - 1)
                    else:
                        with torch.no_grad():
                            # Policy network inference for action selection
                            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            q_vals = self.policy_net(state_t)
                            action = int(q_vals.argmax(dim=1).item())

                    _, reward, done, info = self.env.step(action)
                    next_state = self.env.get_continuous_state()

                    if info.get("hit_ai_paddle"):
                        episode_hits += 1

                    if info.get("scored_goal"):
                        episode_goals += 1

                    # Penalty for erratic action changes
                    if action != prev_action and steps > 1:
                        reward -= cfg.jitter_penalty
                    prev_action = action

                    # Store transition in replay buffer
                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    # Periodically update the neural network via backpropagation:
                    # 1. Warmup check: Ensure replay buffer has enough diverse transitions before training.
                    # 2. Training frequency: Perform gradient step every `train_freq` steps (modulo check)
                    #    to balance simulation throughput and optimization stability.
                    if len(self.replay_buffer) >= cfg.warmup_steps and (total_steps % cfg.train_freq == 0):
                        self.train_step()

                batch_hits.append(episode_hits)
                batch_goals.append(episode_goals)
                if episode_hits > max_rally_in_batch:
                    max_rally_in_batch = episode_hits

                epsilon = max(cfg.epsilon_min, epsilon * cfg.epsilon_decay)

                # Update live tqdm progress bar postfix every 10 episodes
                if episode % 10 == 0:
                    pbar.set_postfix({
                        "Steps": total_steps,
                        "Eps": f"{epsilon:.3f}",
                        "Buffer": len(self.replay_buffer),
                        "Record": f"{self.best_score:.2f}"
                    })

                # Report and save checkpoint periodically
                if episode % report_interval == 0:
                    avg_hits = float(np.mean(batch_hits)) if batch_hits else 0.0
                    p25_hits = float(np.percentile(batch_hits, 25)) if batch_hits else 0.0
                    p50_hits = float(np.median(batch_hits)) if batch_hits else 0.0
                    p75_hits = float(np.percentile(batch_hits, 75)) if batch_hits else 0.0
                    iqr_hits = p75_hits - p25_hits
                    avg_goals = float(np.mean(batch_goals)) if batch_goals else 0.0

                    # Fast pure greedy evaluation (40 episodes)
                    eval_hits, eval_goals, eval_score, eval_max = self.evaluate(n_episodes=40)
                    is_new_best = False
                    
                    if eval_score > self.best_score:
                        pbar.write(f" -> Record achieved! ({eval_score:.2f} > {self.best_score:.2f}). Saving...")
                        self.best_score = eval_score
                        is_new_best = True
                        self.best_weights = copy.deepcopy(self.policy_net.state_dict())
                        self.save_model_to_disk()

                    best_marker = f" [RECORD: {self.best_score:.2f}]" if is_new_best else ""
                    rival_info = f" | Opponent: {int(current_opp_ratio * 100)}%" if cfg.opponent == "heuristic" else ""
                    goals_text = f" | Goals: {avg_goals:.2f} ({int(avg_goals*100)}%)" if cfg.opponent == "heuristic" else ""

                    pbar.write(
                        f"Episode [{episode:4d}/{cfg.total_episodes}]{rival_info} | "
                        f"Train Avg: {avg_hits:4.2f} | P25: {p25_hits:4.1f} | Med: {p50_hits:4.1f} | IQR: {iqr_hits:4.1f}{goals_text} | "
                        f"Eval Score (eps=0): {eval_score:5.2f} | Max: {eval_max:2d} | "
                        f"Eps: {epsilon:.3f}{best_marker}"
                    )
                    batch_hits = []
                    batch_goals = []
                    max_rally_in_batch = 0

        except KeyboardInterrupt:
            print("\nTraining interrupted manually with Ctrl+C.")
            print("Evaluating and restoring best model weights...")

        # Final greedy evaluation pass (epsilon=0) to ensure the latest weights didn't beat the record
        try:
            print("\nEvaluating final policy in pure greedy mode (60 episodes, epsilon=0)...")
            final_hits, final_goals, final_score, final_max = self.evaluate(n_episodes=60)
            if final_score > self.best_score:
                print(f"Final policy achieved NEW RECORD: {final_score:.2f} > {self.best_score:.2f}!")
                self.best_score = final_score
                self.best_weights = copy.deepcopy(self.policy_net.state_dict())
            else:
                print(f"Final policy ({final_score:.2f}) did not exceed record ({self.best_score:.2f}). Restoring best checkpoint.")
        except Exception as e:
            print(f"Final evaluation notice: {e}")

        # Always restore best checkpoint and save to disk
        self.policy_net.load_state_dict(self.best_weights)
        self.save_model_to_disk()

        print("=" * 65)
        print("Save completed successfully!")
        print(f"Best DQN model saved at: {os.path.abspath(self.resolved_save_path)}")
        print(f"ONNX model exported to: {os.path.abspath(self.onnx_save_path)}")
        print(f"Historical record achieved: {self.best_score:.2f}")
        print("=" * 65)
        return self.policy_net

    def train_failures(self, failures_path: str, attempts_per_shot: int = 10, anchor_episodes: int = 30) -> PongDQN:
        """
        Failure-Targeted Replay Training (Failure Clinic).
        Loads snapshots of shots where the agent conceded a goal and runs targeted
        counterfactual search to learn how to intercept and save them.
        Uses balanced anchor gameplay against the heuristic bot to prevent catastrophic policy drift.
        """
        resolved_path = os.path.join(os.path.dirname(__file__), "..", failures_path)
        if not os.path.isfile(resolved_path):
            resolved_path = failures_path

        # If specified as .json but .jsonl exists (or vice versa), auto-resolve to active dataset
        if not os.path.isfile(resolved_path):
            if resolved_path.endswith(".json") and os.path.isfile(resolved_path + "l"):
                resolved_path = resolved_path + "l"
            elif resolved_path.endswith(".json") and os.path.isfile(resolved_path.replace(".json", ".jsonl")):
                resolved_path = resolved_path.replace(".json", ".jsonl")
            elif resolved_path.endswith(".jsonl") and os.path.isfile(resolved_path[:-1]):
                resolved_path = resolved_path[:-1]

        if not os.path.isfile(resolved_path):
            print(f"Error: Failures dataset not found at: {failures_path}")
            return self.policy_net

        with open(resolved_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                failures = []
            elif content.startswith("["):
                failures = json.loads(content)
            else:
                failures = [json.loads(line) for line in content.splitlines() if line.strip()]

        if not failures:
            print("No failure records found in dataset.")
            return self.policy_net

        print("=" * 65)
        print("Starting Failure-Targeted Clinic for DQN (Balanced Anchor Mode)")
        print(f"   - Dataset: {resolved_path} ({len(failures)} failure shots)")
        print(f"   - Max attempts per shot: {attempts_per_shot}")
        print(f"   - Balanced anchor episodes: {anchor_episodes}")
        print(f"   - Device: {self.device}")
        print("=" * 65)

        # Establish protected baseline score before clinic training
        self.best_weights = copy.deepcopy(self.policy_net.state_dict())
        print("Evaluating pre-clinic baseline performance (40 episodes, pure greedy)...")
        b_hits, b_goals, b_score, b_max = self.evaluate(n_episodes=40)
        self.best_score = b_score
        print(
            f"Pre-clinic Baseline Skill | "
            f"Avg Hits: {b_hits:.2f} | Goals: {b_goals:.2f} ({int(b_goals*100)}%) | Robust Score: {self.best_score:.2f} | Max: {b_max} hits"
        )
        print(f"Protection active: Post-clinic policy must exceed {self.best_score:.2f} to overwrite model weights.\n")

        # 1. Anchor Buffer: Pre-populate replay buffer with diverse, balanced full-court gameplay
        if anchor_episodes > 0:
            print(f"Pre-populating Replay Buffer with {anchor_episodes} balanced anchor episodes against heuristic bot...")
            saved_opp = self.env.opponent
            self.env.opponent = "heuristic"
            anchor_pbar = tqdm(range(anchor_episodes), desc="Anchor Games", unit="ep")
            for _ in anchor_pbar:
                self.env.reset()
                state = self.env.get_continuous_state()
                done = False
                prev_action = 0
                step_count = 0
                while not done and step_count < 350:
                    step_count += 1
                    if random.random() < 0.05:
                        action = random.randint(0, NUM_ACTIONS - 1)
                    else:
                        with torch.no_grad():
                            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            action = int(self.policy_net(state_t).argmax(dim=1).item())

                    _, reward, done, info = self.env.step(action)
                    next_state = self.env.get_continuous_state()
                    if action != prev_action and step_count > 1:
                        reward -= self.cfg.jitter_penalty
                    prev_action = action

                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state
            self.env.opponent = saved_opp
            print(f"Anchor initialized! Replay buffer contains {len(self.replay_buffer)} balanced transitions.\n")

        total_saved = 0
        pbar = tqdm(failures, desc="Failures Clinic", unit="shot")

        for idx, shot in enumerate(pbar, 1):
            saved = False

            for attempt in range(1, attempts_per_shot + 1):
                # Guided exploration noise: start at 0.35, decay towards 0.05
                epsilon = max(0.05, 0.35 * (1.0 - (attempt - 1) / float(attempts_per_shot)))

                self.env.reset_to_shot(shot)
                state = self.env.get_continuous_state()
                done = False
                steps = 0
                prev_action = 0

                while not done and steps < 250:
                    steps += 1
                    if random.random() < epsilon:
                        action = random.randint(0, NUM_ACTIONS - 1)
                    else:
                        with torch.no_grad():
                            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                            q_vals = self.policy_net(state_t)
                            action = int(q_vals.argmax(dim=1).item())

                    _, reward, done, info = self.env.step(action)
                    next_state = self.env.get_continuous_state()

                    # High redemption bonus for successfully intercepting the shot
                    if info.get("hit_ai_paddle"):
                        reward += 5.0
                        saved = True

                    if action != prev_action and steps > 1:
                        reward -= self.cfg.jitter_penalty
                    prev_action = action

                    self.replay_buffer.push(state, action, reward, next_state, done)
                    state = next_state

                    if len(self.replay_buffer) >= 32:
                        self.train_step(batch_size=min(64, len(self.replay_buffer)))

                    if info.get("hit_ai_paddle"):
                        break

                if saved:
                    total_saved += 1
                    # Extra consolidation gradient steps with balanced buffer
                    for _ in range(4):
                        self.train_step(batch_size=min(64, len(self.replay_buffer)))
                    break

            pbar.set_postfix({
                "Saved": f"{total_saved}/{idx}",
                "Rate": f"{(total_saved / max(1, idx))*100:.1f}%"
            })

        print("=" * 65)
        print(f"Failure Clinic Finished! Successfully saved {total_saved} / {len(failures)} shots ({(total_saved / max(1, len(failures)))*100:.1f}%)")

        # Rigorous Two-Phase evaluation before saving to prevent policy degradation
        print("\nEvaluating post-clinic policy in pure greedy mode (Phase 1: 40 episodes)...")
        post_hits, post_goals, post_score, post_max = self.evaluate(n_episodes=40)

        saved_successfully = False
        if post_score > self.best_score:
            print(f" -> Potential improvement ({post_score:.2f} > {self.best_score:.2f}). Running 100 confirmation episodes...")
            conf_hits, conf_goals, conf_score, conf_max = self.evaluate(n_episodes=100)
            if conf_score > self.best_score:
                print(f" -> Clinic improvement CONFIRMED! ({conf_score:.2f} > {self.best_score:.2f}). Saving to disk...")
                self.best_score = conf_score
                self.best_weights = copy.deepcopy(self.policy_net.state_dict())
                self.save_model_to_disk()
                saved_successfully = True
            else:
                print(f" -> Clinic improvement REJECTED. Confirmation score was only {conf_score:.2f} <= {self.best_score:.2f}.")
        else:
            print(f" -> Post-clinic score ({post_score:.2f}) did not exceed baseline ({self.best_score:.2f}).")

        if not saved_successfully:
            print("Restoring protected baseline weights (discarding degraded clinic weights).")
            self.policy_net.load_state_dict(self.best_weights)
        else:
            print(f"DQN weights saved at: {os.path.abspath(self.resolved_save_path)}")
            print(f"ONNX model exported to: {os.path.abspath(self.onnx_save_path)}")
        print("=" * 65)
        return self.policy_net


def main():
    parser = argparse.ArgumentParser(
        description="Train RL algorithms for Pong",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--algo",
        type=str,
        choices=["q_learning", "tabular", "dqn", "heuristic"],
        default="q_learning",
        help="Algorithm to train or evaluate: 'q_learning' (tabular), 'dqn' (neural network), or 'heuristic'"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Total number of episodes (default 30000 for tabular, 1200 for dqn)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["fronton", "heuristic"],
        default="fronton",
        help="Opponent type: 'fronton' (wall) or 'heuristic' (paddle)"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pre-trained model for Transfer Learning (e.g. 'models/dqn_pong.pth')"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the model"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate (e.g., 0.00015 for fine-tuning, 0.001 from scratch)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Tabular Q-learning learning rate (e.g., 0.10 or 0.15)"
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=None,
        help="Initial epsilon for exploration (e.g., 0.20 for fine-tuning, 1.0 from scratch)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=3500,
        help="Max steps per episode"
    )
    parser.add_argument(
        "--no_boost",
        action="store_true",
        help="Disable residual error boosting and backward failure credit assignment"
    )
    parser.add_argument(
        "--train_failures",
        type=str,
        default=None,
        help="Path to failed_shots.json for targeted failure replay training"
    )
    parser.add_argument(
        "--attempts_per_shot",
        type=int,
        default=10,
        help="Max search exploration attempts per failed shot"
    )
    parser.add_argument(
        "--anchor_episodes",
        type=int,
        default=30,
        help="Number of balanced anchor episodes against heuristic bot to pre-populate replay buffer"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run pure greedy evaluation (epsilon=0) without training"
    )

    args = parser.parse_args()

    if args.eval:
        eval_episodes = args.episodes or 200
        print("=" * 65)
        print(f"Starting Pure Greedy Evaluation ({eval_episodes} episodes, epsilon=0)")
        print(f"   - Algorithm: {args.algo.upper()}")
        print(f"   - Opponent: {args.opponent.upper()}")
        if args.algo == "heuristic":
            print(f"   - Model: Rule-Based Tracking Heuristic (Right Paddle)")
            print("=" * 65)
            env = FastPongEnv(opponent=args.opponent, opp_speed_ratio=1.0)
            hits_list = []
            goals_list = []
            ep_iter = tqdm(range(eval_episodes), desc="Evaluating", unit="ep")
            for _ in ep_iter:
                env.reset()
                done = False
                steps = 0
                ep_hits = 0
                ep_goals = 0
                while not done and steps < 3500:
                    steps += 1
                    dy = env.ball_y - env.ai_y
                    if abs(dy) < 15:
                        action = 0
                    elif dy < 0:
                        action = 1
                    else:
                        action = 2
                    _, _, done, info = env.step(action)
                    if info.get("hit_ai_paddle"):
                        ep_hits += 1
                    if info.get("scored_goal"):
                        ep_goals += 1
                hits_list.append(ep_hits)
                goals_list.append(ep_goals)

            avg_hits = float(np.mean(hits_list)) if hits_list else 0.0
            p25_hits = float(np.percentile(hits_list, 25)) if hits_list else 0.0
            p50_hits = float(np.median(hits_list)) if hits_list else 0.0
            p75_hits = float(np.percentile(hits_list, 75)) if hits_list else 0.0
            iqr_hits = p75_hits - p25_hits
            avg_goals = float(np.mean(goals_list)) if goals_list else 0.0
            max_hits = int(max(hits_list)) if hits_list else 0

            hit_score = (avg_hits + p50_hits + p25_hits) / 3.0 - (iqr_hits / 4.0)
            score = (hit_score + avg_goals * 5.0) if args.opponent == "heuristic" else hit_score
        elif args.algo == "q_learning":
            model_file = args.pretrained or "models/q_table.npy"
            print(f"   - Model: {model_file}")
            print("=" * 65)
            tab_cfg = TabularConfig(
                opponent=args.opponent,
                pretrained_path=model_file,
                skip_baseline_eval=True
            )
            trainer = TabularTrainer(tab_cfg)
            avg_hits, avg_goals, score, max_hits = trainer.evaluate(n_episodes=eval_episodes, show_progress=True)
        else:
            model_file = args.pretrained or "models/dqn_pong.onnx"
            print(f"   - Model: {model_file}")
            print("=" * 65)
            model_path = os.path.abspath(model_file)
            if not os.path.isfile(model_path):
                model_path = os.path.join(os.path.dirname(__file__), "..", model_file)

            if model_path.endswith(".onnx"):
                import onnxruntime as ort
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                input_name = session.get_inputs()[0].name
                env = FastPongEnv(opponent=args.opponent, opp_speed_ratio=1.0)
                hits_list = []
                goals_list = []
                ep_iter = tqdm(range(eval_episodes), desc="Evaluating (ONNX Runtime)", unit="ep")
                st_buf = np.empty((1, 4), dtype=np.float32)
                for _ in ep_iter:
                    env.reset()
                    state = env.get_continuous_state()
                    done = False
                    steps = 0
                    ep_hits = 0
                    ep_goals = 0
                    while not done and steps < (args.max_steps or 3500):
                        steps += 1
                        st_buf[0] = state
                        q_vals = session.run(None, {input_name: st_buf})[0]
                        action = int(np.argmax(q_vals[0]))
                        _, _, done, info = env.step(action)
                        if info.get("hit_ai_paddle"):
                            ep_hits += 1
                        if info.get("scored_goal"):
                            ep_goals += 1
                        state = env.get_continuous_state()
                    hits_list.append(ep_hits)
                    goals_list.append(ep_goals)

                avg_hits = float(np.mean(hits_list)) if hits_list else 0.0
                p25_hits = float(np.percentile(hits_list, 25)) if hits_list else 0.0
                p50_hits = float(np.median(hits_list)) if hits_list else 0.0
                p75_hits = float(np.percentile(hits_list, 75)) if hits_list else 0.0
                iqr_hits = p75_hits - p25_hits
                avg_goals = float(np.mean(goals_list)) if goals_list else 0.0
                max_hits = int(max(hits_list)) if hits_list else 0
                hit_score = (avg_hits + p50_hits + p25_hits) / 3.0 - (iqr_hits / 4.0)
                score = (hit_score + avg_goals * 5.0) if args.opponent == "heuristic" else hit_score
            else:
                dqn_cfg = DQNConfig(
                    opponent=args.opponent,
                    pretrained_path=model_path,
                    skip_baseline_eval=True
                )
                trainer = DQNTrainer(dqn_cfg)
                trainer.policy_net.load_state_dict(torch.load(model_path, map_location=trainer.device))
                avg_hits, avg_goals, score, max_hits = trainer.evaluate(n_episodes=eval_episodes, show_progress=True)
        print("=" * 65)
        print("Evaluation Finished!")
        print(f"   - Average Hits: {avg_hits:.2f}")
        print(f"   - Goals Scored: {avg_goals:.2f} ({int(avg_goals * 100)}%)")
        print(f"   - Robust Quality Score: {score:.2f}")
        print(f"   - Max Rally: {max_hits} hits")
        print("=" * 65)
        return

    if args.train_failures:
        dqn_cfg = DQNConfig(
            save_path=args.save_path or "models/dqn_pong.pth",
            pretrained_path=args.pretrained or "models/dqn_pong.pth",
            max_steps=args.max_steps,
            skip_baseline_eval=True
        )
        if args.lr is not None:
            dqn_cfg.lr = args.lr
        trainer = DQNTrainer(dqn_cfg)
        trainer.train_failures(
            args.train_failures,
            attempts_per_shot=args.attempts_per_shot,
            anchor_episodes=args.anchor_episodes
        )
        return

    if args.algo == "q_learning":
        cfg = TabularConfig(
            total_episodes=args.episodes or 50000,
            opponent=args.opponent,
            save_path=args.save_path or "models/q_table.npy",
            pretrained_path=args.pretrained,
            max_steps=args.max_steps,
            boost_errors=not args.no_boost
        )
        if args.alpha is not None:
            cfg.alpha = args.alpha
        if args.epsilon_start is not None:
            cfg.epsilon_start = args.epsilon_start

        TabularTrainer(cfg).train()

    elif args.algo == "dqn":
        dqn_cfg = DQNConfig(
            total_episodes=args.episodes or 1200,
            opponent=args.opponent,
            save_path=args.save_path or "models/dqn_pong.pth",
            pretrained_path=args.pretrained,
            max_steps=args.max_steps
        )
        if args.lr is not None:
            dqn_cfg.lr = args.lr
        if args.epsilon_start is not None:
            dqn_cfg.epsilon_start = args.epsilon_start

        DQNTrainer(dqn_cfg).train()


if __name__ == "__main__":
    main()