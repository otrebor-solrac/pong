"""
N-Step Experience Replay Buffer for Deep Q-Networks (DQN)
---------------------------------------------------------------
Stores transitions and computes multi-step discounted returns
to reduce temporal myopia and accelerate reward propagation.
"""

import random
from collections import deque
from typing import Tuple
import numpy as np


class NStepReplayBuffer:
    """
    Multi-Step Experience Buffer (N-Step Returns):
    Accumulates n consecutive transitions and computes the discounted return:
        R_n = Σ_{i=0}^{n-1} gamma^i * r_{t+i}
    """
    def __init__(self, capacity: int = 50000, n_step: int = 5, gamma: float = 0.99):
        self.capacity = capacity
        # Main replay buffer to store completed transitions (n-step)
        self.buffer = deque(maxlen=capacity)
        # Temporal buffer to accumulate n steps
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def _compute_n_step_return(self) -> Tuple[np.ndarray, int, float, np.ndarray, bool]:
        """
        Compute discounted return R_n over the n-step window.
        """
        reward = 0.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            reward += (self.gamma ** i) * r

        state, action, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state, done = self.n_step_buffer[-1]
        return state, action, reward, next_state, done

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Append transition and emit to main replay buffer once n steps accumulate or on episode termination (done=True).
        """
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) == self.n_step or done:
            n_state, n_action, n_reward, n_next, n_done = self._compute_n_step_return()
            self.buffer.append((n_state, n_action, n_reward, n_next, n_done))
            if done:
                self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a uniform mini-batch converted to NumPy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
