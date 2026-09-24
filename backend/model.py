"""
Pong RL Agent and State Inference Module
---------------------------------------------------------------
Defines the PongQAgent class managing action predictions for the AI paddle,
supporting Tabular Q-Learning, Deep Q-Networks (PyTorch / ONNX), and heuristics.

EMBEDDING POINTS:
1. Continuous -> Discrete State Conversion: see `discretize_state()`
2. Model Checkpoint Loading (Q-Matrix / ONNX / PyTorch): see `load_q_table()` / `load_onnx_model()`
3. Policy Action Inference: see `get_action()`
"""

import os
import numpy as np
from typing import Tuple, Dict, Any, Optional

from constants import (
    ACTION_STAY,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_NAMES,
    FIELD_WIDTH,
    FIELD_HEIGHT,
    PADDLE_HEIGHT,
    PADDLE_OFFSET_X,
    BALL_SIZE,
    MAX_BALL_SPEED
)

from env import PongStateDiscretizer

import torch
import torch.nn as nn
import onnxruntime as ort


class PongDQN(nn.Module):
    """
    Neural Network for Deep Q-Learning in Pong.
    Input: 4 continuous normalized dimensions [dx, dy, vx, vy].
    Output: 3 Q-values for actions [0: STAY, 1: UP, 2: DOWN].
    """
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64, output_dim: int = 3):
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


class PongQAgent:
    """
    Pong Q-Learning agent supporting multiple policies:
      - 'random': Uniformly random action selection (initial default mode).
      - 'q_learning': Direct inference over discrete Q[s, a] matrix.
      - 'dqn': Continuous state inference with Deep Q-Network (ONNX / PyTorch).
      - 'heuristic': Simple tracking heuristic rule (for baseline comparisons).
    """
    def __init__(self, mode: str = "random"):
        self.discretizer = PongStateDiscretizer()
        self.n_actions = 3  # [0: STAY, 1: UP, 2: DOWN]
        self.mode = mode
        
        # Tabular Q-Matrix
        self.q_table: Optional[np.ndarray] = None
        # PyTorch DQN Model
        self.dqn_model = None
        # Optimized ONNX Runtime Sessions (4D Classic and 5D No-Blind)
        self.onnx_session_4d = None
        self.onnx_session_5d = None
        self.onnx_session = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_loaded = False
        self.model_path = ""

    def load_onnx_model(self, filepath: str = "models/dqn_pong.onnx") -> bool:
        """
        Load optimized ONNX model using ONNX Runtime for high-performance inference.
        Detects model dimensionality (4D or 5D) and routes to onnx_session_4d or onnx_session_5d.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            filepath,
            os.path.join(base_dir, filepath),
            os.path.join(base_dir, "..", filepath),
            os.path.join(base_dir, "..", "models", os.path.basename(filepath)),
            os.path.join("/workspace", filepath),
            "/workspace/rl_lab/pong/models/dqn_pong.onnx",
            os.path.join(base_dir, "..", "models", "dqn_pong.onnx"),
            os.path.join(base_dir, "..", "models", "dqn_noblind.onnx")
        ]

        resolved = None
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                resolved = candidate
                break

        if not resolved:
            print(f"[PongQAgent] ONNX file not found. Searched at: {filepath}")
            return False

        try:
            avail = ort.get_available_providers()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in avail else ['CPUExecutionProvider']
            session = ort.InferenceSession(resolved, providers=providers)
            in_dim = session.get_inputs()[0].shape[1] if len(session.get_inputs()[0].shape) > 1 and isinstance(session.get_inputs()[0].shape[1], int) else (5 if "noblind" in resolved else 4)

            if in_dim == 5 or "noblind" in resolved:
                self.onnx_session_5d = session
                self.onnx_path_5d = resolved
                self.last_onnx_5d_mtime = os.path.getmtime(resolved)
                print(f"[PongQAgent] 5D No-Blind ONNX loaded from {resolved} (Providers: {providers})!")
            else:
                self.onnx_session_4d = session
                self.onnx_session = session
                self.onnx_path = resolved
                self.last_onnx_mtime = os.path.getmtime(resolved)
                print(f"[PongQAgent] 4D Classic ONNX loaded from {resolved} (Providers: {providers})!")

            self.model_loaded = True
            self.model_path = resolved
            return True
        except Exception as e:
            print(f"[PongQAgent] Error initializing ONNX session from {resolved}: {e}")
            return False

    def check_auto_reload(self) -> bool:
        """
        Check if ONNX (4D / 5D) or Q-table model files on disk were modified and hot-reload them seamlessly.
        """
        reloaded = False
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Check Classic 4D ONNX file
        onnx_candidates = [
            getattr(self, "onnx_path", None),
            "models/dqn_pong.onnx",
            os.path.join(base_dir, "..", "models", "dqn_pong.onnx"),
            "/workspace/models/dqn_pong.onnx"
        ]
        for p in onnx_candidates:
            if p and os.path.isfile(p):
                try:
                    mtime = os.path.getmtime(p)
                    last_mtime = getattr(self, "last_onnx_mtime", 0.0)
                    if mtime > last_mtime:
                        print(f"[PongQAgent] Detected updated 4D ONNX weights ({mtime} > {last_mtime}). Hot-reloading...")
                        if self.load_onnx_model(p):
                            reloaded = True
                            break
                except Exception as e:
                    print(f"[PongQAgent] Auto-reload 4D ONNX warning: {e}")

        # 2. Check No-Blind 5D ONNX file
        onnx_5d_candidates = [
            getattr(self, "onnx_path_5d", None),
            "models/dqn_noblind.onnx",
            os.path.join(base_dir, "..", "models", "dqn_noblind.onnx"),
            "/workspace/models/dqn_noblind.onnx"
        ]
        for p in onnx_5d_candidates:
            if p and os.path.isfile(p):
                try:
                    mtime = os.path.getmtime(p)
                    last_mtime = getattr(self, "last_onnx_5d_mtime", 0.0)
                    if mtime > last_mtime:
                        print(f"[PongQAgent] Detected updated 5D ONNX weights ({mtime} > {last_mtime}). Hot-reloading...")
                        if self.load_onnx_model(p):
                            reloaded = True
                            break
                except Exception as e:
                    print(f"[PongQAgent] Auto-reload 5D ONNX warning: {e}")

        # 3. Check Q-table file
        q_candidates = [
            getattr(self, "q_table_path", None),
            "models/q_table.npy",
            os.path.join(base_dir, "..", "models", "q_table.npy"),
            "/workspace/models/q_table.npy"
        ]
        for p in q_candidates:
            if p and os.path.isfile(p):
                try:
                    mtime = os.path.getmtime(p)
                    last_mtime = getattr(self, "last_q_table_mtime", 0.0)
                    if mtime > last_mtime:
                        print(f"[PongQAgent] Detected updated Q-Table weights ({mtime} > {last_mtime}). Hot-reloading...")
                        if self.load_q_table(p):
                            reloaded = True
                            break
                except Exception as e:
                    print(f"[PongQAgent] Auto-reload Q-table warning: {e}")

        return reloaded

    def load_dqn_model(self, filepath: str = "models/dqn_pong.pth") -> bool:
        """
        Load weights of a PongDQN neural network from a `.pth` or `.pt` file.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            filepath,
            os.path.join(base_dir, filepath),
            os.path.join(base_dir, "..", filepath),
            os.path.join(base_dir, "..", "models", os.path.basename(filepath)),
            os.path.join("/workspace", filepath),
            "/workspace/rl_lab/pong/models/dqn_pong.pth",
            os.path.join(base_dir, "..", "models", "dqn_pong.pth")
        ]

        resolved = None
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                resolved = candidate
                break

        if not resolved:
            print(f"[PongQAgent] DQN file not found. Searched at: {filepath}")
            return False

        try:
            model = PongDQN(input_dim=4, hidden_dim=64, output_dim=self.n_actions).to(self.device)
            state_dict = torch.load(resolved, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()

            self.dqn_model = model
            self.model_loaded = True
            self.model_path = resolved
            self.mode = "dqn"
            print(f"[PongQAgent] DQN model loaded successfully from {resolved} on device {self.device}!")
            return True
        except Exception as e:
            print(f"[PongQAgent] Error loading DQN model from {resolved}: {e}")
            return False

    def load_q_table(self, filepath: str = "models/q_table.npy") -> bool:
        """
        ========================================================================
        EMBEDDING POINT #2: TRAINED MODEL CHECKPOINT LOADING
        ========================================================================
        Load Q-table from `.npy`, `.onnx`, or `.pth`.
        """
        if filepath.endswith(".onnx"):
            return self.load_onnx_model(filepath)

        if filepath.endswith(".pth") or filepath.endswith(".pt"):
            return self.load_dqn_model(filepath)

        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            filepath,
            os.path.join(base_dir, filepath),
            os.path.join(base_dir, "..", filepath),
            os.path.join(base_dir, "..", "models", os.path.basename(filepath)),
            os.path.join("/workspace", filepath),
            "/workspace/rl_lab/pong/models/q_table.npy",
            os.path.join(base_dir, "..", "models", "q_table.npy")
        ]

        resolved = None
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                resolved = candidate
                break

        if not resolved:
            print(f"[PongQAgent] File not found. Searched candidates for: {filepath}")
            return False

        try:
            if resolved.endswith(".npy"):
                loaded = np.load(resolved)
            elif resolved.endswith(".npz"):
                data = np.load(resolved)
                loaded = data['q_table'] if 'q_table' in data else data[data.files[0]]
            else:
                import pickle
                with open(resolved, "rb") as f:
                    loaded = pickle.load(f)

            if isinstance(loaded, np.ndarray) and loaded.ndim == 2 and loaded.shape[1] == self.n_actions:
                self.q_table = loaded.astype(np.float32)
                self.model_loaded = True
                self.model_path = resolved
                self.q_table_path = resolved
                self.last_q_table_mtime = os.path.getmtime(resolved)
                self.mode = "q_learning"
                print(f"[PongQAgent] Q-Matrix loaded successfully from {resolved}! Shape: {self.q_table.shape}")
                return True
            else:
                print(f"[PongQAgent] Invalid Q-Matrix format: shape={getattr(loaded, 'shape', None)}")
                return False

        except Exception as e:
            print(f"[PongQAgent] Error loading Q-Matrix from {resolved}: {e}")
            return False

    def generate_demo_q_table(self) -> np.ndarray:
        """
        Generate a synthetic demonstration Q-table based on vertical offset dy,
        used to test 'q_learning' mode prior to completing full training.
        """
        total = self.discretizer.total_states
        q = np.zeros((total, self.n_actions), dtype=np.float32)
        
        # Populate Q-table with values favoring UP when dy < 0 and DOWN when dy > 0
        for s in range(total):
            rem = s % (self.discretizer.n_bins_dy * self.discretizer.n_bins_vx * self.discretizer.n_bins_vy)
            bin_dy = rem // (self.discretizer.n_bins_vx * self.discretizer.n_bins_vy)
            
            mid = self.discretizer.n_bins_dy // 2
            if bin_dy < mid - 1:
                # Ball is above the paddle -> Prefer UP (action 1)
                q[s, ACTION_UP] = 1.0 + np.random.uniform(0.1, 0.5)
                q[s, ACTION_DOWN] = 0.0
                q[s, ACTION_STAY] = 0.2
            elif bin_dy > mid + 1:
                # Ball is below the paddle -> Prefer DOWN (action 2)
                q[s, ACTION_UP] = 0.0
                q[s, ACTION_DOWN] = 1.0 + np.random.uniform(0.1, 0.5)
                q[s, ACTION_STAY] = 0.2
            else:
                # Aligned -> Prefer STAY (action 0)
                q[s, ACTION_STAY] = 1.0
                q[s, ACTION_UP] = 0.1
                q[s, ACTION_DOWN] = 0.1
                
        self.q_table = q
        self.model_loaded = True
        self.model_path = "demo_synthetic_q_table"
        return q

    def get_action(
        self,
        ball_x: float,
        ball_y: float,
        ball_vx: float,
        ball_vy: float,
        paddle_x: float = 770.0,
        paddle_y: float = 250.0,
        player_paddle_y: Optional[float] = None,
        field_width: float = 800.0,
        field_height: float = 500.0
    ) -> Tuple[int, Dict[str, Any]]:
        """
        ========================================================================
        EMBEDDING POINT #3: POLICY ACTION INFERENCE
        ========================================================================
        Calculates action to take according to current operating mode.
        Returns:
          - action: int (0: STAY, 1: UP, 2: DOWN)
          - debug_info: dict with discretization details and Q-values
        """
        # 1. Obtain discrete state and diagnostic technical info
        state_id, state_info = self.discretizer.discretize(
            ball_x=ball_x,
            ball_y=ball_y,
            ball_vx=ball_vx,
            ball_vy=ball_vy,
            paddle_x=paddle_x,
            paddle_y=paddle_y,
            field_width=field_width,
            field_height=field_height
        )

        q_values = [0.0, 0.0, 0.0]
        
        # 2. Action selection depending on mode
        if self.mode in ["dqn_noblind", "dqn_5d"]:
            # -----------------------------------------------------------------
            # 5D Tactical DQN Inference (No-Blind with Opponent Position)
            # -----------------------------------------------------------------
            max_dx = FIELD_WIDTH - PADDLE_OFFSET_X
            max_dy = FIELD_HEIGHT - PADDLE_HEIGHT / 2.0 - BALL_SIZE / 2.0
            max_v = MAX_BALL_SPEED

            dx = abs(paddle_x - ball_x)
            dx_norm = float(np.clip(dx / max_dx, 0.0, 1.0))
            dy = ball_y - paddle_y
            dy_norm = float(np.clip(dy / max_dy, -1.0, 1.0))

            vx_eff = ball_vx if paddle_x >= (FIELD_WIDTH / 2.0) else -ball_vx
            vx_norm = float(np.clip(vx_eff / max_v, -1.0, 1.0))
            vy_norm = float(np.clip(ball_vy / max_v, -1.0, 1.0))

            opp_y = player_paddle_y if player_paddle_y is not None else (field_height / 2.0)
            opp_y_norm = float(np.clip((opp_y - field_height / 2.0) / (field_height / 2.0), -1.0, 1.0))

            continuous_state_5d = np.array([[dx_norm, dy_norm, vx_norm, vy_norm, opp_y_norm]], dtype=np.float32)

            if self.onnx_session_5d is None:
                self.load_onnx_model("models/dqn_noblind.onnx")

            if self.onnx_session_5d is not None:
                input_name = self.onnx_session_5d.get_inputs()[0].name
                q_out = self.onnx_session_5d.run(None, {input_name: continuous_state_5d})[0]
                q_values = q_out[0].tolist()
                action = int(np.argmax(q_values))
            else:
                # Fallback to 4D session if 5D weights have not yet been trained
                sess_4d = self.onnx_session_4d or self.onnx_session
                if sess_4d is not None:
                    input_name = sess_4d.get_inputs()[0].name
                    q_out = sess_4d.run(None, {input_name: continuous_state_5d[:, :4]})[0]
                    q_values = q_out[0].tolist()
                    action = int(np.argmax(q_values))
                else:
                    action = int(np.random.choice([ACTION_STAY, ACTION_UP, ACTION_DOWN]))

        elif self.mode == "dqn" and (self.onnx_session_4d is not None or self.onnx_session is not None or self.dqn_model is not None):
            # -----------------------------------------------------------------
            # Continuous DQN Neural Network Inference (ONNX / PyTorch)
            # -----------------------------------------------------------------
            max_dx = FIELD_WIDTH - PADDLE_OFFSET_X
            max_dy = FIELD_HEIGHT - PADDLE_HEIGHT / 2.0 - BALL_SIZE / 2.0
            max_v = MAX_BALL_SPEED

            dx = abs(paddle_x - ball_x)
            dx_norm = float(np.clip(dx / max_dx, 0.0, 1.0))
            dy = ball_y - paddle_y
            dy_norm = float(np.clip(dy / max_dy, -1.0, 1.0))

            # Symmetric support (P1 left or P2 right paddle):
            vx_eff = ball_vx if paddle_x >= (FIELD_WIDTH / 2.0) else -ball_vx
            vx_norm = float(np.clip(vx_eff / max_v, -1.0, 1.0))
            vy_norm = float(np.clip(ball_vy / max_v, -1.0, 1.0))

            continuous_state = np.array([[dx_norm, dy_norm, vx_norm, vy_norm]], dtype=np.float32)

            sess_4d = self.onnx_session_4d or self.onnx_session
            if sess_4d is not None:
                # Ultrafast optimized inference with ONNX Runtime
                input_name = sess_4d.get_inputs()[0].name
                q_out = sess_4d.run(None, {input_name: continuous_state})[0]
                q_values = q_out[0].tolist()
                action = int(np.argmax(q_values))
            else:
                # Fallback: PyTorch inference
                with torch.no_grad():
                    st_tensor = torch.tensor(continuous_state, dtype=torch.float32, device=self.device)
                    q_out = self.dqn_model(st_tensor)
                    q_values = q_out.squeeze(0).cpu().tolist()
                    action = int(q_out.argmax(dim=1).item())

        elif self.mode == "q_learning":
            # -----------------------------------------------------------------
            # Tabular Q-Learning Inference
            # -----------------------------------------------------------------
            if self.q_table is None:
                self.load_q_table("models/q_table.npy")

            if self.q_table is not None:
                idx = state_id if state_id < len(self.q_table) else (state_id // 3 if (state_id // 3) < len(self.q_table) else 0)
                q_values = self.q_table[idx].tolist()
                action = int(np.argmax(self.q_table[idx]))
            else:
                action = int(np.random.choice([ACTION_STAY, ACTION_UP, ACTION_DOWN]))

        elif self.mode == "heuristic":
            # Simple direct tracking heuristic
            dy = ball_y - paddle_y
            if abs(dy) < 15:
                action = ACTION_STAY
            elif dy < 0:
                action = ACTION_UP
            else:
                action = ACTION_DOWN
        else:
            # -----------------------------------------------------------------
            # Random Exploration Mode
            # -----------------------------------------------------------------
            action = int(np.random.choice([ACTION_STAY, ACTION_UP, ACTION_DOWN]))

        debug_info = {
            "mode": self.mode,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "state_id": state_id,
            "q_values": q_values,
            "model_loaded": self.model_loaded,
            "state_info": state_info
        }

        return action, debug_info
