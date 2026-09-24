# 5. Failure diagnosis and targeted replay (failure clinic)

## 5.1 Empirical diagnosis: Markovian rebound blindness

Reinforcement learning agents operating under continuous MDPs often suffer from structural failure modes that are masked by aggregate performance metrics. In Pong, an agent may achieve high average rally counts while consistently failing on specific high-angle shots.

Detailed analysis of recorded trajectories reveals the primary root cause: **Markovian rebound blindness**.

### 5.1.1 Root cause analysis
1. **Absence of temporal recurrence or frame stacking**:
   The policy network receives an instantaneous state snapshot $\mathbf{s} = [dx, dy, vx, vy]$. Without recurrent hidden state memory (e.g., LSTM/GRU) or multi-frame stacking, the network cannot infer higher-order trajectory curvature or wall contact timing.
2. **Greedy spatial tracking bias**:
   When the ball travels toward a wall at high vertical velocity ($|v_y| > 6.0$), the immediate vertical offset $\Delta y = y_{\text{ball}} - y_{\text{paddle}}$ points toward the current ball position. A reactive policy moves the paddle directly toward the ball's current altitude.
3. **Wall reflection inversion**:
   Upon contacting the wall ($y = 0$ or $y = H$), the vertical velocity abruptly flips sign: $v_y \leftarrow -v_y$. The ball then reflects across the arena toward the opposite edge. Because the agent previously moved in the wrong direction, it lacks sufficient paddle speed ($10.0\text{ px/frame}$) to reverse course and intercept the ball before it crosses the goal line.

---

## 5.2 Structured failure persistence (failed_shots.jsonl)

To facilitate systematic diagnostics and counterfactual training, the environment records state snapshots whenever the agent concedes an unforced goal:

```json
{"id": 1, "agent_mode": "dqn", "agent_side": "p1", "rally_hits": 4, "peak_speed": 10.81, "shot_origin": {"ball_x": 420.5, "ball_y": 485.2, "ball_vx": 8.4, "ball_vy": 6.8, "paddle_y": 180.0}, "miss_impact": {"ball_x": 30.0, "ball_y": 120.0, "paddle_y": 250.0}}
```

By persisting these failure scenarios to `data/failed_shots.jsonl` via atomic append operations, the training system converts random runtime failures into a deterministic, reproducible diagnostic benchmark.

---

## 5.3 Counterfactual exploration and targeted micro-training

The **failure clinic** (`DQNTrainer.train_failures`) loads the recorded failure dataset and executes targeted counterfactual optimization on each missed trajectory.

### 5.3.1 Guided exploration schedule
For each failure scenario, the environment is initialized directly to the failure state via `env.reset_to_shot(shot)`. The agent is granted up to $K = 10$ attempts per shot, with guided exploration noise decaying per attempt:

$$\epsilon(k) = \max\left( 0.05, \; 0.35 \cdot \left( 1.0 - \frac{k - 1}{K} \right) \right)$$

Higher initial noise forces the policy to test alternate motor trajectories away from its ingrained, failing greedy path.

### 5.3.2 Redemption reward bonus
When the agent discovers a counterfactual motor sequence that successfully intercepts the ball, an immediate redemption bonus is awarded:

$$R_{\text{redemption}} = R(s, a, s') + 5.0$$

Upon a successful interception, 4 additional gradient descent updates are applied to consolidate the discovery into the network weights.

---

## 5.4 Mitigating catastrophic forgetting via anchor gameplay

A well-known vulnerability in targeted fine-tuning is **catastrophic forgetting**: adjusting network weights exclusively on specialized edge cases distorts the global policy manifold, causing the agent to lose its baseline defensive ability on standard shots.

To stabilize policy geometry, the failure clinic implements **anchor gameplay**:

1. **Replay buffer pre-population**:
   Prior to executing failure scenario replays, the agent plays $N_{\text{anchor}} = 30$ full-court baseline games against the heuristic bot.
2. **Experience interweaving**:
   Transitions from these anchor episodes fill the experience replay buffer $\mathcal{D}$ with hundreds of diverse, standard rallies.
3. **Joint mini-batch updates**:
   During failure clinic training, mini-batches sampled from $\mathcal{D}$ mix clinic transitions with anchor transitions:
   $$\mathcal{B} \sim \alpha \mathcal{D}_{\text{clinic}} + (1 - \alpha) \mathcal{D}_{\text{anchor}}$$

This regularizes gradient updates and preserves the global value function landscape across all court positions.

---

## 5.5 Regression-free model preservation protocol

To guarantee that clinic training cannot degrade production model quality, the trainer enforces a dual-phase validation protocol:

```
[Start Failure Clinic]
        │
        ▼
[Pre-Clinic Baseline Evaluation (40 episodes)] ──> Record Baseline Score S_base
        │
        ▼
[Execute Targeted Failure Replays + Anchor Buffer]
        │
        ▼
[Phase 1 Evaluation (40 episodes greedy)]
        │
   S_eval > S_base?
     ├── NO  ──> [Discard Clinic Weights] ──> Restore S_base Checkpoint
     └── YES ──> [Phase 2 Confirmation (100 episodes greedy)]
                     │
               S_conf > S_base?
                 ├── NO  ──> [Discard Clinic Weights] ──> Restore S_base
                 └── YES ──> [Commit Checkpoint] ──> Overwrite models/dqn_pong.pth & .onnx
```

This strict guardrail ensures that only interventions yielding a statistically confirmed global improvement are merged into production checkpoints.
