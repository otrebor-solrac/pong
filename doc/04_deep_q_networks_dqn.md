# 4. Deep reinforcement learning: deep Q-networks (DQN)

## 4.1 Transition from tabular to continuous deep function approximation

While tabular discretization partitions continuous space into arbitrary discrete bins, deep function approximation models the action-value function as a parameterized continuous mapping:

$$Q(s, a; \theta) \approx Q^*(s, a)$$

where $\theta \in \mathbb{R}^d$ denotes the vector of trainable neural network parameters. Deep Q-Networks (DQN) overcome the resolution limits and curse of dimensionality inherent to tabular Q-learning by generalizing across continuous spatial and kinematic states.

---

## 4.2 State representation and normalization

The policy network processes a 4-dimensional normalized continuous vector $\mathbf{s} \in [-1, 1]^4$:

$$\mathbf{s} = \begin{bmatrix}
\frac{x_{\text{paddle}} - x_{\text{ball}}}{W} \\
\frac{y_{\text{ball}} - y_{\text{paddle}}}{H} \\
\frac{v_{x,\text{ball}}}{v_{\text{max}}} \\
\frac{v_{y,\text{ball}}}{v_{\text{max}}}
\end{bmatrix}$$

where $W = 800.0$, $H = 500.0$, and $v_{\text{max}} = 15.0\text{ px/frame}$. 

This relative coordinate representation confers translational invariance: the network learns relative interception dynamics regardless of the absolute coordinate origin.

---

## 4.3 Neural network architecture

The Q-function approximator (`PongDQN`) is structured as a compact Multi-Layer Perceptron (MLP) designed for low-latency inference:

```
Input (4) 
  --> Linear(4, 64)   --> ReLU()
  --> Linear(64, 64)  --> ReLU()
  --> Linear(64, 3)   --> Output Q-values [Q(s, STAY), Q(s, UP), Q(s, DOWN)]
```

### 4.3.1 Parameter count and layer specifications
- **Hidden layer 1**: $4 \times 64 + 64 = 320$ parameters.
- **Hidden layer 2**: $64 \times 64 + 64 = 4,160$ parameters.
- **Output layer**: $64 \times 3 + 3 = 195$ parameters.
- **Total trainable parameters**: $4,675$ parameters ($18.7\text{ KB}$ in FP32).

The minimal parameter footprint guarantees sub-millisecond forward-pass execution on standard CPUs and web browser runtimes via WebAssembly or ONNX Runtime.

---

## 4.4 Stabilization mechanisms in deep Q-learning

Standard Q-learning with non-linear neural function approximators is inherently prone to divergence due to the "deadly triad": function approximation, bootstrapping, and off-policy training. To ensure numerical stability, three complementary mechanisms are implemented.

### 4.4.1 Experience replay buffer with n-step returns
Transitions are collected during interaction and stored in a FIFO cyclic buffer $\mathcal{D}$ with capacity $|\mathcal{D}| = 100,000$:

$$\tau_t = (s_t, a_t, R_{t:t+n}, s_{t+n}, d_{t+n})$$

To accelerate temporal credit propagation over extended rallies, an $n$-step return window ($n = 3$) accumulates discounted intermediate rewards:

$$R_{t:t+n} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1}$$

During each training step, a mini-batch of size $B = 64$ is sampled uniformly at random from $\mathcal{D}$:

$$\mathbb{E}_{(s, a, R, s', d) \sim \mathcal{U}(\mathcal{D})} [L(\theta)]$$

Random batch sampling breaks the temporal autocorrelation between consecutive frames, transforming non-stationary trajectory streams into independently and identically distributed (i.i.d.) observations.

### 4.4.2 Double DQN (DDQN) target decoupling
Classic DQN uses the same target network both to select and to evaluate actions in the Bellman target, inducing an upward maximization bias:

$$\mathbb{E}\left[ \max_a Q(s', a) \right] \ge \max_a \mathbb{E}\left[ Q(s', a) \right]$$

To eliminate this overestimation, Double DQN decouples action selection from action evaluation:
1. **Action selection**: The active online policy network chooses the optimal greedy action:
   $$a^* = \arg\max_{a \in \mathcal{A}} Q(s_{t+n}, a; \theta_{\text{online}})$$
2. **Action evaluation**: The target network evaluates the expected value of that chosen action:
   $$Y_t = R_{t:t+n} + \gamma^n Q(s_{t+n}, a^*; \theta_{\text{target}}) \cdot (1 - d_{t+n})$$

### 4.4.3 Polyak soft target updates
Rather than performing periodic hard weight replacements ($\theta_{\text{target}} \leftarrow \theta_{\text{online}}$ every $C$ steps), target weights track the online network continuously via exponential moving averaging (Polyak averaging):

$$\theta_{\text{target}} \leftarrow \tau \theta_{\text{online}} + (1 - \tau) \theta_{\text{target}}$$

with tracking rate $\tau = 0.005$. This continuous soft update produces smooth, monotonic evolution of the Bellman target surface.

### 4.4.4 Loss function and gradient clipping
The temporal difference loss is computed using the Smooth L1 (Huber) criterion:

$$L(\theta) = \frac{1}{B} \sum_{i=1}^B \ell_{\delta_i}, \quad \delta_i = Y_i - Q(s_i, a_i; \theta)$$

$$\ell_{\delta} = \begin{cases}
0.5 \delta^2 & \text{if } |\delta| < 1.0 \\
|\delta| - 0.5 & \text{otherwise}
\end{cases}$$

The Huber loss acts as quadratic L2 regression for minor residuals, transitioning to linear L1 penalization for large prediction errors. To prevent gradient explosion from sudden terminal failure penalties, gradients are clipped to unit norm:

$$\|\nabla_{\theta} L\|_2 \le 1.0$$

---

## 4.5 Curriculum learning against a heuristic opponent

Learning competitive Pong directly against a full-speed deterministic opponent causes premature exploration collapse: the agent misses all incoming balls before learning basic paddle positioning.

To establish stable learning trajectories, a progressive curriculum schedules the tracking speed ratio $\rho_{\text{opp}} \in [0.45, 0.90]$ of the heuristic opponent as training advances across episode $k \in [1, K]$:

$$\rho_{\text{opp}}(k) = 0.45 + 0.55 \cdot \left( \frac{k - 1}{K - 1} \right)$$

- **Early phase ($\rho_{\text{opp}} \approx 0.45$)**: The opponent paddle moves slowly ($4.5\text{ px/frame}$), allowing the agent to sustain initial returns and accumulate dense contact rewards.
- **Mid phase ($\rho_{\text{opp}} \approx 0.72$)**: The opponent returns sharp angles, forcing the agent to learn anticipatory positioning along the vertical axis.
- **Final phase ($\rho_{\text{opp}} \approx 1.00$)**: The opponent operates at maximum tracking fidelity ($10.0\text{ px/frame}$), demanding precise momentum transfers and trick shots to score goals.

---

## 4.6 Robust distributional evaluation and checkpointing

Because reinforcement learning policies can exhibit high variance across individual episodes, naive checkpointing based on maximum single-episode score often commits lucky outlier models to disk.

The evaluation routine assesses the policy across $N = 200$ pure greedy episodes ($\epsilon = 0$) and computes a robust quality metric based on quartile distribution:

$$\text{Score} = \frac{\bar{H} + \text{Med}(H) + P_{25}(H)}{3} - \frac{\text{IQR}(H)}{4} + 5.0 \cdot \bar{G}$$

where:
- $\bar{H}$ is the empirical mean rally hits.
- $\text{Med}(H)$ is the median rally length.
- $P_{25}(H)$ is the 25th percentile (representing the guaranteed defensive skill floor).
- $\text{IQR}(H) = P_{75}(H) - P_{25}(H)$ penalizes high variance and erratic defensive lapses.
- $\bar{G}$ is the mean goals scored per episode against the heuristic opponent.

**Standardized Environmental Evaluation**
To prevent metric distortion caused by the Curriculum Learning schedule, the evaluation routine explicitly locks the environmental opponent to its maximum capability ($\rho_{\text{opp}} = 1.00$) for the duration of the evaluation. This ensures that a baseline calculated at the beginning of a run is directly comparable to the final evaluation at episode $K$, yielding an absolute measure of the model's performance regardless of the curriculum's current progression state.

When the evaluation score against the maximum-speed opponent exceeds the historical baseline, the online network weights are committed to disk (`models/dqn_pong.pth`), and the computational graph is exported to ONNX format.

---

## 4.7 Cross-platform ONNX export pipeline

To decouple real-time inference from PyTorch, the trainer executes an atomic export after every confirmed record:

```python
dummy_input = torch.zeros(1, 4, dtype=torch.float32, device=self.device)
torch.onnx.export(
    self.policy_net,
    dummy_input,
    "models/dqn_pong.onnx",
    input_names=["state"],
    output_names=["q_values"],
    dynamic_axes={"state": {0: "batch_size"}, "q_values": {0: "batch_size"}},
    opset_version=14
)
```

The exported ONNX model allows direct deployment in:
1. **FastAPI backend**: High-throughput batched evaluations using ONNX Runtime CPU/CUDA.
2. **React browser UI**: Zero-install client inference via `onnxruntime-web` with WebAssembly and WebGL execution backends.
