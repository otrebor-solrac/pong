# Pong RL Studio

An interactive reinforcement learning environment and research suite for Pong. The platform features tabular Q-learning, deep Q-networks with n-step experience replay, a unified high-performance physics engine written in Rust compiled to both WebAssembly and native shared libraries, accelerated inference via ONNX Runtime, and a responsive web interface built with React.

## Key features

- Interactive dual-player simulation: play human versus artificial intelligence, artificial intelligence versus artificial intelligence, or heuristic rules versus trained neural agents in real time at 60 frames per second.
- Unified Rust physics engine:
  - Single source of truth for court physics implemented in Rust.
  - Compiled to WebAssembly for client-side browser execution and to a native C-ABI shared library for Python headless training.
  - Continuous swept bounding box collision detection with dynamic sub-stepping to eliminate ball tunneling at high velocities.
  - Realistic paddle deflection dynamics with transverse paddle friction, stochastic angular perturbation within plus or minus 3.5 degrees, and progressive post-impact acceleration.
- Reinforcement learning engines:
  - Tabular Q-learning: state space discretization mapping relative ball-to-paddle positions and directional velocities into deterministic action policies.
  - Deep Q-networks: deep neural network trained with n-step experience replay, decoupled target networks, smooth L1 Huber loss, and direct export to ONNX format.
- Curriculum learning support:
  - Practice wall mode: left-wall bounce environment designed to maximize hit density during early exploration stages.
  - Heuristic opponent mode: reactive adversary with adjustable tracking speed from 30 percent to 95 percent to train offensive shot placement.
- Containerized workflow: complete Docker and Docker Compose environment with optional NVIDIA GPU acceleration support.

## Performance benchmarks

| Agent | State representation | Average hits | Max rally | Inference latency |
| :--- | :--- | :---: | :---: | :---: |
| Tabular Q-learning | Discrete (1,920 states) | 12.26 | 15 | less than 0.05 ms |
| Deep Q-network (ONNX) | Continuous normalized vector | 9.97 | 12 | approx 0.8 ms |
| Reactive heuristic | Rule-based center tracking | 4.80 | 7 | less than 0.01 ms |

## Theoretical background

### Tabular Q-learning update rule

The tabular agent updates its action-value function via temporal difference learning:

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]$$

Key hyperparameters:
- Learning rate: 0.15
- Discount factor: 0.98
- Exploration schedule: epsilon-greedy policy decaying geometrically toward 0.02

### Deep Q-networks with n-step returns

To reduce bias during early training, the experience buffer computes multi-step accumulated returns:

$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k+1}$$

The network parameters are optimized by minimizing the Huber loss between evaluation and target networks:

$$\mathcal{L}(\theta) = \mathbb{E} \left[ \text{Smooth}_{L1} \left( R_t^{(n)} + \gamma^n \max_{a'} Q_{\theta^-}(s_{t+n}, a') - Q_\theta(s_t, a_t) \right) \right]$$

## Physics architecture and anti-tunneling

At high ball velocities (exceeding 15 pixels per frame), discrete frame updates can cause the ball to pass through paddle colliders. The unified Rust engine eliminates this artifact through continuous collision math:

1. Dynamic trajectory subdivision:
   $$\text{sub\_steps} = \max\left(1, \; \left\lceil \frac{\sqrt{\Delta x^2 + \Delta y^2}}{4.0} \right\rceil\right)$$
2. Swept bounding box intersection testing at each sub-step.
3. Transverse friction transfer based on paddle linear velocity.
4. Bounded stochastic deflection jitter to prevent deterministic infinite loops.

## Quick start guide

### Prerequisites

- Docker and Docker Compose
- Optional: Git and GitHub CLI

### Running with Docker

Start both the backend API (FastAPI on port 8000) and the web interface (React on port 5173):

```bash
make up
```

Access points:
- Web interface: http://localhost:5173
- API documentation: http://localhost:8000/docs

### Compiling the Rust engine locally

To build the physics binaries directly outside of Docker:

```bash
# Build native shared library for Python backend
make build-rust

# Build WebAssembly package for React frontend
make build-wasm
```

## Model training

### Training tabular Q-learning

```bash
make train ARGS="--episodes 15000 --pretrained models/q_table.npy"
```

### Training deep Q-networks

```bash
make train-dqn ARGS="--episodes 25000 --export_onnx models/dqn_pong.onnx"
```

## Repository structure

```text
pong/
├── backend/
│   ├── constants.py        # Environment dimensions and physics constants
│   ├── rust_bridge.py      # Ctypes interface to native Rust physics library
│   ├── env.py              # Simulation environment for RL training
│   ├── buffer.py           # N-step experience replay buffer
│   ├── model.py            # Q-table agent and ONNX Runtime inference
│   ├── train.py            # Training pipeline and evaluation routines
│   └── main.py             # FastAPI REST endpoints and websocket feeds
├── frontend/
│   ├── src/
│   │   ├── wasm/           # Compiled WebAssembly modules and JavaScript bindings
│   │   ├── components/
│   │   │   ├── PongCanvas.jsx    # Canvas rendering driven by WebAssembly physics
│   │   │   ├── ControlsPanel.jsx # Gameplay mode and difficulty settings
│   │   │   └── QTableViewer.jsx  # Real-time Q-table policy heatmap
│   │   └── App.jsx
│   └── package.json
├── rust/
│   └── pong_physics/       # Unified physics implementation (C-ABI and WebAssembly)
├── models/
│   ├── q_table.npy         # Trained tabular weights
│   ├── dqn_pong.onnx       # Exported deep Q-network in ONNX format
│   └── dqn_pong.pth        # PyTorch checkpoint weights
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── start.sh
└── README.md
```

## License

Distributed under the MIT License.
