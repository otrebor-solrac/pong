# Technical documentation: Pong RL Studio

Welcome to the technical and mathematical documentation of **Pong RL Studio**, a high-performance reinforcement learning experimentation platform built on top of a continuous 2D Pong physics simulation.

This documentation details the complete stack, from analytical mechanics and continuous collision detection to the Rust physics core, tabular and deep reinforcement learning formulations, failure clinic targeted replay, and full-stack deployment.

---

## Table of contents

1. [Physical and mathematical foundations](01_physical_and_mathematical_foundations.md)
   - Continuous 2D kinematics and physical boundary constraints.
   - The tunneling problem and continuous collision detection (CCD).
   - Surface bounce dynamics, angular deflection, and paddle friction.
   - Kinetic momentum transfer (smash effect) and domain randomization.
   - Mathematical formulation of reward shaping.

2. [Physics engine implementation in Rust](02_physics_engine_implementation_in_rust.md)
   - Core data structures.
   - Adaptive continuous substep algorithm and deterministic integration.
   - Lock-free pseudo-random generation for thread safety.
   - Interoperability bridges: C-ABI dynamic library for Python (`ctypes`) and WebAssembly for React.

3. [Classical reinforcement learning: tabular Q-learning](03_tabular_q_learning.md)
   - Markov Decision Process (MDP) of the Pong environment.
   - Continuous state space discretization and action space.
   - Bellman optimality equation for off-policy learning.
   - Convergence properties and tabular limitations.

4. [Deep reinforcement learning: deep Q-networks (DQN)](04_deep_q_networks_dqn.md)
   - Multi-layer perceptron (MLP) architecture and input normalization.
   - Stabilization mechanisms: experience replay buffer and target networks.
   - Optimization scheme and epsilon-greedy exploration schedule.
   - Progressive curriculum learning against a heuristic opponent.
   - Export pipeline and cross-platform ONNX inference.

5. [Failure diagnosis and targeted replay (failure clinic)](05_failure_diagnosis_and_clinic.md)
   - Empirical failure diagnosis: Markovian rebound blindness.
   - Failure snapshot recording and persistence (`failed_shots.json`).
   - Targeted counterfactual exploration and micro-training.
   - Anchor gameplay insertion against catastrophic forgetting.
   - Regression-free model saving protocol.

6. [System architecture and full-stack integration](06_system_architecture_and_fullstack.md)
   - Containerized environment with Docker and GPU acceleration.
   - FastAPI backend services and telemetry streaming.
   - React frontend with dual-mode HTML5 Canvas rendering (client WASM and server API).

7. [Experimental results, metrics, and conclusions](07_experimental_results_and_conclusions.md)
   - Quantitative benchmarks: tabular Q-learning vs. baseline DQN vs. post-clinic DQN.
   - Rally length distribution metrics (P25, median, P75, IQR) and win rates.
   - Conclusions and future research directions.
