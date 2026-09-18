# 6. System architecture and full-stack integration

## 6.1 Containerized deployment architecture

The entire Pong RL Studio stack is packaged into a unified Docker container environment managed by Docker Compose. This ensures environment reproducibility across host machines and operating systems without host dependency pollution.

```
+-------------------------------------------------------------------------------+
|                            Host Machine (Linux / Windows)                     |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        Docker Container (pong_rl_studio)              |   |
|   |                                                                       |   |
|   |   +---------------------+   +-------------------+   +-------------+   |   |
|   |   |   FastAPI Backend   |   |   React Frontend  |   |  Rust Core  |   |   |
|   |   |      (Port 8000)    |   |     (Port 5173)   |   |  (Compiled) |   |   |
|   |   +----------+----------+   +---------+---------+   +------+------+   |   |
|   |              |                        |                    |          |   |
|   |              +---- Shared Volume -----+                    |          |   |
|   |                     (/workspace)                           |          |   |
|   |                          |                                 |          |   |
|   |              libpong_physics.so <--------------------------+          |   |
|   |              pong_physics.wasm  <--------------------------+          |   |
|   +-----------------------------------------------------------------------+   |
|                                      |                                        |
|   Ports:                             | GPU Acceleration:                      |
|     - 8000: FastAPI Telemetry / API  |   - NVIDIA Container Toolkit           |
|     - 5173: React Interactive UI     |   - CUDA Runtime Passthrough           |
+-------------------------------------------------------------------------------+
```

### 6.1.1 Hardware acceleration passthrough
The `docker-compose.yml` configuration specifies direct NVIDIA GPU reservations:

```yaml
services:
  pong:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: pong_rl_studio
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [ gpu ]
    volumes:
      - .:/workspace
    ports:
      - "8000:8000"   # FastAPI Backend
      - "5173:5173"   # React Frontend
```

When an NVIDIA GPU is available, PyTorch leverages CUDA tensors for high-throughput mini-batch gradient updates; otherwise, the runtime seamlessly falls back to optimized CPU execution.

---

## 6.2 Backend service layer (FastAPI)

The backend (`backend/main.py`) provides asynchronous HTTP services:

1. **System health and telemetry (`GET /api/status`)**:
   Reports active reinforcement learning model versions (ONNX and PyTorch checkpoints), device status (CUDA vs. CPU), historical best scores, and training progress.
2. **Binary Q-matrix distribution (`GET /api/q_table`)**:
   Streams the 69,120-byte raw float32 buffer of the trained Tabular Q-matrix ($5,760 \times 3$) to the frontend for local, zero-latency client-side evaluation.
3. **Failure dataset access (`GET /api/failures`)**:
   Exposes serialized failure snapshots from `data/failed_shots.json` to the frontend for interactive replay visualization.
4. **Remote policy prediction (`POST /api/predict`)**:
   Executes continuous Deep Q-Network (ONNX Runtime / PyTorch) and Tabular inference on incoming court states. Automatically checks disk timestamps (`mtime`) on every request to hot-reload newly trained checkpoints seamlessly.

---

## 6.3 Frontend architecture (React + HTML5 Canvas)

The client application is built with React and Vite (`frontend/`), delivering responsive real-time interaction.

### 6.3.1 Dual-mode physics and zero-latency inference
The game canvas couples native compilation with client-side inference:
1. **Client WebAssembly physics (Rust WASM)**:
   The browser initializes the compiled WebAssembly module (`frontend/src/wasm/pong_physics.js`). Every 60 FPS animation frame invokes the Rust physics engine directly in WebAssembly memory, eliminating simulation lag.
2. **Zero-latency local Q-learning inference**:
   Rather than incurring 15–40 ms of network round-trip delay per frame via HTTP, the browser pre-fetches the 69 KB Q-matrix buffer once into a typed `Float32Array`. State discretization and argmax action selection execute synchronously in JavaScript in $< 0.001\text{ ms}$, ensuring the paddle reacts to the exact current frame with zero lag.
3. **Remote DQN inference**:
   For Deep Q-Networks, the canvas streams continuous states to FastAPI and applies hot-reloaded ONNX models in real time.

### 6.3.2 Interactive dashboard capabilities
- **Algorithm switching**: Seamless toggle between Tabular Q-Learning, Deep Q-Network (DQN), and heuristic bot baselines.
- **Opponent control**: Switch between solo wall-bounce mode, heuristic tracking bot, or human keyboard controls (`W`/`S` or arrow keys).
- **Physical parameter tuning**: Real-time adjustment of simulation speed multipliers, paddle speeds, and visual debug overlays (trajectory vectors, collider boxes, and hit zones).
- **Failure clinic launcher**: Trigger targeted replay sessions directly from the user interface.
