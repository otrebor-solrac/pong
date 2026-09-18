# 2. Physics engine implementation in Rust

## 2.1 Unified architecture and design motivation

A critical challenge in reinforcement learning web applications is **simulation parity**: discrepancies between the physics used during offline training in Python and the physics rendered in the user browser create perceptual mismatches and invalidate learned policy actions.

To eliminate simulation divergence, the physics engine is implemented as a single, unified codebase in **Rust** (`rust/pong_physics`), compiled into two distinct execution artifacts:
1. **Native dynamic library (`libpong_physics.so`)**: Linked directly to the Python runtime via C-ABI and `ctypes` for accelerated simulation (up to 120,000 steps per second).
2. **WebAssembly binary (`.wasm`)**: Compiled using `wasm-pack` with `wasm-bindgen` bindings, executing directly inside the client browser at 60 FPS with zero network round-trip latency.

---

## 2.2 Memory layout and core data structures

To facilitate zero-copy pointer exchanges across the C Foreign Function Interface (FFI), Rust data structures are explicitly declared with `#[repr(C)]`, guaranteeing deterministic C-compatible struct alignment and field ordering.

### 2.2.1 State representation (PongState)
The `PongState` struct encapsulates the continuous spatial and kinematic properties of the environment:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct PongState {
    pub ball_x: f32,
    pub ball_y: f32,
    pub ball_vx: f32,
    pub ball_vy: f32,
    pub p1_y: f32,
    pub p2_y: f32,
    pub p1_vy: f32,
    pub p2_vy: f32,
}
```

- Field alignments are 32-bit single-precision floats (`f32`), resulting in a compact 32-byte memory footprint per state.
- Supports stack allocation and pass-by-reference without heap allocation overhead.

### 2.2.2 Discrete step telemetry (StepOutcome)
The `StepOutcome` struct records discrete collision events and boundary breaches occurring during the integration step:

```rust
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "wasm", derive(serde::Serialize, serde::Deserialize))]
pub struct StepOutcome {
    pub hit_p1: bool,
    pub hit_p2: bool,
    pub goal_p1: bool,
    pub goal_p2: bool,
    pub hit_offset_p1: f32,
    pub hit_offset_p2: f32,
    pub wall_bounce: bool,
}
```

---

## 2.3 Deterministic and high-performance pseudo-random generation

Physics simulations requiring domain randomization must avoid the performance cost of OS-level entropy calls (`/dev/urandom`). 

The engine uses `rand::rngs::SmallRng`, a fast, non-cryptographic pseudo-random number generator backed by a 64-bit seed based on the golden ratio constant:

```rust
pub struct PongPhysics {
    rng: SmallRng,
}

impl PongPhysics {
    pub fn new() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(0x9E3779B97F4A7C15),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    #[inline]
    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        self.rng.gen_range(min..=max)
    }
}
```

This structure provides deterministic reproduction of identical trajectories given matching seeds, which is essential for unit testing and failure scenario regression.

---

## 2.4 Continuous step and collision integration

### 2.4.1 Substep integration loop (substep)
The core discrete micro-step method executes linear translation, boundary checks, and Axis-Aligned Bounding Box (AABB) intersection tests:

```rust
pub fn substep(&mut self, state: &mut PongState, dt_scale: f32) -> StepOutcome {
    let prev_x = state.ball_x;
    let half_ball = BALL_SIZE / 2.0;
    let half_paddle = PADDLE_HEIGHT / 2.0;
    let mut outcome = StepOutcome::default();

    // 1. Position update along continuous trajectory
    state.ball_x += state.ball_vx * dt_scale;
    state.ball_y += state.ball_vy * dt_scale;

    // 2. Specular reflection on arena walls
    outcome.wall_bounce = self.resolve_wall_bounce(state);

    // 3. Collision check against Player 1 (Left paddle)
    if state.ball_vx < 0.0
        && state.ball_x - half_ball <= P1_FRONT_X
        && prev_x + half_ball >= P1_BACK_X
        && state.ball_y >= state.p1_y - half_paddle - half_ball
        && state.ball_y <= state.p1_y + half_paddle + half_ball
    {
        state.ball_x = P1_FRONT_X + half_ball + 1.0;
        outcome.hit_p1 = true;
        outcome.hit_offset_p1 = self.resolve_paddle_bounce(
            &mut state.ball_vx,
            &mut state.ball_vy,
            state.p1_y,
            state.p1_vy,
            state.ball_y,
            true,
        );
    }

    // 4. Collision check against Player 2 (Right paddle / Agent)
    if state.ball_vx > 0.0
        && state.ball_x + half_ball >= P2_FRONT_X
        && prev_x - half_ball <= P2_BACK_X
        && state.ball_y >= state.p2_y - half_paddle - half_ball
        && state.ball_y <= state.p2_y + half_paddle + half_ball
    {
        state.ball_x = P2_FRONT_X - half_ball - 1.0;
        outcome.hit_p2 = true;
        outcome.hit_offset_p2 = self.resolve_paddle_bounce(
            &mut state.ball_vx,
            &mut state.ball_vy,
            state.p2_y,
            state.p2_vy,
            state.ball_y,
            false,
        );
    }

    // 5. Goal detection
    if state.ball_x < 0.0 {
        outcome.goal_p2 = true;
    } else if state.ball_x > FIELD_WIDTH {
        outcome.goal_p1 = true;
    }

    outcome
}
```

### 2.4.2 Deflection and momentum injection (resolve_paddle_bounce)
When a collision is registered, the deflection angle, tangential friction, and smash kinetic transfer are resolved in a unified routine:

```rust
pub fn resolve_paddle_bounce(
    &mut self,
    ball_vx: &mut f32,
    ball_vy: &mut f32,
    paddle_y: f32,
    paddle_vy: f32,
    ball_y: f32,
    is_left: bool,
) -> f32 {
    let half_h = PADDLE_HEIGHT / 2.0;
    let hit_offset = ((ball_y - paddle_y) / half_h).clamp(-1.0, 1.0);

    // Kinetic energy transfer from paddle stroke (smash effect)
    let kinetic_transfer = paddle_vy.abs() * 0.20;

    // Restitution multiplier and scalar speed update
    let speed_factor = self.rand_range(1.02, 1.08);
    let raw_speed = ball_vx.hypot(*ball_vy) * speed_factor + kinetic_transfer;
    let current_speed = raw_speed.clamp(INITIAL_BALL_SPEED, MAX_BALL_SPEED);

    // Angular deflection and tangential friction impulse
    let base_angle = hit_offset * 0.75;
    let base_vy = current_speed * base_angle.sin();
    let impulse_vy = paddle_vy * PADDLE_FRICTION;
    let new_angle = (base_vy + impulse_vy).atan2((current_speed * base_angle.cos()).abs());

    // Domain randomization jitter
    let jitter = self.rand_range(-0.06, 0.06);
    let clamped_angle = (new_angle + jitter).clamp(-0.95, 0.95);

    // Directional velocity assignment
    let bounce_vx = (current_speed * clamped_angle.cos()).abs();
    *ball_vx = if is_left { bounce_vx } else { -bounce_vx };
    *ball_vy = current_speed * clamped_angle.sin();

    hit_offset
}
```

---

## 2.5 C-ABI foreign function interface (FFI) for Python

The Rust crate exports symbols using `extern "C"` with unmangled symbol names (`#[no_mangle]`), providing a stable binary interface that Python can access via standard `ctypes`.

A static global instance wrapped in a `std::sync::Mutex` manages state synchronization:

```rust
use std::sync::Mutex;
static GLOBAL_ENGINE: Mutex<Option<PongPhysics>> = Mutex::new(None);

fn with_global_engine<R>(f: impl FnOnce(&mut PongPhysics) -> R) -> R {
    let mut lock = GLOBAL_ENGINE.lock().unwrap();
    if lock.is_none() {
        *lock = Some(PongPhysics::new());
    }
    f(lock.as_mut().unwrap())
}

#[no_mangle]
pub extern "C" fn pong_c_step(state: *mut PongState, sub_steps: u32) -> StepOutcome {
    if state.is_null() {
        return StepOutcome::default();
    }
    with_global_engine(|engine| unsafe { engine.step(&mut *state, sub_steps, 1.0) })
}
```

### 2.5.1 Python ctypes integration (rust_bridge.py)
On the Python side, matching structures and function signatures are bound at runtime:

```python
import ctypes

class PongStateC(ctypes.Structure):
    _fields_ = [
        ("ball_x", ctypes.c_float),
        ("ball_y", ctypes.c_float),
        ("ball_vx", ctypes.c_float),
        ("ball_vy", ctypes.c_float),
        ("p1_y", ctypes.c_float),
        ("p2_y", ctypes.c_float),
        ("p1_vy", ctypes.c_float),
        ("p2_vy", ctypes.c_float),
    ]

lib = ctypes.CDLL("/workspace/rust/pong_physics/target/release/libpong_physics.so")
lib.pong_c_step.argtypes = [ctypes.POINTER(PongStateC), ctypes.c_uint32]
lib.pong_c_step.restype = StepOutcomeC
```

If the compiled binary is not found, the Python runtime gracefully falls back to a native Python simulation implementation, ensuring operational resilience during initial setup.

---

## 2.6 WebAssembly (WASM) bindings for the React frontend

When the `wasm` Cargo feature is enabled, the module exposes a JavaScript-compatible class `WasmPongEngine`:

```rust
#[cfg(feature = "wasm")]
pub mod wasm_api {
    use super::*;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    pub struct WasmPongEngine {
        inner: PongPhysics,
    }

    #[wasm_bindgen]
    impl WasmPongEngine {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Self {
            Self {
                inner: PongPhysics::new(),
            }
        }

        #[wasm_bindgen]
        pub fn step(
            &mut self,
            ball_x: f32, ball_y: f32, ball_vx: f32, ball_vy: f32,
            p1_y: f32, p2_y: f32, p1_vy: f32, p2_vy: f32,
            sub_steps: u32,
            speed_multiplier: f32,
        ) -> Result<JsValue, JsValue> {
            let mut state = PongState {
                ball_x, ball_y, ball_vx, ball_vy,
                p1_y, p2_y, p1_vy, p2_vy,
            };

            let outcome = self.inner.step(&mut state, sub_steps, speed_multiplier);
            
            // Serialize outcome to JavaScript object via serde-wasm-bindgen
            serde_wasm_bindgen::to_value(&StepResponse { ... })
                .map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}
```

### 2.6.1 Frontend consumption in React (GameCanvas.jsx)
In the React frontend, the WebAssembly module is loaded asynchronously on component mount:

```javascript
import init, { WasmPongEngine } from "../wasm/pong_physics.js";

await init();
const engine = new WasmPongEngine();
const result = engine.step(
  ball.x, ball.y, ball.vx, ball.vy,
  p1.y, p2.y, p1.vy, p2.vy,
  subSteps, speedMultiplier
);
```

This enables client-side rendering with sub-millisecond execution times, delivering identical physics to the training environment without network dependency.

---

## 2.7 Build and compilation pipeline

The build workflow is automated via the repository `Makefile`:

- **Compile native Rust library (.so)**:
  ```bash
  make build-rust
  # cargo build --release inside the Rust container
  ```

- **Compile WebAssembly package (.wasm)**:
  ```bash
  make build-wasm
  # wasm-pack build --target web --out-dir frontend/src/wasm -- --features wasm
  ```

- **Compile all targets**:
  ```bash
  make build-all
  ```
