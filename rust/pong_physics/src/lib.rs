//! High-Performance Unified Physics Engine for Pong RL Studio
//! Single ground-truth physics implementation shared by:
//! 1. Python RL training simulation (via C-ABI / ctypes)
//! 2. React Vite frontend browser runtime (via WebAssembly / wasm-bindgen)

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// ==============================================================================
// Physical Field Constants (Synchronized across all modules)
// ==============================================================================
pub const FIELD_WIDTH: f32 = 800.0;
pub const FIELD_HEIGHT: f32 = 500.0;

pub const PADDLE_WIDTH: f32 = 12.0;
pub const PADDLE_HEIGHT: f32 = 86.0;
pub const PADDLE_OFFSET_X: f32 = 30.0;

pub const BALL_SIZE: f32 = 10.0;
pub const INITIAL_BALL_SPEED: f32 = 4.5;
pub const MAX_BALL_SPEED: f32 = 15.0;
pub const PADDLE_FRICTION: f32 = 0.35;

pub const P1_X: f32 = PADDLE_OFFSET_X;
pub const P1_FRONT_X: f32 = P1_X + PADDLE_WIDTH / 2.0;
pub const P1_BACK_X: f32 = P1_X - PADDLE_WIDTH / 2.0;

pub const P2_X: f32 = FIELD_WIDTH - PADDLE_OFFSET_X;
pub const P2_FRONT_X: f32 = P2_X - PADDLE_WIDTH / 2.0;
pub const P2_BACK_X: f32 = P2_X + PADDLE_WIDTH / 2.0;

/// Core state representation for physics computation
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

/// Result returned after executing simulation steps
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

/// Core deterministic & stochastic physics simulator
pub struct PongPhysics {
    rng: SmallRng,
}

impl Default for PongPhysics {
    fn default() -> Self {
        Self::new()
    }
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

    /// Resolve bounce against top and bottom walls with stochastic damping & angular jitter
    pub fn resolve_wall_bounce(&mut self, state: &mut PongState) -> bool {
        let half_ball = BALL_SIZE / 2.0;
        let mut bounced = false;

        // Top wall reflection
        if state.ball_y - half_ball <= 0.0 {
            state.ball_y = half_ball;
            state.ball_vy = state.ball_vy.abs();

            let speed = (state.ball_vx.hypot(state.ball_vy) * self.rand_range(0.97, 1.00))
                .max(INITIAL_BALL_SPEED);
            let current_angle = state.ball_vy.atan2(state.ball_vx);
            let new_angle = current_angle + self.rand_range(-0.035, 0.035);

            let sign_vx = if state.ball_vx >= 0.0 { 1.0 } else { -1.0 };
            state.ball_vx = sign_vx * (speed * new_angle.cos()).abs();
            state.ball_vy = (speed * new_angle.sin()).abs().max(0.5);
            bounced = true;
        }
        // Bottom wall reflection
        else if state.ball_y + half_ball >= FIELD_HEIGHT {
            state.ball_y = FIELD_HEIGHT - half_ball;
            state.ball_vy = -state.ball_vy.abs();

            let speed = (state.ball_vx.hypot(state.ball_vy) * self.rand_range(0.97, 1.00))
                .max(INITIAL_BALL_SPEED);
            let current_angle = state.ball_vy.atan2(state.ball_vx);
            let new_angle = current_angle + self.rand_range(-0.035, 0.035);

            let sign_vx = if state.ball_vx >= 0.0 { 1.0 } else { -1.0 };
            state.ball_vx = sign_vx * (speed * new_angle.cos()).abs();
            state.ball_vy = -(speed * new_angle.sin()).abs().max(0.5);
            bounced = true;
        }

        bounced
    }

    /// Resolve deflection against a paddle with friction and domain randomization
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

        // Accelerated rally dynamics: 1.02 to 1.08 speed multiplier per paddle hit (mean ~1.05)
        let speed_factor = self.rand_range(1.02, 1.08);
        let raw_speed = ball_vx.hypot(*ball_vy) * speed_factor;
        let current_speed = raw_speed.clamp(INITIAL_BALL_SPEED, MAX_BALL_SPEED);

        let base_angle = hit_offset * 0.75;
        let base_vy = current_speed * base_angle.sin();

        let impulse_vy = paddle_vy * PADDLE_FRICTION;
        let new_angle = (base_vy + impulse_vy).atan2((current_speed * base_angle.cos()).abs());

        let jitter = self.rand_range(-0.06, 0.06);
        let clamped_angle = (new_angle + jitter).clamp(-0.95, 0.95);

        let bounce_vx = (current_speed * clamped_angle.cos()).abs();
        *ball_vx = if is_left { bounce_vx } else { -bounce_vx };
        *ball_vy = current_speed * clamped_angle.sin();

        hit_offset
    }

    /// Execute a single discrete sub-step of physics
    pub fn substep(&mut self, state: &mut PongState, dt_scale: f32) -> StepOutcome {
        let prev_x = state.ball_x;
        let half_ball = BALL_SIZE / 2.0;
        let half_paddle = PADDLE_HEIGHT / 2.0;
        let mut outcome = StepOutcome::default();

        state.ball_x += state.ball_vx * dt_scale;
        state.ball_y += state.ball_vy * dt_scale;

        outcome.wall_bounce = self.resolve_wall_bounce(state);

        // Paddle 1 (Left) Collision Check
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

        // Paddle 2 (Right) Collision Check
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

        // Goal Checks
        if state.ball_x < 0.0 {
            outcome.goal_p2 = true; // P2 scores against P1
        } else if state.ball_x > FIELD_WIDTH {
            outcome.goal_p1 = true; // P1 scores against P2
        }

        outcome
    }

    /// Multi-substep loop with dynamic speed_multiplier
    pub fn step(&mut self, state: &mut PongState, sub_steps: u32, speed_multiplier: f32) -> StepOutcome {
        let sub_steps = sub_steps.max(1);
        let dt_scale = speed_multiplier / (sub_steps as f32);
        let mut final_outcome = StepOutcome::default();

        for _ in 0..sub_steps {
            let res = self.substep(state, dt_scale);
            if res.hit_p1 {
                final_outcome.hit_p1 = true;
                final_outcome.hit_offset_p1 = res.hit_offset_p1;
            }
            if res.hit_p2 {
                final_outcome.hit_p2 = true;
                final_outcome.hit_offset_p2 = res.hit_offset_p2;
            }
            if res.wall_bounce {
                final_outcome.wall_bounce = true;
            }
            if res.goal_p1 {
                final_outcome.goal_p1 = true;
                break;
            }
            if res.goal_p2 {
                final_outcome.goal_p2 = true;
                break;
            }
        }

        final_outcome
    }
}

// ==============================================================================
// C-ABI Exports (For direct, ultra-fast Python integration via ctypes)
// ==============================================================================
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

#[no_mangle]
pub extern "C" fn pong_c_resolve_paddle(
    ball_vx: *mut f32,
    ball_vy: *mut f32,
    paddle_y: f32,
    paddle_vy: f32,
    ball_y: f32,
    is_left: bool,
) -> f32 {
    if ball_vx.is_null() || ball_vy.is_null() {
        return 0.0;
    }
    with_global_engine(|engine| unsafe {
        engine.resolve_paddle_bounce(
            &mut *ball_vx,
            &mut *ball_vy,
            paddle_y,
            paddle_vy,
            ball_y,
            is_left,
        )
    })
}

// ==============================================================================
// WebAssembly (WASM) Bindings for Vite/React Frontend
// ==============================================================================
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

        /// Advances game physics by sub_steps with speed_multiplier and returns updated state + events
        #[wasm_bindgen]
        pub fn step(
            &mut self,
            ball_x: f32,
            ball_y: f32,
            ball_vx: f32,
            ball_vy: f32,
            p1_y: f32,
            p2_y: f32,
            p1_vy: f32,
            p2_vy: f32,
            sub_steps: u32,
            speed_multiplier: f32,
        ) -> Result<JsValue, JsValue> {
            let mut state = PongState {
                ball_x,
                ball_y,
                ball_vx,
                ball_vy,
                p1_y,
                p2_y,
                p1_vy,
                p2_vy,
            };

            let outcome = self.inner.step(&mut state, sub_steps, speed_multiplier);

            #[derive(serde::Serialize)]
            struct StepResponse {
                ball_x: f32,
                ball_y: f32,
                ball_vx: f32,
                ball_vy: f32,
                hit_p1: bool,
                hit_p2: bool,
                goal_p1: bool,
                goal_p2: bool,
                hit_offset_p1: f32,
                hit_offset_p2: f32,
                wall_bounce: bool,
            }

            let response = StepResponse {
                ball_x: state.ball_x,
                ball_y: state.ball_y,
                ball_vx: state.ball_vx,
                ball_vy: state.ball_vy,
                hit_p1: outcome.hit_p1,
                hit_p2: outcome.hit_p2,
                goal_p1: outcome.goal_p1,
                goal_p2: outcome.goal_p2,
                hit_offset_p1: outcome.hit_offset_p1,
                hit_offset_p2: outcome.hit_offset_p2,
                wall_bounce: outcome.wall_bounce,
            };

            serde_wasm_bindgen::to_value(&response).map_err(|e| JsValue::from_str(&e.to_string()))
        }
    }
}
