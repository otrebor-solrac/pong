"""
Rust Physics Engine Bridge for Python
-------------------------------------
Loads the native compiled Rust cdylib (`libpong_physics.so`) via ctypes.
Provides microsecond-level physics execution for RL simulation loops
with seamless fallback to Python when not compiled.
"""

import os
import ctypes
from typing import Optional, Tuple

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

class StepOutcomeC(ctypes.Structure):
    _fields_ = [
        ("hit_p1", ctypes.c_bool),
        ("hit_p2", ctypes.c_bool),
        ("goal_p1", ctypes.c_bool),
        ("goal_p2", ctypes.c_bool),
        ("hit_offset_p1", ctypes.c_float),
        ("hit_offset_p2", ctypes.c_float),
        ("wall_bounce", ctypes.c_bool),
    ]

_RUST_LIB = None
_IS_INITIALIZED = False

def _find_lib() -> Optional[str]:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(base_dir, "rust", "pong_physics", "target", "release", "libpong_physics.so"),
        os.path.join(base_dir, "rust", "pong_physics", "target", "debug", "libpong_physics.so"),
        "/workspace/rust/pong_physics/target/release/libpong_physics.so",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

def get_rust_lib():
    global _RUST_LIB, _IS_INITIALIZED
    if _IS_INITIALIZED:
        return _RUST_LIB

    _IS_INITIALIZED = True
    lib_path = _find_lib()
    if lib_path:
        try:
            lib = ctypes.CDLL(lib_path)
            # Setup signatures
            lib.pong_c_step.argtypes = [ctypes.POINTER(PongStateC), ctypes.c_uint32]
            lib.pong_c_step.restype = StepOutcomeC

            lib.pong_c_resolve_paddle.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_bool,
            ]
            lib.pong_c_resolve_paddle.restype = ctypes.c_float

            _RUST_LIB = lib
            print(f"[RustEngine] Successfully loaded native physics core from {lib_path}")
        except Exception as e:
            print(f"[RustEngine] Notice: Failed to bind {lib_path}: {e}. Using pure Python fallback.")
            _RUST_LIB = None
    return _RUST_LIB

def is_rust_available() -> bool:
    return get_rust_lib() is not None

def rust_step(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    p1_y: float,
    p2_y: float,
    p1_vy: float = 0.0,
    p2_vy: float = 0.0,
    sub_steps: int = 1
) -> Tuple[PongStateC, StepOutcomeC]:
    lib = get_rust_lib()
    if not lib:
        raise RuntimeError("Rust library is not loaded.")

    state = PongStateC(
        ball_x=float(ball_x),
        ball_y=float(ball_y),
        ball_vx=float(ball_vx),
        ball_vy=float(ball_vy),
        p1_y=float(p1_y),
        p2_y=float(p2_y),
        p1_vy=float(p1_vy),
        p2_vy=float(p2_vy),
    )
    outcome = lib.pong_c_step(ctypes.byref(state), ctypes.c_uint32(sub_steps))
    return state, outcome
