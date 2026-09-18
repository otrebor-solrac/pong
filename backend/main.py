"""
FastAPI server for inference of Pong agent.
"""

import os
import json
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from constants import ACTION_NAMES
from model import PongQAgent


app = FastAPI(
    title="Pong RL FastAPI Server",
    description="API for inference of Pong agent using Q-Learning or DQN.",
    version="1.0.0"
)

# CORS configuration to allow connections from the React Vite client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In development, any origin is allowed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instance of the Agent (Tabular / DQN ONNX)

agent = PongQAgent(mode="dqn")

# Load both models if they exist for smooth switching in frontend
has_q = agent.load_q_table("models/q_table.npy")
has_onnx = agent.load_onnx_model("models/dqn_pong.onnx")

if not has_onnx and not has_q:
    agent.mode = "heuristic"
else:
    agent.mode = "dqn" if has_onnx else "q_learning"


# Pydantic Models / Schemas
class PongStateInput(BaseModel):
    ball_x: float = Field(..., description="Ball X position in pixels")
    ball_y: float = Field(..., description="Ball Y position in pixels")
    ball_vx: float = Field(..., description="Ball velocity X")
    ball_vy: float = Field(..., description="Ball velocity Y")
    ai_paddle_y: float = Field(..., description="AI paddle Y position (center of the paddle)")
    ai_paddle_x: float = Field(..., description="AI paddle X position (770 for P2, 30 for P1)")
    player_paddle_y: Optional[float] = Field(None, description="Player paddle Y position (center of the paddle)")
    field_width: float = Field(800.0, description="Field width in pixels")
    field_height: float = Field(500.0, description="Field height in pixels")
    mode: Optional[str] = Field(None, description="Optional inference mode: 'dqn', 'q_learning', 'heuristic', 'random'")


class ModeInput(BaseModel):
    mode: str = Field(..., description="Operation mode: 'random', 'q_learning', or 'heuristic'")


class LoadModelInput(BaseModel):
    filepath: str = Field(..., description="Path to the Q-table file (.npy or .pkl)")


class FailureRecordInput(BaseModel):
    agent_mode: str = Field(..., description="Agent mode: 'dqn' or 'q_learning'")
    agent_side: str = Field(..., description="Paddle side: 'p1' or 'p2'")
    rally_hits: int = Field(0, description="Rally hits count")
    peak_speed: float = Field(0.0, description="Peak speed during rally in px/f")
    shot_origin: Dict[str, Any] = Field(..., description="State snapshot when opponent hit the ball / serve began")
    miss_impact: Dict[str, Any] = Field(..., description="State snapshot at the moment of missing the ball")
    trajectory: Optional[List[Dict[str, Any]]] = Field(None, description="Recent frame-by-frame trajectory")


DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "failed_shots.json")


def _read_failures() -> List[Dict[str, Any]]:
    if not os.path.isfile(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _write_failures(records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


@app.get("/")
def read_root():
    """
    Return the status of the FastAPI service.
    """
    
    return {
        "status": "online",
        "service": "Pong RL FastAPI Service",
        "current_mode": agent.mode,
        "model_loaded": agent.model_loaded,
        "docs_url": "/docs"
    }


@app.get("/api/status")
def get_status():
    """
    Return the status of the FastAPI service.
    """
    return {
        "mode": agent.mode,
        "model_loaded": agent.model_loaded,
        "model_path": agent.model_path,
        "total_states": agent.discretizer.total_states,
        "n_actions": agent.n_actions,
        "q_table_shape": agent.q_table.shape if agent.q_table is not None else None
    }


@app.post("/api/predict")
def predict_action(payload: PongStateInput):
    """
    Receives the current state of the Pong game and returns the recommended action (0: STAY, 1: UP, 2: DOWN).
    Supports specifying the mode ('dqn' or 'q_learning') directly in the payload.
    """
    try:
        agent.check_auto_reload()
        # Allow specifying the mode per request without depending on a global variable
        target_mode = payload.mode if payload.mode in ["dqn", "q_learning", "heuristic", "random"] else agent.mode
        prev_mode = agent.mode
        agent.mode = target_mode
        try:
            action, debug_info = agent.get_action(
                ball_x=payload.ball_x,
                ball_y=payload.ball_y,
                ball_vx=payload.ball_vx,
                ball_vy=payload.ball_vy,
                paddle_x=payload.ai_paddle_x,
                paddle_y=payload.ai_paddle_y,
                field_width=payload.field_width,
                field_height=payload.field_height
            )
        finally:
            agent.mode = prev_mode

        return {
            "action": action,
            "action_name": ACTION_NAMES.get(action, "UNKNOWN"),
            "debug_info": debug_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing prediction: {str(e)}")


@app.post("/api/mode")
def set_mode(payload: ModeInput):
    """
    Change the mode of the AI agent between 'random', 'q_learning', 'dqn' and 'heuristic'.
    """
    valid_modes = ["random", "q_learning", "dqn", "heuristic"]
    if payload.mode not in valid_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{payload.mode}'. Valid modes are: {valid_modes}"
        )
    
    agent.mode = payload.mode
    
    # If it changes to q_learning mode, try to load/reload the trained q_table.npy
    if payload.mode == "q_learning":
        loaded = agent.load_q_table("models/q_table.npy")
        if not loaded and not agent.model_loaded:
            agent.generate_demo_q_table()
    
    elif payload.mode == "dqn":
        loaded = agent.load_onnx_model("models/dqn_pong.onnx") or agent.load_dqn_model("models/dqn_pong.pth")
        
    return {
        "status": "success",
        "mode": agent.mode,
        "model_loaded": agent.model_loaded,
        "model_path": agent.model_path,
        "message": f"Mode changed to '{agent.mode}'"
    }


@app.post("/api/demo_model")
def load_demo_model():
    """
    Generate and load the Q table for demo purposes.
    """
    agent.generate_demo_q_table()
    agent.mode = "q_learning"
    return {
        "status": "success",
        "message": "Synthetic Q matrix generated successfully. Mode set to 'q_learning'.",
        "q_table_shape": agent.q_table.shape
    }


@app.post("/api/load_model")
def load_model_file(payload: LoadModelInput):
    """
    Load a Q-table (.npy), ONNX model (.onnx) or PyTorch model (.pth) file from disk.
    """
    success = agent.load_q_table(payload.filepath)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load file from {payload.filepath}")
    
    shape_str = str(agent.q_table.shape) if agent.q_table is not None else "ONNX-DQN"
    return {
        "status": "success",
        "filepath": payload.filepath,
        "model_info": shape_str,
        "mode": agent.mode
    }


@app.post("/api/record_failure")
def record_failure(payload: FailureRecordInput):
    """
    Record an episode failure shot where the RL agent conceded a goal.
    Appends the shot origin, miss coordinates, and trajectory to failed_shots.json.
    """
    try:
        records = _read_failures()
        entry = {
            "id": len(records) + 1,
            "timestamp": time.time(),
            "agent_mode": payload.agent_mode,
            "agent_side": payload.agent_side,
            "rally_hits": payload.rally_hits,
            "peak_speed": round(payload.peak_speed, 2),
            "shot_origin": payload.shot_origin,
            "miss_impact": payload.miss_impact,
            "trajectory": payload.trajectory
        }
        records.append(entry)
        _write_failures(records)
        return {
            "status": "success",
            "recorded_id": entry["id"],
            "total_failures": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to persist failure shot: {str(e)}")


@app.get("/api/failures")
def get_failures():
    """
    Retrieve statistics and recent failure records.
    """
    records = _read_failures()
    return {
        "status": "success",
        "count": len(records),
        "file_path": DATA_FILE,
        "recent_samples": records[-10:] if len(records) > 10 else records
    }


@app.delete("/api/failures")
def clear_failures():
    """
    Clear all recorded failure shots.
    """
    _write_failures([])
    return {
        "status": "success",
        "message": "Failure dataset reset to empty.",
        "count": 0
    }


@app.post("/api/reload_models")
def reload_models():
    """
    Force reload Q-table and ONNX DQN models from disk without restarting the service.
    """
    has_q = agent.load_q_table("models/q_table.npy")
    has_onnx = agent.load_onnx_model("models/dqn_pong.onnx")
    return {
        "status": "success",
        "q_table_loaded": has_q,
        "onnx_loaded": has_onnx,
        "mode": agent.mode,
        "message": "Weights reloaded successfully from disk."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
