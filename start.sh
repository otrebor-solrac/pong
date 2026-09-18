#!/bin/bash
set -e

mkdir -p /workspace/logs

echo "=========================================================="
echo "Pong RL Studio: FastAPI (8000) + React UI (5173)"
echo "=========================================================="

# Keep Rust native library and WebAssembly package synced on start
if command -v cargo >/dev/null 2>&1; then
    echo "Checking/updating native Rust physics core (.so)..."
    (cd /workspace/rust/pong_physics && cargo build --release) || true

    echo "Checking/updating WebAssembly physics core for React..."
    (cd /workspace/rust/pong_physics && wasm-pack build --target web --out-dir /workspace/frontend/src/wasm -- --features wasm) || true
fi

echo "Starting React Frontend on http://localhost:5173 ..."
(
    cd /workspace/frontend
    if [ ! -d "node_modules" ]; then
        echo "Installing Node.js dependencies..."
        npm install
    fi
    npm run dev -- --host 0.0.0.0 > /workspace/logs/react_frontend.log 2>&1
) &

echo "Starting FastAPI Backend on http://localhost:8000 ..."
cd /workspace/backend
exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --no-access-log
