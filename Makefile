# Makefile for Pong RL Studio

.PHONY: help up down stop restart build shell train train-dqn train-failures clean build-rust build-wasm build-all

help:
	@echo "Pong RL Studio - Available commands:"
	@echo "  make up             - Start web interface (5173) and API (8000)"
	@echo "  make down           - Stop containers"
	@echo "  make build          - Rebuild Docker image"
	@echo "  make build-rust     - Compile Rust physics library (.so) for Python"
	@echo "  make build-wasm     - Compile Rust physics engine to WebAssembly for React"
	@echo "  make build-all      - Compile both Rust native library and WASM package"
	@echo "  make shell          - Open interactive bash terminal inside container"
	@echo "  make train          - Train Tabular Q-Learning agent"
	@echo "  make train-dqn      - Train DQN (Deep Q-Network) agent"
	@echo "  make train-failures - Train DQN on real-world failed shots clinic"
	@echo "  make clean          - Clean logs and cache files"

build-rust:
	docker compose run --rm pong bash -c "cd /workspace/rust/pong_physics && cargo build --release"

build-wasm:
	docker compose run --rm pong bash -c "cd /workspace/rust/pong_physics && wasm-pack build --target web --out-dir /workspace/frontend/src/wasm -- --features wasm"

build-all: build-rust build-wasm

up:
	docker compose run --rm --service-ports pong bash /workspace/start.sh

down:
	docker compose down --remove-orphans
	@docker ps -q --filter "name=pong" | xargs -r docker stop

stop: down

restart: stop up

build:
	docker compose build

shell:
	docker compose run --rm --service-ports pong bash

train:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo q_learning $(ARGS)

train-dqn:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo dqn $(ARGS)

eval-dqn:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo dqn --eval --episodes 200 --opponent heuristic --pretrained models/dqn_pong.onnx $(ARGS)

eval-heuristic:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo heuristic --eval --episodes 200 --opponent heuristic $(ARGS)

eval-tabular:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo q_learning --eval --episodes 200 --opponent heuristic --pretrained models/q_table.npy $(ARGS)

train-failures:
	docker compose run --rm pong python3 /workspace/backend/train.py --algo dqn --train_failures data/failed_shots.json $(ARGS)

clean:
	rm -rf logs/*.log frontend/node_modules backend/__pycache__
