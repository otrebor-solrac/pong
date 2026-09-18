FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Install uv for ultra-fast Python package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Minimum system dependencies and Node.js 20 LTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Layer 1: Install Rust toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Layer 2: Add WebAssembly compilation target
RUN rustup target add wasm32-unknown-unknown

# Layer 3: Install wasm-pack for browser WASM packaging
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

WORKDIR /workspace

# Installation of libraries for RL, FastAPI and evaluation
RUN uv pip install --system \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    rich \
    onnx \
    onnxruntime-gpu \
    fastapi \
    uvicorn \
    pydantic \
    pudb \
    scikit-learn

ENV PYTHONPATH="/workspace/backend:${PYTHONPATH}"

EXPOSE 8000 5173

CMD ["bash", "/workspace/start.sh"]
