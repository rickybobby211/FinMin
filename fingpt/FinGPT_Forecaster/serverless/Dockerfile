# FinGPT Forecaster - RunPod Serverless Dockerfile
# ================================================
# Build: docker build -t fingpt-forecaster .
# Push:  docker push your-dockerhub/fingpt-forecaster:latest

# Use NVIDIA CUDA base image to start fresh and avoid Conda conflicts
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/tmp/transformers_cache

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

# Install PyTorch specifically compatible with CUDA 12.1
# We use the wheel directly to ensure no other version sneaks in
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install \
    "transformers>=4.40.0,<5.0.0" \
    peft \
    accelerate \
    pandas \
    yfinance \
    finnhub-python \
    openai \
    runpod \
    huggingface_hub \
    protobuf \
    bitsandbytes \
    scipy \
    sentencepiece

# Copy handler code
COPY handler.py /app/handler.py

# Run handler
CMD ["python", "-u", "/app/handler.py"]
