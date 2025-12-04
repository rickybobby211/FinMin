# FinGPT Forecaster - RunPod Serverless Dockerfile
# ================================================
# Build: docker build -t fingpt-forecaster .
# Push:  docker push your-dockerhub/fingpt-forecaster:latest

# Using smaller runtime image (not devel)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers>=4.32.0 \
    peft>=0.5.0 \
    accelerate \
    pandas \
    yfinance \
    finnhub-python \
    runpod \
    huggingface_hub \
    sentencepiece \
    protobuf

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/transformers_cache

# Run handler
CMD ["python", "-u", "/app/handler.py"]

