# ==============================================================================
# Dockerfile: AMD Quark Quantization Pipeline for ResNet INT8
# MLOps Decoupled Build Environment (Google Cloud Console / Linux / Docker)
# ==============================================================================
# Purpose:
#   Isolates the AMD Quark Quantization and PyTorch-to-ONNX export pipeline
#   from the host Ryzen AI runtime environment. This completely avoids the
#   Python package dependency conflicts with 'flexml 1.8.0' (Issue #385).
# ==============================================================================

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python quantization dependencies
COPY requirements_quantize.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_quantize.txt

# Copy source code and model directory
COPY resnet_utils.py prepare_model_data.py resnet_quantize.py ./
COPY models/ ./models/

# Create data directory
RUN mkdir -p /app/data /app/models

# Entry command: prepare data, export ONNX, and run Quark INT8 quantization
CMD ["sh", "-c", "python prepare_model_data.py && python resnet_quantize.py && echo '=== Quantization Complete! Artifact: models/resnet_quantized.onnx ==='"]

