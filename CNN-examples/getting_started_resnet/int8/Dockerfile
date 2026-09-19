# ==============================================================================
# Dockerfile: AMD Quark Quantization Pipeline for ResNet INT8
# MLOps Decoupled Build Environment (Google Cloud Console / Linux / Docker)
# ==============================================================================
# Purpose:
#   Isolates the AMD Quark Quantization and PyTorch-to-ONNX export pipeline
#   from the host Ryzen AI runtime environment. This completely avoids the
#   Python package dependency conflicts with 'flexml 1.8.0' (Issue #385).
#
# Volume Mount Contract:
#   - /app/models : Mounted from host to persist exported ONNX & quantized models
#   - /app/data   : Mounted from host to cache downloaded CIFAR-10 datasets
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

# Install exact pinned Python quantization dependencies for reproducibility
COPY requirements_quantize.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_quantize.txt

# Copy source scripts
COPY resnet_utils.py prepare_model_data.py resnet_quantize.py ./

# Create explicit directories for host volume mounts
RUN mkdir -p /app/data /app/models

# Entry command: prepare data, export ONNX, and run Quark INT8 quantization
CMD ["sh", "-c", "python prepare_model_data.py && python resnet_quantize.py && echo '=== Quantization Complete! Artifact: models/resnet_quantized.onnx ==='"]
