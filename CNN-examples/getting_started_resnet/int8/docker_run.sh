#!/bin/bash
# ==============================================================================
# AMD Quark Quantization Pipeline Runner (Docker / Google Cloud Console)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " Building Docker image: ryzenai-resnet-quark"
echo "=================================================================="
docker build -t ryzenai-resnet-quark:latest -f Dockerfile .

echo "=================================================================="
echo " Running Quantization Pipeline in Container"
echo " Mounts local 'models' and 'data' directories"
echo "=================================================================="
mkdir -p models data
docker run --rm \
    -v "${PWD}/models:/app/models" \
    -v "${PWD}/data:/app/data" \
    ryzenai-resnet-quark:latest

echo "=================================================================="
echo " SUCCESS!"
echo " Quantized model generated at: ${PWD}/models/resnet_quantized.onnx"
echo " You can now transfer this model to your Ryzen AI PC for NPU inference."
echo "=================================================================="

