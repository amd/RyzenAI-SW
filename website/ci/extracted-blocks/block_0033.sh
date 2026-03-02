# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\onnx-model-preparation.mdx:170
conda create --name oga_builder_env python=3.10
conda activate oga_builder_env

pip install onnxruntime-genai==0.9.2
pip install torch transformers onnx numpy


python3 -m onnxruntime_genai.models.builder -m <input quantized model> -o <output OGA model> -p int4 -e dml
