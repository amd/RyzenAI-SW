# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\onnx-model-preparation.mdx:127
conda activate ryzen-ai-<version>

model_generate --npu <output_dir> <quantized_model_path>  --optimize decode
