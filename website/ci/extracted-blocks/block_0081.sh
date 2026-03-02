# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\llms\linux-setup.mdx:151
pip install onnx-ir 

model_generate --npu <output_dir> <quantized_model_path> --optimize decode
