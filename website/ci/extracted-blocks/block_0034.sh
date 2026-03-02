# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\operator-preparation.mdx:32
conda activate ryzen-ai-<version>
python -m onnxruntime_genai.models.builder \
     -i <quantized model folder> -o <dml model folder> \
     -p int4 -e dml
