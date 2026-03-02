# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\onnx-model-preparation.mdx:40
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
