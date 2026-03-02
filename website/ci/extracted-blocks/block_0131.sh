# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\getting_started_resnet\bf16\README.mdx:37
conda create --name resnet_bf16 --clone ryzen-ai-<version>
conda activate resnet_bf16
set RYZEN_AI_INSTALLATION_PATH = <path/to/RyzenAI/installation>

cd <RyzenAI-SW>\CNN-examples\getting_started_resnet\bf16
python -m pip install -r requirements.txt
