# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\tools\ai-analyzer.mdx:31
import onnxruntime as ort
from pathlib import Path

cache_dir = Path(__file__).parent.resolve()
providers = ['VitisAIExecutionProvider']

provider_options = [{
    'config_file': 'vaip_config.json',
    'cacheDir': str(cache_dir),
    'cacheKey': 'modelcachekey',
    'ai_analyzer_visualization': True,
    'ai_analyzer_profiling': True,
}]
session = ort.InferenceSession(
    "model.onnx",
    providers=providers,
    provider_options=provider_options,
)
