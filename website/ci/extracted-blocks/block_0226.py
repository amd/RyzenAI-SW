# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\tools\ai-analyzer.mdx:56
import onnxruntime as ort
from pathlib import Path

cache_dir = Path(__file__).parent.resolve()
providers = ['VitisAIExecutionProvider']

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True

provider_options = [{
    'config_file': 'vaip_config.json',
    'cacheDir': str(cache_dir),
    'cacheKey': 'modelcachekey',
    'ai_analyzer_visualization': True,
    'ai_analyzer_profiling': True,
}]

session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=providers,
    provider_options=provider_options,
)
