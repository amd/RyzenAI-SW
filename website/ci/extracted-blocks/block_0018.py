# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:301
import onnxruntime
from pathlib import Path

vai_ep_options = {
    'cache_dir': str(Path(__file__).parent.resolve()),
    'cache_key': 'compiled_resnet50_int8',
    'enable_cache_file_io_in_mem': '0'
}

session = onnxruntime.InferenceSession(
    "resnet50_int8.onnx",
    providers=['VitisAIExecutionProvider'],
    provider_options=[vai_ep_options]
)
