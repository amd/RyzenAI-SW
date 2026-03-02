# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:170
import onnxruntime

vai_ep_options = {
    'config_file': 'vai_ep_config.json',
}

session = onnxruntime.InferenceSession(
    "resnet50.onnx",
    providers=['VitisAIExecutionProvider'],
    provider_options=[vai_ep_options]
)
