# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:234
import os
import onnxruntime

vai_ep_options = {
    'cache_dir': './model_cache',
    'cache_key': 'resnet_trained_for_cifar10',
    'enable_cache_file_io_in_mem': '0',
    'target': 'X2' # Default option 'X2'
}

session = onnxruntime.InferenceSession(
    "resnet50_int8.onnx",
    providers=['VitisAIExecutionProvider'],
    provider_options=[vai_ep_options]
)
