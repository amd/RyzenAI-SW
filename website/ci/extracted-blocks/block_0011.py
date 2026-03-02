# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:28
import onnxruntime

session_options = onnxruntime.SessionOptions()
vai_ep_options  = {}                          # Vitis AI EP options go here

session = onnxruntime.InferenceSession(
    path_or_bytes = model,                    # Path to the ONNX model
    sess_options = session_options,           # Standard ORT options
    providers = ['VitisAIExecutionProvider'], # Use the Vitis AI Execution Provider
    provider_options = [vai_ep_options]       # Pass options to the Vitis AI Execution Provider
)
