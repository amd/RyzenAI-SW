# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:85
import onnxruntime

session_options = onnxruntime.SessionOptions()
session_options.log_severity_level = 1  # Set log level (see table below)

vai_ep_options = {}

session = onnxruntime.InferenceSession(
    path_or_bytes = model,
    sess_options = session_options,
    providers = ['VitisAIExecutionProvider'],
    provider_options = [vai_ep_options]
)
