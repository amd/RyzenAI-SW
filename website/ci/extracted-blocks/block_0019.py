# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\develop\model-deployment.mdx:351
import onnxruntime

vai_ep_options = {
    'encryption_key': '89703f950ed9f738d956f6769d7e45a385d3c988ca753838b5afbc569ebf35b2'
}

# Compilation session
session_options = onnxruntime.SessionOptions()
session_options.add_session_config_entry('ep.context_enable', '1')
session_options.add_session_config_entry('ep.context_file_path', 'context_model.onnx')
session_options.add_session_config_entry('ep.context_embed_mode', '1')
session = onnxruntime.InferenceSession(
    path_or_bytes='resnet50_int8.onnx',  # Load the ONNX model
    sess_options=session_options,
    providers=['VitisAIExecutionProvider'],
    provider_options=[vai_ep_options]
)

# Inference session
session_options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(
    path_or_bytes='context_model.onnx', # Load the EP context model
    sess_options=session_options,
    providers=['VitisAIExecutionProvider'],
    provider_options=[vai_ep_options]
)
