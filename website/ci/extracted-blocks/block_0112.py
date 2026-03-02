# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\cnn-examples.mdx:100
from quark.onnx.quantization.config import (Config, get_default_config)
from quark.onnx import ModelQuantizer

# Get quantization configuration
quant_config = get_default_config("XINT8")
config = Config(global_quant_config=quant_config)

# Create an ONNX quantizer
quantizer = ModelQuantizer(config)

# Quantize the ONNX model
quantizer.quantize_model(input_model_path, output_model_path, dr)
