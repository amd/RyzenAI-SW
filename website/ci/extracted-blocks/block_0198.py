# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:158
from quark.onnx import ModelQuantizer
# Create an ONNX Quantizer
quantizer = ModelQuantizer(config)

# Quantize the ONNX model
quant_model = quantizer.quantize_model(model_input = input_model_path, 
                                       model_output = output_model_path, 
                                       calibration_data_path = None)
