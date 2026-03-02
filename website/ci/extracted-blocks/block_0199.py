# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:173
print("Model Size:")
print("Float32 model size: {:.2f} MB".format(os.path.getsize(input_model_path)/(1024 * 1024)))
print("Int8 quantized model size: {:.2f} MB".format(os.path.getsize(output_model_path)/(1024 * 1024)))
