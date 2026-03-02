# -*- coding: utf-8 -*-
# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\quark_quantization\README.mdx:183
from utils import evaluate_onnx_model  

print("Model Accuracy:")
top1_acc, top5_acc = evaluate_onnx_model(input_model_path, imagenet_data_path='calib_data')
print("Float32 model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))
top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path='calib_data')
print("Int8 quantized model accuracy: Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))
top1_acc, top5_acc = evaluate_onnx_model(output_model_path, imagenet_data_path='calib_data', device='npu')
print("Int8 quantized model accuracy (NPU): Top1 {:.3f}, Top5 {:.3f} ".format(top1_acc, top5_acc))
