# Ryzen™ AI Software Platform

## Introduction

AMD Ryzen™ AI Software Platform includes the tools and runtime libraries for for optimizing and deploying AI inference on your [AMD Ryzen™ AI](https://www.amd.com/en/products/ryzen-ai) based PC. It enables developers to quickly build and run a variety of AI applications for Ryzen™ AI. It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Ryzen™ AI.

This repository contains the demos, examples and tutorials, demonstrating usage and capabilities of the Ryzen™ AI Software Platform. It is a subset of the Ryzen™ AI Software release.

    
## Examples

- [Run LLM OPT-1.3B model with ONNXRuntime](example/opt-1.3b/opt-onnx)
- [Run LLM OPT-1.3B model with PyTorch](example/opt-1.3b/opt-pytorch)
- [Run Whipser-tiny model with ONNXRuntime](example/Whipser-tiny) 
- [Run multiple concurrent AI applications with ONNXRuntime](example/multi-model)
- [Real-time object detection with Yolov8](example/Yolov8)

## Demos

- [Cloud-to-Client demo on Ryzen AI](demo/cloud-to-client)
- [Multiple model concurrency demo on Ryzen AI](demo/multi-model-exec)

## Tutorials

- [A Getting Started Tutorial with a ResNet50 model](tutorial/getting_started_resnet)
- [End-to-end Object Detection](tutorial/yolov8_e2e)
- [Quantization for Ryzen AI](tutorial/RyzenAI_quant_tutorial)



## Getting Started
    
To run the demos and examples in this repository, please follow the instructions of README.md in each directory. 

## Reference

- [Ryzen™ AI Developer Guide](https://ryzenai.docs.amd.com/en/latest)
- [ONNXRuntime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [Quantization with Microsoft Olive](https://github.com/microsoft/Olive/tree/main/examples/resnet)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)
