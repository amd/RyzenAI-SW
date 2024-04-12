# Ryzen™ AI Software 

## Introduction

AMD Ryzen™ AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on your [AMD Ryzen™ AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) based PC. It enables developers to quickly build and run a variety of AI applications for Ryzen™ AI. It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Ryzen™ AI.

This repository contains the demos, examples and tutorials, demonstrating usage and capabilities of the Ryzen™ AI Software. It is a subset of the Ryzen™ AI Software release.

## Git LFS and Instructions to clone: 

 Due to the presence of large files in some examples/tutorials, Git Large File Storage (LFS) has been configured in this repository. Follow the instructions below to ensure Git LFS is properly set up: 
 - Install Git LFS by downloading it from the [official website](https://git-lfs.com/)
 - After installation, run the following command in your terminal to set up Git LFS on your machine:
```
 git lfs install
```
 - Clone the repository (or a fork of it): 
```
git clone https://github.com/amd/RyzenAI-SW.git
```
- Pull the actual LFS files: 
```
git lfs pull
```

## Examples

- [Run Vitis AI ONNX Quantizer example](example/onnx_quantizer)
- [Real-time object detection with Yolov8](example/yolov8)
- [Run multiple concurrent AI applications with ONNXRuntime](example/multi-model)
- [Run Ryzen AI Library example](example/Ryzen-AI-Library)
- [Run ONNX end-to-end examples with custom pre/post-processing nodes running on IPU](https://github.com/amd/RyzenAI-SW/tree/main/example/onnx-e2e)
- Generative AI Examples
   - [Run LLM OPT-1.3B model with ONNXRuntime](example/transformers/)
   - [Run LLM OPT-1.3B model with PyTorch](example/transformers/)
   - [Run LLM Llama 2 model with PyTorch](example/transformers/)

## Demos

- [Cloud-to-Client demo on Ryzen AI](demo/cloud-to-client)
- [Multiple model concurrency demo on Ryzen AI](demo/multi-model-exec)

## Tutorials

- [A Getting Started Tutorial with a fine-tuned ResNet model](tutorial/getting_started_resnet)
- [Hello World Jupyter Notebook Tutorial](tutorial/hello_world)
- [End-to-end Object Detection](tutorial/yolov8_e2e)
- [Quantization for Ryzen AI](tutorial/RyzenAI_quant_tutorial)

## Benchmarking 

- [ONNX Benchmarking utilities](onnx-benchmark)



## Getting Started
    
To run the demos and examples in this repository, please follow the instructions of README.md in each directory. 

## Reference

- [Ryzen™ AI Developer Guide](https://ryzenai.docs.amd.com/en/latest)
- [ONNXRuntime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)
