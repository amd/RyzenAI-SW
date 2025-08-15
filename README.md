<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Software </h1>
    </td>
 </tr>
</table>

## Introduction

AMD Ryzen™ AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on your [AMD Ryzen™ AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) based PC. It enables developers to quickly build and run a variety of AI applications for Ryzen™ AI. It is designed with high efficiency and ease-of-use in mind, unleashing the full potential of AI acceleration on Ryzen™ AI.

This repository contains the demos, examples and tutorials, demonstrating usage and capabilities of the Ryzen™ AI Software. It is a subset of the Ryzen™ AI Software release.

Follow the instructions at [Ryzen™ AI Software](https://ryzenai.docs.amd.com/en/latest/inst.html) for installation.

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

To run the demos and examples in this repository, please follow the instructions of README.md in each directory. 


## Getting Started Tutorials

- [Getting started tutorial with a fine-tuned ResNet model](tutorial/getting_started_resnet)
- [Hello world jupyter notebook tutorial](tutorial/hello_world)
- [Getting started ResNet50 example on iGPU](example/iGPU/getting_started)

## LLM Flow

- [LLMs on RyzenAI with ONNX Runtime GenAI API](example/llm/oga_api)
- [ONNX Runtime GenAI(OGA)‑based RAG LLM](example/llm/RAG-OGA)

## Examples

- BF16 Model Examples
  - [Finetuned DistilBERT for Text Classification](example/DistilBERT_text_classification_bf16)
  - [Image classification](example/image_classification)
  - [Text Embedding Model Alibaba-NLP/gte-large-en-v1.5](example/gte-large-en-v1.5-bf16)
- [Object detection with Yolov8](tutorial/object_detection)


## Demos

- [NPU-GPU pipeline on RyzenAI](demo/NPU-GPU-Pipeline)
- [Automatic Speech Recognition using OpenAI Whisper](demo/ASR/Whisper)


## Other Tutorials

- [AMD Quark Quantization](tutorial/quark_quantization)
- [Run Ryzen AI CVML library application](Ryzen-AI-CVML-Library)
- [Torchvision models End-to-End inference with Ryzen AI](tutorial/torchvision_inference)


## Benchmarking 

- [ONNX benchmark utilities](onnx-benchmark)


## Reference

- [Ryzen™ AI Developer Guide](https://ryzenai.docs.amd.com/en/latest)
- [ONNX Runtime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)
