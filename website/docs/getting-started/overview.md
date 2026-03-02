# Overview & Architecture

> import CIStatus from '@site/src/components/CIStatus';
import FeatureState from '@site/src/components/FeatureState';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import CIStatus from '@site/src/components/CIStatus';
import FeatureState from '@site/src/components/FeatureState';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Ryzen AI Software

<CIStatus validated={false} />

AMD Ryzen™ AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on AMD Ryzen™ AI powered PCs. Ryzen AI software enables applications to run on the neural processing unit (NPU) built in the AMD XDNA™ architecture, as well as on the integrated GPU and discrete GPU. Developers can build and deploy models trained in PyTorch or TensorFlow and run them directly on laptops powered by Ryzen AI.

## Development Flow

The Ryzen AI development flow does not require modifications to existing model training processes. A pre-trained model is the starting point:

1. **Quantize** — Convert model parameters to lower precision (INT8, INT4) using [AMD Quark](/develop/model-quantization) for better performance and lower power consumption. Float32 models are also supported and internally converted to bfloat16.
2. **Compile & Deploy** — Deploy the quantized model using [ONNX Runtime](/develop/model-deployment) with the Vitis AI Execution Provider, which automatically partitions operations between the NPU and CPU.
3. **Build Applications** — Use the [Python SDK](/models-tutorials/llms/python-api), [Server Interface](/models-tutorials/llms/server-interface), or [C++ API](/models-tutorials/llms/hybrid-inference) to integrate AI into your application.

## Supported Workloads

| Domain | Examples | Devices | Documentation |
|--------|----------|---------|---------------|
| Large Language Models | [Llama 3.x](https://huggingface.co/meta-llama), [Mistral](https://huggingface.co/mistralai), [Phi-3/4](https://huggingface.co/microsoft), [Qwen 2.5](https://huggingface.co/Qwen), [DeepSeek](https://huggingface.co/deepseek-ai) | NPU, GPU | [LLM Tutorials](/models-tutorials/llms/overview) |
| Vision | [ResNet](https://huggingface.co/microsoft/resnet-50), [YOLOv8](https://huggingface.co/Ultralytics/YOLOv8), [Stable Diffusion](https://huggingface.co/stabilityai), [MobileNet](https://huggingface.co/google/mobilenet_v2_1.0_224) | NPU, GPU | [Vision Tutorials](/models-tutorials/vision/cnn-examples) |
| Audio | [Whisper](https://huggingface.co/openai/whisper-large-v3) (speech-to-text) | NPU | [Whisper Tutorial](/models-tutorials/audio/whisper) |
| Multimodal | [Gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) (VLM) | NPU | [Multimodal Models](/models-tutorials/multimodal/supported-models) |

## LLM Interfaces

The Ryzen AI LLM stack provides three development interfaces, each suited for different use cases:

| Interface | Use Case | Language | Details |
|-----------|----------|----------|---------|
| [Python SDK (Lemonade API)](/models-tutorials/llms/python-api) | Rapid prototyping, scripting | Python | High-level API built on OGA |
| [Server Interface (REST)](/models-tutorials/llms/server-interface) | Integration with existing apps, OpenAI-compatible | HTTP/REST | OpenAI-compatible endpoint |
| [OGA / C++ API](/models-tutorials/llms/hybrid-inference) | Production native apps, hybrid NPU+GPU | C++ | Low-level control, hybrid inference |

## Quick Links

- **[Installation](/getting-started/installation)** — Get Ryzen AI running in minutes
- **[Quickstart](/getting-started/quickstart)** — Run your first model on the NPU
- **[Applications](/applications)** — Pre-built AI PC applications using Ryzen AI
- **[Models & Tutorials](/models-tutorials)** — Verified models and step-by-step guides
- **[Supported Models](/reference/model-list)** — Full list of validated models
- **[GitHub](https://github.com/amd/RyzenAI-SW)** — Source code and examples

## What's New in v1.7

Based on the [Release Notes](/reference/changelog):

- Strix Halo and Krackan Point NPU support
- Hybrid NPU + GPU inference for LLMs
- AMD Quark quantization toolkit
- CVML Library for optimized vision pipelines
- LLM Server Interface with OpenAI-compatible REST API (Lemonade)
- Linux support for LLMs (Ubuntu 22.04)
