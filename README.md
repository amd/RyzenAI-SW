<table width="100%">
 <tr width="100%">
    <td align="center"><h1>Ryzen&trade; AI Software</h1></td>
 </tr>
</table>

## Introduction

AMD Ryzen&trade; AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on your [AMD Ryzen&trade; AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) based PC. It enables developers to quickly build and run a variety of AI applications for Ryzen&trade; AI, taking advantage of the neural processing unit (NPU), integrated GPU, and CPU.

This repository contains the demos, examples, and tutorials demonstrating the usage and capabilities of the Ryzen&trade; AI Software, along with the source for the documentation site.

Follow the instructions at [Ryzen&trade; AI Software Installation](https://ryzenai.docs.amd.com/en/latest/inst.html) to get set up.

## Documentation

**Full documentation:** [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com)

The documentation source lives in the [`docs/`](docs/) directory.

## Git LFS and Instructions to clone

Due to the presence of large files in some examples/tutorials, Git Large File Storage (LFS) is configured in this repository. Follow the instructions below to ensure Git LFS is properly set up:

- Install Git LFS from the [official website](https://git-lfs.com/).
- After installation, set up Git LFS on your machine:

```
git lfs install
```

- Clone the repository (or a fork of it):

```
git clone https://github.com/amd/RyzenAI-SW.git
```

- Pull the LFS files:

```
git lfs pull
```

## Getting Started Tutorials

- [Getting started with a fine-tuned ResNet model](docs/vision/getstartex.mdx)
- [Hello World tutorial](docs/vision/hello-world.mdx)
- [ResNet50 on iGPU](docs/vision/igpu-getting-started.mdx)

## LLM Flow

- [LLMs on Ryzen AI with the ONNX Runtime GenAI (OGA) API](docs/llms/oga-cpp-api.mdx)
- [ONNX Runtime GenAI (OGA)-based RAG LLM](docs/llms/rag-oga.mdx)
- [Vision Language Model (VLM) on Ryzen AI NPU](docs/llms/vlm.mdx)
- [GPT-OSS-20B with chat template](docs/llms/oga-inference.mdx)

## Examples

- BF16 model examples
  - [Finetuned DistilBERT for Text Classification](docs/llms/distilbert-example.mdx)
  - [Image classification](docs/vision/image-classification.mdx)
- [Object detection with YOLOv8](docs/vision/yolov8m.mdx)
- [Super-Resolution](docs/vision/super_resolution.mdx)
- [Nemotron OCR v2 on AMD Ryzen AI NPU](docs/vision/nemotron-ocr-v2.mdx)

## Windows ML Examples

- [Running ResNet with Windows ML](docs/winml-examples/resnet.mdx)
- [Running Transformer models with Windows ML](docs/winml-examples/googlebert.mdx)
- [Running CLIP with Windows ML](docs/winml-examples/clip.mdx)

## Demos

- [NPU-GPU pipeline on Ryzen AI](docs/vision/npu-gpu-pipeline.mdx)
- [Automatic Speech Recognition using OpenAI Whisper](docs/audio/whisper-asr.mdx)
- [Automatic Speech Recognition using NVIDIA Parakeet TDT optimized for AMD Ryzen AI](docs/audio/parakeet-tdt.mdx)

## Other Tutorials

- [AMD Quark Quantization](docs/benchmarking/quark-quantization.mdx)
- [Run Ryzen AI CVML library application](docs/vision/cvml.mdx)
- [Torchvision models end-to-end inference with Ryzen AI](docs/vision/torchvision.mdx)

## Benchmarking

- [ONNX benchmark utilities](docs/benchmarking/onnx-benchmark.mdx)

## Reference

- [Ryzen&trade; AI Developer Guide](https://ryzenai.docs.amd.com/en/latest)
- [ONNX Runtime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)
- [AMD Developer Community Discord](https://discord.gg/amd-dev)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
