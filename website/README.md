# Ryzen AI Software

> AMD Ryzen AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on [AMD Ryzen AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) PCs. It enables developers to build and run AI applications on the neural processing unit (NPU), integrated GPU, and discrete GPU.

# Ryzen AI Software

AMD Ryzen AI Software includes the tools and runtime libraries for optimizing and deploying AI inference on [AMD Ryzen AI](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) PCs. It enables developers to build and run AI applications on the neural processing unit (NPU), integrated GPU, and discrete GPU.

This repository contains documentation, examples, and tutorials demonstrating the usage and capabilities of Ryzen AI Software.

## Documentation

The full documentation site is built from the `docs/` directory using [Docusaurus](https://docusaurus.io/).

**Live docs:** [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com)

To run the docs site locally:

```bash
cd docs/website
npm install
npx docusaurus start
```

## Repository Structure

```
RyzenAI-SW/
├── docs/                        # Documentation and examples
│   ├── getting-started/         # Installation, quickstart, hardware support
│   ├── applications/            # Showcased AI PC applications
│   ├── models-tutorials/        # Models, tutorials, and example code
│   │   ├── llms/                # LLM tutorials and examples
│   │   ├── vision/              # CNN, object detection, image classification
│   │   ├── audio/               # Whisper ASR examples
│   │   ├── multimodal/          # VLM and multi-model pipelines
│   │   └── nlp/                 # DistilBERT and NLP examples
│   ├── develop/                 # Developer guides (deployment, quantization)
│   ├── tools/                   # AI Analyzer, NPU management, benchmarking
│   ├── reference/               # Changelog, model list, supported operators
│   └── website/                 # Docusaurus build infrastructure
└── .github/workflows/           # CI/CD pipelines
```

## Getting Started

- [Installation](docs/getting-started/installation.mdx)
- [Quickstart](docs/getting-started/quickstart.mdx)
- [Supported Hardware](docs/getting-started/supported-hardware.mdx)

## LLM Tutorials

- [LLMs on Ryzen AI with OGA API](docs/models-tutorials/llms/oga_api)
- [RAG with OGA](docs/models-tutorials/llms/RAG-OGA)
- [Vision Language Model (VLM)](docs/models-tutorials/llms/VLM)
- [OGA Inference](docs/models-tutorials/llms/oga_inference)
- [LLM Fine-tuning and Deployment](docs/models-tutorials/llms/llm-sft-deploy)

## Vision Examples

- [Getting Started with ResNet](docs/models-tutorials/vision/getting_started_resnet)
- [Hello World Notebook](docs/models-tutorials/vision/hello_world)
- [iGPU Getting Started](docs/models-tutorials/vision/iGPU/getting_started)
- [Image Classification](docs/models-tutorials/vision/image_classification)
- [Object Detection (YOLOv8)](docs/models-tutorials/vision/object_detection)
- [Super-Resolution](docs/models-tutorials/vision/super-resolution)
- [Torchvision Inference](docs/models-tutorials/vision/torchvision_inference)
- [AMD Quark Quantization](docs/models-tutorials/vision/quark_quantization)
- [CVML Library](docs/models-tutorials/vision/cvml)

## Audio Examples

- [Whisper ASR](docs/models-tutorials/audio/whisper)

## Multimodal Examples

- [NPU-GPU Pipeline](docs/models-tutorials/multimodal/npu-gpu-pipeline)

## NLP Examples

- [DistilBERT Text Classification](docs/models-tutorials/nlp/distilbert)

## Tools

- [ONNX Benchmark Utilities](docs/tools/benchmarking)
- [NPU Check](docs/tools/npu-check)

## Git LFS

Some examples contain large files managed by Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

## Reference

- [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html)
- [AMD Developer Community Discord](https://discord.gg/amd-dev)
- [AMD AI Developer Forum](https://community.amd.com/t5/ai/ct-p/amd_ai)
- [Ryzen AI Developer Guide](https://ryzenai.docs.amd.com)
- [ONNX Runtime Vitis-AI EP](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html)
