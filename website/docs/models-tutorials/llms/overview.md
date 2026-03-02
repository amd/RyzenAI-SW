# LLM Deployment Overview

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# LLM Deployment Overview

<CIStatus validated={false} />

Large Language Models (LLMs) can be deployed on Ryzen AI PCs with NPU and GPU acceleration. NPU-only and Hybrid execution modes, which utilize both the NPU and integrated GPU (iGPU), are supported via ONNXRuntime GenAI (OGA). GPU-only acceleration is enabled through llama.cpp. See the [Execution Modes table](#execution-modes) below for detailed information.

## Execution Modes

| Mode | Framework(s) | Compute Allocation | Primary Use Case |
|------|--------------|-------------------|------------------|
| **NPU-Only** | OnnxRuntime GenAI (OGA) | Neural Processing Unit (NPU) exclusive | Maximum NPU utilization while preserving iGPU for parallel workloads |
| **Hybrid** | OnnxRuntime GenAI (OGA) | Dynamic NPU + iGPU partitioning | Interactive inference with optimal prefill/decode performance |
| **GPU** | llama.cpp | Dedicated GPU execution | High-throughput inference on discrete/integrated GPU |
| **CPU** | OGA or llama.cpp | Traditional CPU-based inference | Baseline compatibility across all processor generations |

## Hardware Requirements

| Processor Series | NPU-Only | Hybrid | GPU/CPU |
|------------------|-----------|--------|---------|
| Ryzen AI 300 (STX/KRK) | ✓ | ✓ | ✓ |
| Ryzen AI 7000/8000 | ✗ | ✗ | ✓ |

## Development Interfaces

The Ryzen AI LLM software stack is available through three development interfaces, each suited for specific use cases as outlined in the sections below. All three interfaces are built on top of native OnnxRuntime GenAI (OGA) libraries or llama.cpp libraries, as shown in the diagram below.

The high-level Python APIs, as well as the Server Interface, also leverage the Lemonade SDK, which is multi-vendor open-source software that provides everything necessary for quickly getting started with LLMs on OGA or llama.cpp.

A key benefit of Lemonade is that software developed against their interfaces is portable to many other execution backends.

**Ryzen AI Software Stack:**

| Your Python Application | Your LLM Stack | Your Native Application |
|-------------------------|----------------|-------------------------|
| [Lemonade Python API](/models-tutorials/llms/python-api) | [Lemonade Server Interface](/models-tutorials/llms/server-interface) | [OGA C++ Headers](/models-tutorials/llms/hybrid-inference) **OR** [llama.cpp C++ Headers](https://github.com/ggml-org/llama.cpp) |
| Custom [AMD OnnxRuntime GenAI (OGA)](https://github.com/microsoft/onnxruntime-genai) **OR** [llama.cpp](https://github.com/ggml-org/llama.cpp) | | |
| [AMD Ryzen AI Driver and Hardware](https://www.amd.com/en/products/processors/consumer/ryzen-ai.html) | | |

* indicates open-source software (OSS).

### Server Interface (REST API)

The Server Interface provides a convenient means to integrate with applications that:

- Already support an LLM server interface, such as the Ollama server or OpenAI API.
- Are written in any language (C++, C#, Javascript, etc.) that supports REST APIs.
- Benefits from process isolation for the LLM backend.

Lemonade Server is available in two ways:

- **Standalone Windows GUI installer**: Quick setup with a desktop shortcut for immediate use. (Recommended for end users, see [Server Interface](/models-tutorials/llms/server-interface))
- **Full Lemonade SDK**: Complete development toolkit with server interface included. (Recommended for developers, see [High-Level Python SDK](/models-tutorials/llms/python-api) for Python SDK)

For example applications that have been tested with Lemonade Server, see the [Lemonade Server Examples](https://github.com/lemonade-sdk/lemonade/tree/main/docs/server/apps).

### High-Level Python SDK

The high-level Python SDK, Lemonade, allows you to get started using PyPI installation in approximately 5 minutes.

This SDK allows you to:

- Experiment with models in hybrid or NPU-only execution mode on Ryzen AI hardware.
- Validate inference speed and task performance.
- Integrate with Python apps using a high-level API.

To get started in Python, follow these instructions: [High-Level Python SDK](/models-tutorials/llms/python-api).

### OGA APIs for C++ Libraries and Python

Native C++ libraries for OGA are available to give full customizability for deployment into native applications. The Python bindings for OGA also provide a customizable interface for Python development.

To get started with the OGA APIs, follow these instructions: [Hybrid OGA](/models-tutorials/llms/hybrid-inference).

## Supported LLMs

AMD provides pre-optimized LLMs ready to deploy with Ryzen AI Software. Hugging Face collections: [Hybrid models](https://huggingface.co/collections/amd/ryzen-ai-17-hybrid-llm) | [NPU-only models](https://huggingface.co/collections/amd/ryzen-ai-17-npu-llm)

| Model Family | Parameters | Hybrid | NPU-Only |
|---|---|:---:|:---:|
| [Llama-2](https://huggingface.co/amd/Llama-2-7b-chat-hf-onnx-ryzenai-hybrid) | 7B | ✓ | ✓ |
| [Llama-3 / 3.1 / 3.2](https://huggingface.co/amd/Meta-Llama-3-8B-onnx-ryzenai-hybrid) | 1B–8B | ✓ | ✓ |
| [DeepSeek-R1-Distill](https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-onnx-ryzenai-hybrid) | 1.5B–8B | ✓ | ✓ |
| [Phi-3 / 3.5 / 4](https://huggingface.co/amd/Phi-3-mini-4k-instruct-onnx-ryzenai-hybrid) | 3.8B–4B | ✓ | ✓ |
| [Qwen-2 / 2.5 / 3](https://huggingface.co/amd/Qwen2-7B-onnx-ryzenai-hybrid) | 0.5B–8B | ✓ | ✓ |
| [Mistral-7B](https://huggingface.co/amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-hybrid) | 7B | ✓ | ✓ |
| [CodeLlama](https://huggingface.co/amd/CodeLlama-7b-Instruct-hf-onnx-ryzenai-hybrid) | 7B | ✓ | ✓ |
| [Gemma-2](https://huggingface.co/amd/gemma-2-2b-onnx-ryzenai-hybrid) | 2B | ✓ | — |
| [AMD-OLMo](https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO-onnx-ryzenai-hybrid) | 1B | ✓ | — |
| [ChatGLM3](https://huggingface.co/amd/chatglm3-6b-onnx-ryzenai-hybrid) | 6B | ✓ | ✓ |

For the full list with individual model variants, see [Supported LLMs](/models-tutorials/llms/supported-models).

Fine-tuned versions of these models are also supported. For instructions on preparing a fine-tuned OGA model, refer to [ONNX Model Preparation](/develop/onnx-model-preparation).

## End-to-End OGA Validation

The Lemonade CLI provides built-in tools for end-to-end validation of OGA hybrid and NPU-only execution, including:

- Prompting with templates
- Benchmarking (time-to-first-token and tokens-per-second)
- Accuracy measurement
- Memory profiling

For CLI usage and validation commands, see the [Lemonade Server CLI Guide](https://lemonade-server.ai/docs/server/lemonade-server-cli/). For model-specific validation examples, see each model's page in [Supported LLMs](/models-tutorials/llms/supported-models).
