# Supported LLMs

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Supported LLMs

<CIStatus validated={false} />

This model list is sourced from the [Ryzen AI 1.7 documentation](https://ryzenai.docs.amd.com/en/latest/). Pre-optimized ONNX models are available on Hugging Face. The "Verified on Device" column will be updated as models are tested through CI on AMD hardware.

:::note Quantization
Most models use **INT4 (AWQ, group size 128)**. Phi-4 models use **GPTQ** quantization.
:::

## Collections

- **[Hybrid (CPU + NPU)](https://huggingface.co/collections/amd/ryzen-ai-17-hybrid-llm)** — Models that run across CPU and NPU
- **[NPU-Only](https://huggingface.co/collections/amd/ryzen-ai-17-npu-llm)** — Models that run entirely on the NPU

---

## Llama Family

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| Llama-2-7b-chat-hf | 7B | ✓ | ✓ | 4K | [Llama-2-7b-chat-hf](https://huggingface.co/amd/Llama-2-7b-chat-hf-onnx-ryzenai-hybrid) | Pending |
| Llama-2-7b-hf | 7B | ✓ | ✓ | 4K | [Llama-2-7b-hf](https://huggingface.co/amd/Llama-2-7b-hf-onnx-ryzenai-hybrid) | Pending |
| Meta-Llama-3-8B | 8B | ✓ | ✓ | 4K | [Meta-Llama-3-8B](https://huggingface.co/amd/Meta-Llama-3-8B-onnx-ryzenai-hybrid) | Pending |
| Llama-3.1-8B | 8B | ✓ | ✓ | 4K | [Llama-3.1-8B](https://huggingface.co/amd/Llama-3.1-8B-onnx-ryzenai-hybrid) | Pending |
| Meta-Llama-3.1-8B-Instruct | 8B | ✓ | ✓ | 4K | [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/amd/Meta-Llama-3.1-8B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Llama-3.2-1B | 1B | ✓ | ✓ | 4K | [Llama-3.2-1B](https://huggingface.co/amd/Llama-3.2-1B-onnx-ryzenai-hybrid) | Pending |
| Llama-3.2-1B-Instruct | 1B | ✓ | ✓ | 4K | [Llama-3.2-1B-Instruct](https://huggingface.co/amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Llama-3.2-3B | 3B | ✓ | — | 4K | [Llama-3.2-3B](https://huggingface.co/amd/Llama-3.2-3B-onnx-ryzenai-hybrid) | Pending |
| Llama-3.2-3B-Instruct | 3B | ✓ | — | 4K | [Llama-3.2-3B-Instruct](https://huggingface.co/amd/Llama-3.2-3B-Instruct-onnx-ryzenai-hybrid) | Pending |
| CodeLlama-7b-Instruct-hf | 7B | ✓ | ✓ | 4K | [CodeLlama-7b-Instruct-hf](https://huggingface.co/amd/CodeLlama-7b-Instruct-hf-onnx-ryzenai-hybrid) | Pending |

## DeepSeek Family

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| DeepSeek-R1-Distill-Llama-8B | 8B | ✓ | ✓ | 4K | [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-onnx-ryzenai-hybrid) | Pending |
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | ✓ | ✓ | 4K | [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-1.5B-onnx-ryzenai-hybrid) | Pending |
| DeepSeek-R1-Distill-Qwen-7B | 7B | ✓ | ✓ | 4K | [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-7B-onnx-ryzenai-hybrid) | Pending |

## Phi Family

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| Phi-3-mini-4k-instruct | 3.8B | ✓ | ✓ | 4K | [Phi-3-mini-4k-instruct](https://huggingface.co/amd/Phi-3-mini-4k-instruct-onnx-ryzenai-hybrid) | Pending |
| Phi-3-mini-128k-instruct | 3.8B | ✓ | ✓ | 4K | [Phi-3-mini-128k-instruct](https://huggingface.co/amd/Phi-3-mini-128k-instruct-onnx-ryzenai-hybrid) | Pending |
| Phi-3.5-mini-instruct | 3.8B | ✓ | ✓ | 4K | [Phi-3.5-mini-instruct](https://huggingface.co/amd/Phi-3.5-mini-instruct-onnx-ryzenai-hybrid) | Pending |
| Phi-4-mini-instruct | 4B | ✓ | — | 4K | [Phi-4-mini-instruct](https://huggingface.co/amd/Phi-4-mini-instruct-onnx-ryzenai-hybrid) | Pending |
| Phi-4-mini-reasoning | 4B | ✓ | — | 4K | [Phi-4-mini-reasoning](https://huggingface.co/amd/Phi-4-mini-reasoning-onnx-ryzenai-hybrid) | Pending |

## Qwen Family

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| Qwen-2.5-1.5B-Instruct | 1.5B | ✓ | ✓ | 4K | [Qwen-2.5-1.5B-Instruct](https://huggingface.co/amd/Qwen-2.5-1.5B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen1.5-7B-Chat | 7B | ✓ | ✓ | 4K | [Qwen1.5-7B-Chat](https://huggingface.co/amd/Qwen1.5-7B-Chat-onnx-ryzenai-hybrid) | Pending |
| Qwen2-1.5B | 1.5B | ✓ | ✓ | 4K | [Qwen2-1.5B](https://huggingface.co/amd/Qwen2-1.5B-onnx-ryzenai-hybrid) | Pending |
| Qwen2-7B | 7B | ✓ | ✓ | 4K | [Qwen2-7B](https://huggingface.co/amd/Qwen2-7B-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-0.5B-Instruct | 0.5B | ✓ | — | 4K | [Qwen2.5-0.5B-Instruct](https://huggingface.co/amd/Qwen2.5-0.5B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-7B-Instruct | 7B | ✓ | ✓ | 4K | [Qwen2.5-7B-Instruct](https://huggingface.co/amd/Qwen2.5-7B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-Coder-0.5B-Instruct | 0.5B | ✓ | — | 4K | [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/amd/Qwen2.5-Coder-0.5B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-Coder-1.5B-Instruct | 1.5B | ✓ | ✓ | 4K | [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/amd/Qwen2.5-Coder-1.5B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-Coder-7B-Instruct | 7B | ✓ | ✓ | 4K | [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/amd/Qwen2.5-Coder-7B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen2.5-3B-Instruct | 3B | ✓ | ✓ | 4K | [Qwen2.5-3B-Instruct](https://huggingface.co/amd/Qwen2.5-3B-Instruct-onnx-ryzenai-hybrid) | Pending |
| Qwen3-1.7B | 1.7B | ✓ | — | 4K | [Qwen3-1.7B](https://huggingface.co/amd/Qwen3-1.7B-awq-quant-onnx-hybrid) | Pending |
| Qwen3-4B | 4B | ✓ | — | 4K | [Qwen3-4B](https://huggingface.co/amd/Qwen3-4B-awq-quant-onnx-hybrid) | Pending |
| Qwen3-8B | 8B | ✓ | — | 4K | [Qwen3-8B](https://huggingface.co/amd/Qwen3-8B-awq-quant-onnx-hybrid) | Pending |

## Mistral Family

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| Mistral-7B-Instruct-v0.1 | 7B | ✓ | ✓ | 4K | [Mistral-7B-Instruct-v0.1](https://huggingface.co/amd/Mistral-7B-Instruct-v0.1-onnx-ryzenai-hybrid) | Pending |
| Mistral-7B-Instruct-v0.2 | 7B | ✓ | ✓ | 4K | [Mistral-7B-Instruct-v0.2](https://huggingface.co/amd/Mistral-7B-Instruct-v0.2-onnx-ryzenai-hybrid) | Pending |
| Mistral-7B-Instruct-v0.3 | 7B | ✓ | ✓ | 4K | [Mistral-7B-Instruct-v0.3](https://huggingface.co/amd/Mistral-7B-Instruct-v0.3-onnx-ryzenai-hybrid) | Pending |
| Mistral-7B-v0.3 | 7B | ✓ | ✓ | 4K | [Mistral-7B-v0.3](https://huggingface.co/amd/Mistral-7B-v0.3-onnx-ryzenai-hybrid) | Pending |

## Other Models

| Model | Parameters | Hybrid | NPU-Only | Context | Hugging Face | Verified on Device |
|-------|------------|:------:|:--------:|---------|--------------|--------------------|
| gemma-2-2b | 2B | ✓ | — | 3K | [gemma-2-2b](https://huggingface.co/amd/gemma-2-2b-onnx-ryzenai-hybrid) | Pending |
| AMD-OLMo-1B-SFT-DPO | 1B | ✓ | — | 2K | [AMD-OLMo-1B-SFT-DPO](https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO-onnx-ryzenai-hybrid) | Pending |
| chatglm3-6b | 6B | ✓ | ✓ | 4K | [chatglm3-6b](https://huggingface.co/amd/chatglm3-6b-onnx-ryzenai-hybrid) | Pending |
