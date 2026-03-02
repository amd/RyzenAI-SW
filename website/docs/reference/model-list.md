# Model Table

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Model Table

<CIStatus validated={false} />

This table is sourced from the [Ryzen AI 1.7 documentation](https://ryzenai.docs.amd.com/en/latest/). The "Verified on Device" column will be updated as models are tested through CI on AMD hardware.

| Model | Domain | Device | Context | Verified on Device |
|-------|--------|--------|---------|-------------------|
| Llama-2-7b-chat-hf | LLM | Hybrid, NPU | 4K | Pending |
| Llama-2-7b-hf | LLM | Hybrid, NPU | 4K | Pending |
| Meta-Llama-3-8B | LLM | Hybrid, NPU | 4K | Pending |
| Llama-3.1-8B | LLM | Hybrid, NPU | 4K | Pending |
| Meta-Llama-3.1-8B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Llama-3.2-1B | LLM | Hybrid, NPU | 4K | Pending |
| Llama-3.2-1B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Llama-3.2-3B | LLM | Hybrid | 4K | Pending |
| Llama-3.2-3B-Instruct | LLM | Hybrid | 4K | Pending |
| CodeLlama-7b-Instruct-hf | LLM | Hybrid, NPU | 4K | Pending |
| DeepSeek-R1-Distill-Llama-8B | LLM | Hybrid, NPU | 4K | Pending |
| DeepSeek-R1-Distill-Qwen-1.5B | LLM | Hybrid, NPU | 4K | Pending |
| Qwen-2.5-1.5B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| DeepSeek-R1-Distill-Qwen-7B | LLM | Hybrid, NPU | 4K | Pending |
| Phi-3-mini-4k-instruct | LLM | Hybrid, NPU | 4K | Pending |
| Phi-3-mini-128k-instruct | LLM | Hybrid, NPU | 4K | Pending |
| Phi-3.5-mini-instruct | LLM | Hybrid, NPU | 4K | Pending |
| Phi-4-mini-instruct | LLM | Hybrid | 4K | Pending |
| Phi-4-mini-reasoning | LLM | Hybrid | 4K | Pending |
| gemma-2-2b | LLM | Hybrid | 3K | Pending |
| Mistral-7B-Instruct-v0.1 | LLM | Hybrid, NPU | 4K | Pending |
| Mistral-7B-Instruct-v0.2 | LLM | Hybrid, NPU | 4K | Pending |
| Mistral-7B-Instruct-v0.3 | LLM | Hybrid, NPU | 4K | Pending |
| Mistral-7B-v0.3 | LLM | Hybrid, NPU | 4K | Pending |
| AMD-OLMo-1B-SFT-DPO | LLM | Hybrid | 2K | Pending |
| chatglm3-6b | LLM | Hybrid, NPU | 4K | Pending |
| Qwen1.5-7B-Chat | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2-1.5B | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2-7B | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2.5-0.5B-Instruct | LLM | Hybrid | 4K | Pending |
| Qwen2.5-7B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2.5-Coder-0.5B-Instruct | LLM | Hybrid | 4K | Pending |
| Qwen2.5-Coder-1.5B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2.5-Coder-7B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Qwen2.5-3B-Instruct | LLM | Hybrid, NPU | 4K | Pending |
| Qwen3-1.7B | LLM | Hybrid | 4K | Pending |
| Qwen3-4B | LLM | Hybrid | 4K | Pending |
| Qwen3-8B | LLM | Hybrid | 4K | Pending |

## Notes

1. All models are supported up to 4K context length, with the following exceptions:

- AMD-OLMo-1B-SFT-DPO: inherently supports only 2K context length
- gemma-2-2b: supports up to 3K context length
