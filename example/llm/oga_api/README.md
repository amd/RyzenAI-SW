# Ryzen AI LLM - Onnxruntime GenAI

Ryzen AI Software includes support for deploying LLMs on Ryzen AI PCs using the ONNX Runtime generate() API (OGA). 

## Pre-optimized Models

AMD provides a set of pre-optimized LLMs ready to be deployed with Ryzen AI Software and the supporting runtime for hybrid execution. These models can be found on Hugging Face: 

### Published models: 

- https://huggingface.co/amd/Phi-3-mini-4k-instruct-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Phi-3.5-mini-instruct-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Mistral-7B-Instruct-v0.3-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Qwen1.5-7B-Chat-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/chatglm3-6b-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Llama-2-7b-hf-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Llama-2-7b-chat-hf-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid/tree/main 
- https://huggingface.co/amd/Llama-3.1-8B-awq-g128-int4-asym-fp16-onnx-hybrid/tree/main 
- https://huggingface.co/amd/Llama-3.2-1B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Llama-3.2-3B-Instruct-awq-g128-int4-asym-fp16-onnx-hybrid 
- https://huggingface.co/amd/Mistral-7B-Instruct-v0.1-hybrid 
- https://huggingface.co/amd/Mistral-7B-Instruct-v0.2-hybrid 
- https://huggingface.co/amd/Mistral-7B-v0.3-hybrid 
- https://huggingface.co/amd/Llama-3.1-8B-Instruct-hybrid 
- https://huggingface.co/amd/CodeLlama-7b-instruct-g128-hybrid 
- https://huggingface.co/amd/DeepSeek-R1-Distill-Llama-8B-awq-asym-uint4-g128-lmhead-onnx-hybrid 
- https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-1.5B-awq-asym-uint4-g128-lmhead-onnx-hybrid
- https://huggingface.co/amd/DeepSeek-R1-Distill-Qwen-7B-awq-asym-uint4-g128-lmhead-onnx-hybrid
- https://huggingface.co/amd/AMD-OLMo-1B-SFT-DPO-hybrid
- https://huggingface.co/amd/Qwen2-7B-awq-uint4-asym-g128-lmhead-fp16-onnx-hybrid
- https://huggingface.co/amd/Qwen2-1.5B-awq-uint4-asym-global-g128-lmhead-g32-fp16-onnx-hybrid
- https://huggingface.co/amd/gemma-2-2b-awq-uint4-asym-g128-lmhead-g32-fp16-onnx-hybrid


## Run Models using C++ and Python Onnxruntime GenAI API

- The steps for deploying the pre-optimized models using Python or C++ APIs for the Hybrid execution mode of LLMs, which leverages both the NPU and GPU can be found in the Official Ryzen AI Software 1.4 documantion page here - https://ryzenai.docs.amd.com/en/develop/hybrid_oga.html

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
