# Multimodal: Supported Models

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Multimodal: Supported Models

<CIStatus validated={false} />

Multimodal models on Ryzen AI support vision-language tasks (VLMs), combining image understanding with natural language. These models can process images and answer questions, generate captions, or perform other joint vision-language tasks.

For a Vision-Language example, see the [VLM example](/models-tutorials/llms/vlm) in the Ryzen AI repository.

For multi-model pipelines that combine NPU and GPU inference, see the [NPU-GPU Pipeline](/models-tutorials/multimodal/npu-gpu-pipeline) tutorial.

| Model | Task | Device | Hugging Face | Verified |
|-------|------|--------|-------------|----------|
| Gemma-3-4b-it-mm | Vision-Language | NPU | [amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu](https://huggingface.co/amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu) | Pending |
