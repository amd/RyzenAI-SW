# Models & Tutorials

> Browse verified models and step-by-step tutorials organized by AI domain. All examples are available in the [RyzenAI-SW GitHub repository](https://github.com/amd/RyzenAI-SW).

# Models & Tutorials

Browse verified models and step-by-step tutorials organized by AI domain. All examples are available in the [RyzenAI-SW GitHub repository](https://github.com/amd/RyzenAI-SW).

| Domain | Description |
|--------|-------------|
| [Audio](/models-tutorials/audio/supported-models) | Speech recognition, text-to-speech |
| [Large Language Models](/models-tutorials/llms/overview) | Chat, RAG, code generation, text classification |
| [Multimodal](/models-tutorials/multimodal/supported-models) | Vision-language models |
| [Vision](/models-tutorials/vision/supported-models) | Classification, detection, generation |

Models listed in the Supported Models tables are from the official AMD Hugging Face collections. The "Verified on Device" column tracks whether each model has been tested on AMD hardware through CI.

## Vision Examples

| Example | Description | Device | Language |
|---------|-------------|--------|----------|
| [Hello World](/models-tutorials/vision/hello-world) | Simple ONNX model on NPU | NPU | Python |
| [ResNet CIFAR-10 (INT8)](/models-tutorials/vision/getting-started-resnet/int8) | Image classification with INT8 quantization | NPU | Python, C++ |
| [ResNet CIFAR-10 (BF16)](/models-tutorials/vision/getting-started-resnet/bf16) | Image classification with BF16 | NPU | Python, C++ |
| [ResNet-50 ImageNet](/models-tutorials/vision/image-classification) | BF16 image classification on ImageNet | NPU | Python |
| [YOLOv8m Object Detection](/models-tutorials/vision/object-detection/yolov8m) | BF16 and XINT8 object detection | NPU | Python |
| [YOLOv8s-WorldV2](/models-tutorials/vision/object-detection/yolov8s-worldv2) | Open-vocabulary object detection | NPU | Python |
| [Torchvision Models](/models-tutorials/vision/torchvision-inference) | Run torchvision models on NPU | NPU | Python, Jupyter |
| [Quark Quantization](/models-tutorials/vision/quark-quantization) | Quantize models with AMD Quark | NPU | Python |
| [ResNet-50 on iGPU](/models-tutorials/vision/igpu-getting-started) | Image classification on integrated GPU | iGPU | Python, C++ |
| [Super Resolution](/models-tutorials/vision/super-resolution) | Real-ESRGAN and SESR-M7 | NPU | - |
| [Stable Diffusion](/models-tutorials/vision/stable-diffusion) | SD 1.5 to 3.5 on iGPU | iGPU | Python |

## LLM Examples

| Example | Description | Device | Language |
|---------|-------------|--------|----------|
| [DistilBERT Text Classification](/models-tutorials/nlp/distilbert) | BF16 text classification | NPU | Python |
| [LLM Fine-Tuning & Deploy](/models-tutorials/llms/llm-sft-deploy) | Fine-tune and deploy LLMs on NPU | NPU | Python |
| [OGA C++ API](/models-tutorials/llms/oga-api) | Native C++ LLM inference | NPU, Hybrid | C++ |
| [OGA Python Inference](/models-tutorials/llms/oga-inference) | Chat with LLMs using OGA Python API | NPU, Hybrid | Python |
| [RAG with OGA](/models-tutorials/llms/rag-oga) | Retrieval-augmented generation | NPU | Python |
| [Vision-Language Model](/models-tutorials/llms/vlm) | Run VLMs with OGA | NPU | Python |

## Audio Examples

| Example | Description | Device | Language |
|---------|-------------|--------|----------|
| [Whisper ASR](/models-tutorials/audio/whisper) | Speech-to-text with Whisper | NPU | Python |

## End-to-End Demos

| Demo | Description | Devices |
|------|-------------|---------|
| [NPU-GPU Pipeline](/models-tutorials/multimodal/npu-gpu-pipeline) | YOLOv8 + RCAN on NPU, Stable Diffusion on iGPU | NPU + iGPU |
