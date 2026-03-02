
# Ryzen AI Documentation Map

This page is the single source of truth for how the old Sphinx/RST documentation at [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/) and the old GitHub examples at [github.com/amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW) map to this new Docusaurus/MDX site.

## Repository Structure

```
RyzenAI-SW/
  docs/           # Pure MDX documentation (source of truth for the website)
  examples/       # Runnable code examples with plain README.md
  website/        # Docusaurus infrastructure (config, theme, scripts)
    scripts/      # Build-time scripts (sync-examples.mjs)
    src/          # Theme customizations, components, CSS
    i18n/         # Translation files (11 locales)
  .github/        # CI/CD workflows
```

## How Tutorial Pages Work

Tutorial pages in `docs/` that correspond to runnable code in `examples/` are auto-generated:

```bash
node website/scripts/sync-examples.mjs        # Generate docs from examples
node website/scripts/sync-examples.mjs --check # Verify sync (used in CI)
```

The `examples/*/README.md` files are the source of truth. Edit those, then re-run sync. Do **not** directly edit auto-generated `.mdx` files (they have a comment at the top).

---

## Old RST to New MDX: Complete Migration Map

The old documentation lived in [github.com/amd/ryzen-ai-documentation](https://github.com/amd/ryzen-ai-documentation) (24 RST files) and was published to [ryzenai.docs.amd.com/en/latest/](https://ryzenai.docs.amd.com/en/latest/). Below is every RST file and what happened to it.

### "Getting Started on the NPU" (old section)

| Old RST File | Old Published URL | New MDX File | Status |
|---|---|---|---|
| `index.rst` | [ryzenai.docs.amd.com/en/latest/](https://ryzenai.docs.amd.com/en/latest/) | `getting-started/overview.mdx` | **Replaced** with "Overview and Architecture" |
| `inst.rst` | [ryzenai.docs.amd.com/en/latest/inst.html](https://ryzenai.docs.amd.com/en/latest/inst.html) | `getting-started/installation.mdx` | **Migrated** (manual install merged in) |
| `examples.rst` | [ryzenai.docs.amd.com/en/latest/examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) | `models-tutorials/index.mdx` | **Merged** into Models and Tutorials landing |
| `relnotes.rst` | [ryzenai.docs.amd.com/en/latest/relnotes.html](https://ryzenai.docs.amd.com/en/latest/relnotes.html) | `reference/changelog/index.mdx` + `getting-started/supported-hardware.mdx` | **Split** into Changelog and Supported Hardware |

### "Running Models on the NPU" (old section)

| Old RST File | Old Published URL | New MDX File | Status |
|---|---|---|---|
| `model_quantization.rst` | [ryzenai.docs.amd.com/en/latest/model_quantization.html](https://ryzenai.docs.amd.com/en/latest/model_quantization.html) | `develop/model-quantization.mdx` | **Migrated** |
| `modelrun.rst` | [ryzenai.docs.amd.com/en/latest/modelrun.html](https://ryzenai.docs.amd.com/en/latest/modelrun.html) | `develop/model-deployment.mdx` | **Migrated** |
| `app_development.rst` | [ryzenai.docs.amd.com/en/latest/app_development.html](https://ryzenai.docs.amd.com/en/latest/app_development.html) | `develop/app-development.mdx` | **Migrated** |
| `whisper_cpp.rst` | [ryzenai.docs.amd.com/en/latest/whisper_cpp.html](https://ryzenai.docs.amd.com/en/latest/whisper_cpp.html) | `models-tutorials/audio/whisper/index.mdx` | **Migrated** under Audio |
| `getstartex.rst` | [ryzenai.docs.amd.com/en/latest/getstartex.html](https://ryzenai.docs.amd.com/en/latest/getstartex.html) | `models-tutorials/vision/cnn-examples.mdx` | **Migrated** as "ResNet INT8 Tutorial" |

### "Running LLMs on the NPU" (old section)

| Old RST File | Old Published URL | New MDX File | Status |
|---|---|---|---|
| `llm/overview.rst` | [ryzenai.docs.amd.com/en/latest/llm/overview.html](https://ryzenai.docs.amd.com/en/latest/llm/overview.html) | `models-tutorials/llms/overview.mdx` | **Migrated** (now LLM landing page) |
| `llm/server_interface.rst` | [ryzenai.docs.amd.com/en/latest/llm/server_interface.html](https://ryzenai.docs.amd.com/en/latest/llm/server_interface.html) | `models-tutorials/llms/server-interface.mdx` | **Migrated** |
| `llm/high_level_python.rst` | [ryzenai.docs.amd.com/en/latest/llm/high_level_python.html](https://ryzenai.docs.amd.com/en/latest/llm/high_level_python.html) | `models-tutorials/llms/python-api.mdx` | **Migrated** |
| `hybrid_oga.rst` | [ryzenai.docs.amd.com/en/latest/hybrid_oga.html](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html) | `models-tutorials/llms/hybrid-inference.mdx` | **Migrated** |
| `oga_model_prepare.rst` | [ryzenai.docs.amd.com/en/latest/oga_model_prepare.html](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html) | `develop/onnx-model-preparation.mdx` | **Migrated** to Develop |
| `oga_op_prepare.rst` | [ryzenai.docs.amd.com/en/latest/oga_op_prepare.html](https://ryzenai.docs.amd.com/en/latest/oga_op_prepare.html) | `develop/operator-preparation.mdx` | **Migrated** to Develop |
| `llm_linux.rst` | (not in public sidebar) | `models-tutorials/llms/linux-setup.mdx` | **Migrated** |
| `hybrid_oga_pip_install_draft.rst` | (unpublished draft) | N/A | **Deleted** |

### "Running Models on the GPU" (old section)

| Old RST File | Old Published URL | New MDX File | Status |
|---|---|---|---|
| `gpu/ryzenai_gpu.rst` | [ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html](https://ryzenai.docs.amd.com/en/latest/gpu/ryzenai_gpu.html) | `develop/rocm-client-gpu.mdx` | **Migrated** as "DirectML Flow (GPU)" |

### "Additional Topics" (old section)

| Old RST File | Old Published URL | New MDX File | Status |
|---|---|---|---|
| `xrt_smi.rst` | [ryzenai.docs.amd.com/en/latest/xrt_smi.html](https://ryzenai.docs.amd.com/en/latest/xrt_smi.html) | `tools/npu-management.mdx` | **Migrated** |
| `ai_analyzer.rst` | [ryzenai.docs.amd.com/en/latest/ai_analyzer.html](https://ryzenai.docs.amd.com/en/latest/ai_analyzer.html) | `tools/ai-analyzer.mdx` | **Migrated** |
| `sd_demo.rst` | [ryzenai.docs.amd.com/en/latest/sd_demo.html](https://ryzenai.docs.amd.com/en/latest/sd_demo.html) | `models-tutorials/vision/stable-diffusion.mdx` | **Migrated** under Vision |
| `ryzen_ai_libraries.rst` | [ryzenai.docs.amd.com/en/latest/ryzen_ai_libraries.html](https://ryzenai.docs.amd.com/en/latest/ryzen_ai_libraries.html) | `develop/cvml-library.mdx` | **Migrated** to Develop |
| `ops_support.rst` | [ryzenai.docs.amd.com/en/latest/ops_support.html](https://ryzenai.docs.amd.com/en/latest/ops_support.html) | `reference/supported-operators.mdx` | **Migrated** |
| `licenses.rst` | [ryzenai.docs.amd.com/en/latest/licenses.html](https://ryzenai.docs.amd.com/en/latest/licenses.html) | `reference/licenses.mdx` | **Migrated** |
| `model_list.rst` | [ryzenai.docs.amd.com/en/latest/model_list.html](https://ryzenai.docs.amd.com/en/latest/model_list.html) | `reference/model-list.mdx` | **Migrated** |

---

## Tutorial and Example Pages: Full Lineage

These pages originate from example code that lived in the **old GitHub repo** ([github.com/amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW)). Some were also linked from pages on the **old docs site** ([ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/)). They have been reorganized into `examples/` with auto-generated `.mdx` docs pages.

### Vision Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| ResNet INT8 Tutorial | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [getstartex.html](https://ryzenai.docs.amd.com/en/latest/getstartex.html) (Getting Started Tutorial) | `examples/vision/getting_started_resnet/` | `vision/cnn-examples.mdx` (hand-written) |
| ResNet Getting Started | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [getstartex.html](https://ryzenai.docs.amd.com/en/latest/getstartex.html) | `examples/vision/getting_started_resnet/` | `vision/getting-started-resnet/index.mdx` (auto-gen) |
| ResNet INT8 Quantization | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [getstartex.html](https://ryzenai.docs.amd.com/en/latest/getstartex.html) | `examples/vision/getting_started_resnet/int8/` | `vision/getting-started-resnet/int8/index.mdx` (auto-gen) |
| ResNet BF16 Tutorial | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (BF16 Model Examples) | `examples/vision/getting_started_resnet/bf16/` | `vision/getting-started-resnet/bf16/index.mdx` (auto-gen) |
| ResNet BF16 C++ Deployment | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) | `examples/vision/getting_started_resnet/bf16/docs/` | `vision/getting-started-resnet/bf16/cpp-deployment.mdx` (auto-gen) |
| Hello World Tutorial | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Hello world jupyter notebook) | `examples/vision/hello_world/` | `vision/hello-world/index.mdx` (auto-gen) |
| iGPU Getting Started | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Getting started on iGPU) | `examples/vision/iGPU/getting_started/` | `vision/igpu-getting-started/index.mdx` (auto-gen) |
| Image Classification | [Transformer-examples/](https://github.com/amd/RyzenAI-SW/tree/main/Transformer-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (BF16 Image classification) | `examples/vision/image_classification/` | `vision/image-classification/index.mdx` (auto-gen) |
| YOLOv8m Object Detection | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Object detection with Yolov8) | `examples/vision/object_detection/yolov8m/` | `vision/object-detection/yolov8m/index.mdx` (auto-gen) |
| YOLOv8s-WorldV2 | [CNN-examples/](https://github.com/amd/RyzenAI-SW/tree/main/CNN-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Object detection with Yolov8) | `examples/vision/object_detection/yolov8s-worldv2/` | `vision/object-detection/yolov8s-worldv2/index.mdx` (auto-gen) |
| Super Resolution | Not in old repo | New | `examples/vision/super-resolution/` | `vision/super-resolution/index.mdx` (auto-gen) |
| Quark Quantization | Not in old repo | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (AMD Quark Quantization) | `examples/vision/quark_quantization/` | `vision/quark-quantization/index.mdx` (auto-gen) |
| Advanced Quantization | Not in old repo | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) | `examples/vision/quark_quantization/docs/` | `vision/quark-quantization/advanced.mdx` (auto-gen) |
| CVML Library Tutorial | [Ryzen-AI-CVML-Library/](https://github.com/amd/RyzenAI-SW/tree/main/Ryzen-AI-CVML-Library) | [ryzen_ai_libraries.html](https://ryzenai.docs.amd.com/en/latest/ryzen_ai_libraries.html) + [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) | `examples/vision/cvml/` | `vision/cvml/index.mdx` (auto-gen) |
| Torchvision Inference | Not in old repo | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Torchvision models) | `examples/vision/torchvision_inference/` | `vision/torchvision-inference/index.mdx` (auto-gen) |
| Stable Diffusion Demo | [Demos/](https://github.com/amd/RyzenAI-SW/tree/main/Demos) | [sd_demo.html](https://ryzenai.docs.amd.com/en/latest/sd_demo.html) | N/A (hand-written page) | `vision/stable-diffusion.mdx` (hand-written) |

### LLM Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| OGA C++ API | [LLM-examples/](https://github.com/amd/RyzenAI-SW/tree/main/LLM-examples) | [hybrid_oga.html](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html) (OGA Flow) | `examples/llms/oga_api/` | `llms/oga-api/index.mdx` (auto-gen) |
| OGA Inference (Python) | [LLM-examples/](https://github.com/amd/RyzenAI-SW/tree/main/LLM-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Running GPT-OSS-20B with chat template) | `examples/llms/oga_inference/` | `llms/oga-inference/index.mdx` (auto-gen) |
| Fine-tune and Deploy LLMs | [LLM-examples/](https://github.com/amd/RyzenAI-SW/tree/main/LLM-examples) | New | `examples/llms/llm-sft-deploy/` | `llms/llm-sft-deploy/index.mdx` (auto-gen) |
| RAG with OGA | [LLM-examples/](https://github.com/amd/RyzenAI-SW/tree/main/LLM-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (OGA-based RAG LLM) | `examples/llms/RAG-OGA/` | `llms/rag-oga/index.mdx` (auto-gen) |
| Vision-Language Models (VLM) | [LLM-examples/](https://github.com/amd/RyzenAI-SW/tree/main/LLM-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Running VLM on RyzenAI NPU) | `examples/llms/VLM/` | `llms/vlm/index.mdx` (auto-gen) |

### Audio Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| Running Whisper on Ryzen AI | [Demos/](https://github.com/amd/RyzenAI-SW/tree/main/Demos) | [whisper_cpp.html](https://ryzenai.docs.amd.com/en/latest/whisper_cpp.html) + [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) | `examples/audio/whisper/` | `audio/whisper/index.mdx` (auto-gen) |

### Multimodal Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| NPU-GPU Pipeline | [Demos/](https://github.com/amd/RyzenAI-SW/tree/main/Demos) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (NPU-GPU pipeline on RyzenAI) | `examples/multimodal/npu-gpu-pipeline/` | `multimodal/npu-gpu-pipeline/index.mdx` (auto-gen) |

### NLP Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| DistilBERT Text Classification | [Transformer-examples/](https://github.com/amd/RyzenAI-SW/tree/main/Transformer-examples) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (Finetuned DistilBERT) | `examples/nlp/distilbert/` | `nlp/distilbert/index.mdx` (auto-gen) |

### Tool Tutorials

| Title | Old GitHub Location | Old Docs Site Reference | New `examples/` Location | New Docs MDX |
|---|---|---|---|---|
| NPU Benchmark Tool | [onnx-benchmark/](https://github.com/amd/RyzenAI-SW/tree/main/onnx-benchmark) | [examples.html](https://ryzenai.docs.amd.com/en/latest/examples.html) (ONNX benchmark utilities) | `examples/tools/benchmarking/` | `tools/benchmarking/index.mdx` (auto-gen) |
| NPU Check Utilities | [utilities/npu_check/](https://github.com/amd/RyzenAI-SW/tree/main/utilities/npu_check) | New | `examples/tools/npu-check/` | `tools/npu-check/index.mdx` (auto-gen) |

---

## New Pages (no old RST or GitHub equivalent)

| New MDX File | Title | Notes |
|---|---|---|
| `getting-started/quickstart.mdx` | Quickstart | New content |
| `applications/index.mdx` | Applications | New: AMD + third-party apps |
| `models-tutorials/index.mdx` | Models and Tutorials | New landing (incorporates old `examples.rst`) |
| `models-tutorials/audio/index.mdx` | Audio Models | New category overview |
| `models-tutorials/audio/supported-models.mdx` | Audio: Supported Models | New model list |
| `models-tutorials/llms/index.mdx` | Large Language Models | New category overview |
| `models-tutorials/llms/supported-models.mdx` | Supported LLMs | New: detailed HuggingFace table |
| `models-tutorials/multimodal/index.mdx` | Multimodal Models | New category overview |
| `models-tutorials/multimodal/supported-models.mdx` | Multimodal: Supported Models | New model list |
| `models-tutorials/vision/index.mdx` | Vision Models | New category overview |
| `models-tutorials/vision/supported-models.mdx` | Vision: Supported Models | New model list |
| `models-tutorials/nlp/index.mdx` | NLP Models | New category overview |
| `develop/index.mdx` | Develop and Tools | New landing page |
| `tools/index.mdx` | Tools | New landing page |
| `reference/index.mdx` | Reference | New landing page |

---

## Complete Current Sidebar (alphabetical within sections)

### Getting Started (landing: Overview and Architecture)

- Installation
- Quickstart
- Supported Hardware

### Applications (landing: Applications index)

### Models and Tutorials (landing: Models and Tutorials index)

**Audio** (click goes to Audio Models overview)
- Audio: Supported Models
- Running Whisper on Ryzen AI

**Large Language Models** (click goes to LLM Deployment Overview)
- DistilBERT Text Classification
- Fine-tune and Deploy LLMs
- High-Level Python SDK
- OGA C++ API
- OGA Inference (Python)
- OnnxRuntime GenAI (OGA) Flow
- RAG with OGA
- Running LLM on Linux
- Server Interface (REST API)
- Supported LLMs
- Vision-Language Models (VLM)

**Multimodal** (click goes to Multimodal: Supported Models)
- NPU-GPU Pipeline

**Vision** (click goes to Vision Models overview)
- Hello World Tutorial
- iGPU Getting Started
- Image Classification
- ResNet BF16 C++ Deployment
- ResNet BF16 Tutorial
- ResNet INT8 Tutorial
- Stable Diffusion Demo
- Super Resolution
- Vision: Supported Models
- YOLOv8m Object Detection
- YOLOv8s-WorldV2

### Develop and Tools (landing: Develop and Tools index)

- Advanced Quantization
- AI Analyzer
- Application Development
- CVML Library (Tutorial)
- DirectML Flow (GPU)
- Model Compilation and Deployment
- Model Quantization
- NPU Benchmark Tool
- NPU Check Utilities
- NPU Management Interface
- ONNX Model Preparation
- Operator Preparation
- Quark Quantization (Tutorial)
- Ryzen AI CVML Library
- Torchvision Inference

### Reference (landing: Reference index)

- Changelog
- Licensing Information
- Model Table
- Supported Operators

---

## Pages Removed or Merged

| Page | What Happened |
|---|---|
| `examples.rst` (old docs site) | **Merged** into `models-tutorials/index.mdx` |
| `hybrid_oga_pip_install_draft.rst` (old docs site) | **Deleted** (unpublished draft) |
| `models-tutorials/examples.mdx` (new site) | **Merged** into `models-tutorials/index.mdx` |
| `getting-started-resnet/index.mdx` (new site) | **Removed from sidebar** (thin wrapper, only linked to INT8/BF16) |
| `getting-started-resnet/int8/index.mdx` (new site) | **Removed from sidebar** (no substantive content) |

---

## Old GitHub Repo Structure to New Structure

The old [github.com/amd/RyzenAI-SW](https://github.com/amd/RyzenAI-SW) repo had these top-level folders for examples:

| Old GitHub Folder | What it Contained | New Location |
|---|---|---|
| `CNN-examples/` | ResNet INT8/BF16, Hello World, iGPU, YOLOv8 | `examples/vision/` (various subfolders) |
| `Transformer-examples/` | DistilBERT, Image Classification | `examples/nlp/distilbert/` + `examples/vision/image_classification/` |
| `LLM-examples/` | OGA API, OGA Inference, RAG, VLM, SFT Deploy | `examples/llms/` (various subfolders) |
| `Demos/` | NPU-GPU Pipeline, Whisper | `examples/multimodal/npu-gpu-pipeline/` + `examples/audio/whisper/` |
| `Ryzen-AI-CVML-Library/` | CVML samples (face detection, face mesh) | `examples/vision/cvml/` |
| `onnx-benchmark/` | ONNX benchmark tool | `examples/tools/benchmarking/` |
| `utilities/npu_check/` | NPU check utility | `examples/tools/npu-check/` |

---

## i18n (Translations)

Translation scaffolding exists for 11 locales. English fallback content is pre-populated at `website/i18n/{locale}/docusaurus-plugin-content-docs/current/`.

| Locale | Language |
|---|---|
| `en` | English (default) |
| `zh-Hans` | Chinese (Simplified) |
| `ja` | Japanese |
| `ko` | Korean |
| `pt-BR` | Portuguese (Brazil) |
| `es` | Spanish |
| `hi` | Hindi |
| `de` | German |
| `fr` | French |
| `ru` | Russian |
| `uk` | Ukrainian |

The Docusaurus dev server only serves one locale at a time. To test translations locally, run `npx docusaurus start --locale fr`. The production build generates all locales.
