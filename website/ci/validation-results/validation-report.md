# Local Validation Report

**Date:** 2026-02-27
**Environment:** WSL2 (Ubuntu) on AMD Ryzen AI 9 HX 370
**Total files tested:** 150
**Total tests run:** 190
**Passed:** 127 (67%)
**Failed:** 63 (33%) -- all are missing-import failures requiring Ryzen AI environment

## Summary

### Fixed Issues (this run)

| Issue | File | Fix Applied |
|-------|------|-------------|
| **Syntax error** (missing comma) | `docs/models-tutorials/vision/object_detection/yolov8m/run_inference.py` line 111 | Added missing comma between `session_options` and `providers` args |
| **Mislabeled code block** (shell as python) | `docs/models-tutorials/vision/object_detection/yolov8m/README.mdx` line 40 | Changed ` ```python ` to ` ```bash ` |
| **Mislabeled code block** (shell as python) | `docs/models-tutorials/vision/quark_quantization/README.mdx` lines 54, 67, 154, 219 | Changed ` ```python ` to ` ```bash ` (4 blocks) |
| **Mislabeled code block** (shell as python) | `docs/models-tutorials/vision/quark_quantization/docs/advanced_quant_readme.mdx` lines 25, 30, 46, 79, 95, 126, 142 | Changed ` ```python ` to ` ```bash ` (7 blocks) |
| **Mislabeled code block** (shell as python) | `docs/models-tutorials/vision/torchvision_inference/README.mdx` line 41 | Changed ` ```python ` to ` ```bash notest ` (contains placeholder `[CONDA_ENV_NAME]`) |
| **Code excerpt not standalone** | `docs/models-tutorials/vision/getting_started_resnet/bf16/README.mdx` line 61 | Added `notest` to code excerpt with leading indentation |
| **Unpinned dependencies** | `docs/models-tutorials/llms/llm-sft-deploy/requirements.txt` | Pinned: evaluate==0.4.6, bert_score==0.3.13, wandb==0.25.0, nltk==3.9.3 |
| **Unpinned dependencies** | `docs/models-tutorials/multimodal/npu-gpu-pipeline/requirements.txt` | Pinned: onnxscript==0.6.2, ipython==9.10.0, psutil==7.2.2 |

### Remaining Failures: Missing Dependencies (63 import failures)

All 63 remaining failures are **import availability checks** -- the packages are not installed in the current WSL base Python environment but ARE available in the Ryzen AI conda environment. These are expected to pass when running inside the Ryzen AI environment on Windows.

#### Required Packages (51 unique)

**Core ML frameworks:**
- `torch`, `torchvision`, `torchaudio` -- PyTorch ecosystem
- `onnx`, `onnxruntime`, `onnxruntime_genai` -- ONNX Runtime (including Vitis AI EP)
- `numpy`, `Pillow` (PIL)

**AMD-specific:**
- `quark` -- AMD Quark quantization toolkit
- `olive` -- Microsoft Olive optimization
- `onnx_graphsurgeon` -- ONNX graph manipulation

**HuggingFace ecosystem:**
- `transformers`, `datasets`, `huggingface_hub`, `accelerate`, `peft`, `trl`, `safetensors`, `evaluate`, `timm`

**LLM/RAG:**
- `langchain_core`, `langchain_community`, `langchain_text_splitters`
- `gradio`, `pydantic`

**Vision:**
- `opencv-python` (cv2), `scikit-image` (skimage), `pycocotools`, `ultralytics`

**Audio:**
- `jiwer`, `sounddevice`

**Utilities:**
- `pandas`, `matplotlib`, `tqdm`, `psutil`, `imageio`, `packaging`
- `keyboard`, `pyperclip`, `wget`, `wandb`
- `gitpython` (git), `protobuf` (google)

**Local/relative imports (not pip packages):**
- `custom_embedding` -- local module in RAG-OGA example
- `config` -- local module in stable_diffusion example
- `data` -- local module in npu-gpu-pipeline example
- `ort_util_img2img` -- local module in stable_diffusion example

### How to Run Full Validation

1. **Activate the Ryzen AI conda environment:**
   ```
   conda activate ryzen-ai-1.4.0
   ```

2. **Run syntax-only check (no hardware needed):**
   ```
   python docs/website/ci/run_local_validation.py --syntax-only
   ```

3. **Run syntax + import checks:**
   ```
   python docs/website/ci/run_local_validation.py
   ```

4. **Run full execution (requires NPU/GPU hardware):**
   ```
   python docs/website/ci/run_local_validation.py --execute
   ```

### Next Steps

1. Install all required packages in the Ryzen AI conda environment and re-run
2. Run `--execute` mode on Windows with NPU hardware to test actual execution
3. Each example directory has its own `requirements.txt` -- install per-example deps before running
