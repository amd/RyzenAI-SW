<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI ResNet INT8 Tutorial </h1>
    </td>
 </tr>
</table>

# Getting Started ResNet with INT8 Quantization & NPU Deployment

This tutorial demonstrates how to prepare, quantize, and deploy a fine-tuned **ResNet-50** model on CIFAR-10 using **AMD Quark** and **Ryzen AI Software 1.8** with ONNX Runtime on the NPU.

---

## ⚠️ Important Notice on Issue #385 & Dependency Isolation

In **Ryzen AI 1.8**, AMD unbundled the Quark quantization toolkit from the default installation. Running `pip install` for Quark inside the active `ryzen-ai-1.8.0` environment creates severe Python package dependency conflicts:
- `flexml 1.8.0` strictly requires `torch==2.6.0+cpu`, `torchvision==0.21.0+cpu`, and `onnx==1.21.0`.
- AMD Quark requires newer or unpinned versions of `torch`, `torchvision`, and `onnx`.
- Installing both into the same environment breaks `flexml`, causing NPU execution (`predict.py --ep npu`) to fail.

### MLOps Best Practice: Decoupled Build vs. Runtime Pipeline

```
┌────────────────────────────────────────────────────────┐
│            Phase 1: Build & Quantization               │
│  (Docker / Google Cloud Console / Isolated Conda)      │
│                                                        │
│  - Export ResNet PyTorch checkpoint -> ONNX            │
│  - AMD Quark Calibration & INT8 Quantization           │
│  - Output Artifact: models/resnet_quantized.onnx       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           │ Transfer Artifact
                           ▼
┌────────────────────────────────────────────────────────┐
│             Phase 2: Edge NPU Serving                  │
│             (Local Ryzen AI 1.8 Host)                  │
│                                                        │
│  - ONNX Runtime with VitisAIExecutionProvider (flexml) │
│  - Minimal runtime dependencies (OpenCV, Pillow)       │
│  - Zero pollution of flexml pinned packages            │
└──────────────────────────┘
```

---

## Option A: Docker / Google Cloud Console (Recommended)

Using Docker keeps the quantization pipeline 100% reproducible and isolated from your host OS. You can run this in **Google Cloud Console (Cloud Shell / GCP VM)** or **Docker Desktop**.

### Step 1: Run Quantization in Docker / GCP Cloud Shell
Navigate to this directory and run:

**On Linux / Google Cloud Console / WSL:**
```bash
chmod +x docker_run.sh
./docker_run.sh
```

**On Windows (Docker Desktop):**
```cmd
docker_run.bat
```

**Or run Docker manually:**
```bash
# 1. Build the image
docker build -t ryzenai-resnet-quark:latest -f Dockerfile .

# 2. Run the container with volume mounts
docker run --rm \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/data:/app/data" \
    ryzenai-resnet-quark:latest
```

This will automatically:
1. Download CIFAR-10 data and the pre-trained ResNet-50 checkpoint (via Git LFS URL).
2. Export `models/resnet_trained_for_cifar10.onnx`.
3. Quantize the model using Quark into `models/resnet_quantized.onnx`.

### Step 2: Transfer Quantized Model to Ryzen AI Host
If you ran Docker in Google Cloud Console:
- Download the generated file `models/resnet_quantized.onnx` to your local machine (place it in `CNN-examples/getting_started_resnet/int8/models/`).

---

## Option B: Dual Conda Environments (Local Windows without Docker)

If you prefer running locally on Windows without Docker, use two separate Conda environments.

### Step 1: Quantization Environment (Quark)
```cmd
# Create an isolated environment for Quark
conda create -n quark_env python=3.10 -y
conda activate quark_env

# Install quantization dependencies
pip install -r requirements_quantize.txt

# Export model to ONNX & Quantize
python prepare_model_data.py
python resnet_quantize.py

# Deactivate when done
conda deactivate
```

---

## Phase 2: Inference & Deployment on Ryzen AI NPU

Now that `models/resnet_quantized.onnx` is ready, run inference using your official `ryzen-ai-1.8.0` environment:

### Step 1: Activate Ryzen AI Environment
```cmd
conda activate ryzen-ai-1.8.0
```

### Step 2: Install Runtime Inference Requirements
```cmd
pip install -r requirements_infer.txt
```
*(Notice: This only installs `opencv-python` and `Pillow`, keeping `flexml 1.8.0` completely intact!)*

### Step 3: Run Inference

**On NPU:**
```cmd
python predict.py --ep npu
```

**On CPU (Fallback):**
```cmd
python predict.py --ep cpu
```

### Step 4: C++ Inference (Optional)
Navigate to `cpp/resnet_cifar`:
```cmd
cd cpp/resnet_cifar
mkdir build && cd build
cmake .. -A x64 -DCMAKE_PREFIX_PATH="%RYZEN_AI_INSTALLATION_PATH%"
cmake --build . --config Release
cd Release
resnet_cifar.exe -m ..\..\..\models\resnet_quantized.onnx -d ..\..\..\data\cifar-10-batches-bin
```

---

## File Overview

| File | Purpose |
| :--- | :--- |
| `Dockerfile` | Container configuration for offline Quark quantization |
| `docker_run.sh` / `docker_run.bat` | One-click script to build & run Docker quantization |
| `requirements_quantize.txt` | Dependencies for Model Export & Quark INT8 quantization |
| `requirements_infer.txt` | Dependencies for NPU runtime inference on Ryzen AI host |
| `prepare_model_data.py` | Downloads CIFAR-10 data, pre-trained weights, and exports ONNX |
| `resnet_quantize.py` | Runs AMD Quark calibration and produces `resnet_quantized.onnx` |
| `predict.py` | Runs inference on Ryzen AI NPU using ONNX Runtime VitisAI EP |
| `resnet_utils.py` | Helper functions for NPU device detection and paths |
