<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI ResNet Tutorial </h1>
    </td>
 </tr>
</table>

# Introduction

This tutorial demonstrates how to use Windows Machine Learning (WinML) for ONNX model inference using Python. It covers setup, running models, and sample code for image classification using ResNet. This tutorial uses Windows ML APIs to run a ResNet model using Python and C++ examples.

## Overview

This Tutorial will help with the steps to deploy ResNet model demonstrating:

- Setup instructions to create the python environment and install dependencies
- Download the ResNet ONNX model
- (Optional) Quantize the model to INT8 using Olive + AMD Quark for optimal NPU performance
- Compile and run the model on NPU using ONNX runtime with Vitis AI Execution provider using Python/C++ code.

## Setup Instructions

Install the required python packages in the conda environment `winml_env` and Windows Apps SDK using the [Python Setup](../../README.md#python-setup) in the main README.:

```sh
conda create -n winml_resnet --clone winml_env
conda activate winml_resnet
cd <RyzenAI-SW>\WinML\CNN\ResNet\python
pip install --upgrade -r requirements.txt
```

## Download Model

Download the ResNet model using the `download_ResNet.py` script. This downloads the ResNet-50 model in ONNX format.

```sh
cd <RyzenAI-SW>\WinML\CNN\ResNet\model\
python download_ResNet.py
```

## Model Quantization (Optional)

Quantizing ResNet50 to INT8 reduces model size ~4x and improves NPU inference latency by up to 50%.

### Option A: Olive + AMD Quark (Recommended)

This is the recommended approach for best NPU performance. It applies CLE and AdaRound algorithms tuned for AMD hardware.

Download calibration data (no authentication required):

```sh
cd <RyzenAI-SW>\WinML\CNN\ResNet\model\
python setup_calib_data.py
```

Run INT8 quantization using `quantize_resnet` script, which produces `resnet50_i8.onnx` in the model directory.

```sh
python quantize_resnet.py
```

### Option B: AI Toolkit (VS Code Extension)

You can also quantize using the [AI Toolkit for VS Code](https://code.visualstudio.com/docs/intelligentapps/modelconversion) extension:

1. Open `resnet50.onnx` in VS Code with the AI Toolkit extension installed
2. Right-click the model file and select **"Convert Model"**
3. Choose the target platform: **AMD NPU**
4. Select quantization settings: **QDQ INT8**
5. The toolkit generates an optimized model — update the model path in the run command accordingly


### Run Inference

Run inference on NPU (Neural Processing Unit):

```sh
cd <RyzenAI-SW>\WinML\CNN\ResNet\python
python run_model.py --model ..\model\resnet50.onnx --image_path ..\images\dog.jpg --ep_policy NPU 
```

Or simply run with defaults (uses NPU policy, resnet50.onnx model, and all images in images folder):

```sh
python run_model.py --ep_policy NPU
```

If using a quantized model, pass its path via `--model`, as shown below

```sh
cd <RyzenAI-SW>\WinML\CNN\ResNet\python\
python run_model.py --model <path-to-converted-model> --ep_policy NPU
```

### Command-Line Arguments

- `--ep_policy <NPU|CPU|DEFAULT|DISABLE>`: Execution provider policy. Default: NPU
- `--model <path>`: Path to input ONNX model (default: ../model/resnet50.onnx)
- `--compiled_output <path>`: Path for compiled output model (default: ../model/resnet50_ctx.onnx)
- `--image_path <path>`: Path to input image (default: all images in ../images folder)

### Example Output
```
Registering execution providers ...
Registered execution provider: VitisAIExecutionProvider with library path: C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.AMD.NPU.EP.1.8_1.8.25.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_vitisai.dll
Creating session ...
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20251009 12:48:36.467561  4136 vitisai_compile_model.cpp:1263] Vitis AI EP Load ONNX Model Success
I20251009 12:48:36.469228  4136 vitisai_compile_model.cpp:1264] Graph Input Node Name/Shape (1)
I20251009 12:48:36.469748  4136 vitisai_compile_model.cpp:1268]          input : [-1x3x224x224]
I20251009 12:48:36.469833  4136 vitisai_compile_model.cpp:1274] Graph Output Node Name/Shape (1)
I20251009 12:48:36.469993  4136 vitisai_compile_model.cpp:1278]          output : [-1x1000]
Active execution providers (priority order): ['VitisAIExecutionProvider', 'CPUExecutionProvider']
Primary provider (highest priority): VitisAIExecutionProvider
Running inference on image: D:\repos\RyzenAI-SW\tutorial\WinML\images\dog.jpg
Preparing input ...
Running inference ...
Top-5 (softmax probabilities):
  Top-1: golden retriever (id=207, p=0.891560)
  Top-2: Labrador retriever (id=208, p=0.093102)
  Top-3: kuvasz (id=222, p=0.002696)
  Top-4: Chesapeake Bay retriever (id=209, p=0.001279)
  Top-5: tennis ball (id=852, p=0.001126)
```

## Python API Components

### Registration of Execution Provider

The script registers WinML execution providers using `ort.register_execution_provider_library()`:

```python
def register_execution_providers():
    worker_script = str(Path(__file__).parent / 'winml_worker.py')
    result = subprocess.check_output([sys.executable, worker_script], text=True)
    paths = json.loads(result)
    for item in paths.items():
        ort.register_execution_provider_library(item[0], item[1])
```

**Key API**: `ort.register_execution_provider_library(name, path)`
- Registers custom execution provider libraries
- Required for WinML to work with ONNX Runtime
- The worker script discovers the WinML EP library path from the Windows App SDK

### Session Options and Provider Selection

Session options configure how ONNX Runtime executes the model:

```python
session_options = ort.SessionOptions()
policy_enum = ort.OrtExecutionProviderDevicePolicy
session_options.set_provider_selection_policy(selected_policy)
```

**Key APIs**:
- `ort.SessionOptions()`: Creates session configuration object
- `set_provider_selection_policy()`: Sets execution provider selection policy
  - `PREFER_NPU`: Prioritizes Neural Processing Unit
  - `PREFER_CPU`: Prioritizes CPU execution
  - `DEFAULT`: Uses default provider selection

### Model Compilation

Model compilation optimizes the ONNX model for specific hardware:

```python
model_compiler = ort.ModelCompiler(session_options, model_path)
model_compiler.compile_to_file(compiled_model_path)
```

### Inference Session

The inference session is the main interface for running predictions:

```python
session = ort.InferenceSession(model_path, sess_options=session_options)
```

**Key APIs**:
- `ort.InferenceSession(model_path, sess_options)`: Creates inference session
- `session.get_providers()`: Returns list of active execution providers
- `session.get_inputs()`: Returns model input metadata
- `session.get_outputs()`: Returns model output metadata


## For a walkthrough tutorial using C++ please follow:

- [Tutorial for WinML in C++](./cpp/README.md)

## Quantization Configuration

The Olive recipe (`model/olive_config_resnet.json`) uses AMD Quark to quantize ResNet50 to INT8 with the following settings:

### Quantization Scheme

```json
{
  "activation": {
    "data_type": "UInt8",
    "symmetric": true,
    "calibration_method": "MinMSE",
    "quant_granularity": "Tensor",
    "scale_type": "PowerOf2"
  },
  "weight": {
    "data_type": "Int8",
    "symmetric": true,
    "calibration_method": "MinMax",
    "quant_granularity": "Tensor",
    "scale_type": "PowerOf2"
  }
}
```

### Optimization Algorithms

| Algorithm | Purpose | Setting |
|-----------|---------|---------|
| **CLE** (Cross-Layer Equalization) | Redistributes weight ranges across layers to reduce quantization error | 6 steps |
| **AdaRound** (Adaptive Rounding) | Learns optimal weight rounding to minimize output error | 200 iterations |

### NPU-Specific Options

```json
{
  "EnableNPUCnn": true,
  "ConvertOpsetVersion": 21,
  "Int32Bias": false
}
```

- `EnableNPUCnn` — enables AMD NPU-specific CNN optimizations in the quantized model
- `ConvertOpsetVersion` — upgrades the model to ONNX opset 21 for full NPU operator support
- `Int32Bias` — keeps bias in 16-bit (not 32-bit) for better NPU memory efficiency

For further details on tuning the quantization recipe, see the [Olive Documentation](https://microsoft.github.io/Olive/), and [AMD Quark Documentation](https://quark.docs.amd.com/latest/).

## References

- [Olive Documentation](https://microsoft.github.io/Olive/)
- [AMD Quark Documentation](https://quark.docs.amd.com/latest/)
- [AI Toolkit Documentation](https://code.visualstudio.com/docs/intelligentapps/modelconversion)
- [WinML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
- [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/)



