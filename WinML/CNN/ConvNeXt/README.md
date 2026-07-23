<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Windows ML ConvNeXt Example </h1>
    </td>
 </tr>
</table>

# Introduction

This tutorial demonstrates how to use Windows Machine Learning (WinML) for ONNX model inference using Python. It covers setup, running models, and sample code for image classification using ConvNeXt. This tutorial uses Windows ML APIs to run a ConvNeXt model using Python examples.

## Overview

This Tutorial will help with the steps to deploy ConvNeXt model demonstrating:

- Setup instructions to create the python environment and install dependencies
- Download the ConvNeXt ONNX model
- (Optional) Quantize the model to INT8 using Olive + AMD Quark for optimal NPU performance
- Compile and run the model on NPU using ONNX runtime with Vitis AI Execution provider using Python code.

## Setup Instructions

Complete the [Python Setup](../../README.md#python-setup) in the main README first, then clone that environment for this example and install its requirements:

```sh
conda create -n winml_convnext --clone winml_env
conda activate winml_convnext
pip install --upgrade -r requirements.txt
```

## Download Model

Download the ConvNeXt model using the `download_ConvNeXt.py` script. This downloads the ConvNeXt-Small model in ONNX format.

```sh
cd <RyzenAI-SW>\WinML\CNN\ConvNeXt\model\
python download_ConvNeXt.py
```

## Model Quantization (Optional)

Quantizing ConvNeXt-Small to INT8 reduces model size ~4x and improves NPU inference latency by up to 50%.

### Option A: Olive + AMD Quark (Recommended)

This is the recommended approach for best NPU performance. It applies CLE and AdaRound algorithms tuned for AMD hardware.

Download calibration data (no authentication required):

```sh
cd <RyzenAI-SW>\WinML\CNN\ConvNeXt\model\
python setup_calib_data.py
```

Run INT8 quantization using `quantize_convnext.py`, which produces `convnext_small_i8.onnx` in the model directory.

```sh
python quantize_convnext.py
```

For a full walkthrough including calibration data options, configuration details, and troubleshooting, see the [AMD Quark Documentation](https://quark.docs.amd.com/latest/).

### Option B: Foundry Toolkit (VS Code Extension)

You can also quantize using the [Foundry Toolkit for VS Code](https://code.visualstudio.com/docs/intelligentapps/modelconversion) extension:

1. Open `convnext_small.onnx` in VS Code with the Foundry Toolkit extension installed
2. Right-click the model file and select **"Convert Model"**
3. Choose the target platform: **AMD NPU**
4. Select quantization settings: **QDQ INT8**
5. The toolkit generates an optimized model — update the model path in the run command accordingly

### Run Inference

Run inference on NPU (Neural Processing Unit):

```sh
cd <RyzenAI-SW>\WinML\CNN\ConvNeXt
python run_model.py --model model\convnext_small.onnx --image_path images\dog.jpg --ep_policy NPU
```

Or simply run with defaults (uses NPU policy, convnext_small.onnx model, and all images in images folder):

```sh
python run_model.py --ep_policy NPU
```

If using a quantized model, pass its path via `--model`, as shown below

```sh
cd <RyzenAI-SW>\WinML\CNN\ResNet\python\
python run_model.py --model <path-to-converted-model> --ep_policy NPU
```

### Command-Line Arguments

- `--ep_policy <NPU|CPU|DEFAULT>`: Execution provider policy. Default: NPU
- `--model <path>`: Path to input ONNX model (default: model/convnext_small.onnx)
- `--compiled_output <path>`: Path for compiled output model (default: model/convnext_small_ctx.onnx)
- `--image_path <path>`: Path to input image (default: all images in images folder)

### Example Output

Input Image:

![Input Image](./images/dog.jpg)

Output:
```
Registering execution providers ...
Registered execution provider: VitisAIExecutionProvider with library path: C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.AMD.NPU.EP.1.8_1.8.51.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_vitisai.dll
Creating session ...
Set provider selection policy to: NPU
Active execution providers (priority order): ['VitisAIExecutionProvider', 'CPUExecutionProvider']
Primary provider (highest priority): VitisAIExecutionProvider
Running inference on image: images\dog.jpg
Preparing input ...
Running inference ...
Top-5 (softmax probabilities):
  Top-1: golden retriever (id=207, p=0.793348)
  Top-2: Labrador retriever (id=208, p=0.018658)
  Top-3: Sussex spaniel (id=220, p=0.009382)
  Top-4: cocker spaniel (id=219, p=0.002773)
  Top-5: Irish setter (id=213, p=0.002688)
```

## Quantization Configuration

The Olive recipe (`model/olive_config_convnext.json`) uses AMD Quark to quantize ConvNeXt-Small to INT8 with the following settings:

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
- `Int32Bias` — keeps bias in 16-bit for better NPU memory efficiency

For further details on tuning the quantization recipe, see the [Olive Documentation](https://microsoft.github.io/Olive/), and [AMD Quark Documentation](https://quark.docs.amd.com/latest/).

## References

- [Olive Documentation](https://microsoft.github.io/Olive/)
- [AMD Quark Documentation](https://quark.docs.amd.com/latest/)
- [AI Toolkit Documentation](https://code.visualstudio.com/docs/intelligentapps/modelconversion)
- [WinML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/)
- [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/)
