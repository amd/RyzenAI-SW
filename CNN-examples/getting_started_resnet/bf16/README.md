<!-- Getting started with RESNET18_BF16 flow test with Python and C++ deployment -->
<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI ResNet BF16 Tutorial </h1>
    </td>
 </tr>
</table>

## Introduction

This tutorial demonstrates the inference workflow using a pre trained ResNet50 model. The focus is to converting the model to bf16 precision using VAIML compiler and offloading it to an NPU using Python/C++ workflows.

## Overview

This Tutorial will help with the steps to deploy RESNET50 model demonstrating:

- Download the RESNET50 model from Hugging Face and export to ONNX format.
- Quantize the model to BF16 using VAIML compiler.
- Compile and run the model on NPU using ONNX runtime with Vitis AI Execution provider using Python/C++ code.

## Requirements

To build the example, the following software must be installed:

- Ryzen AI
- Cmake
- Visual Studio 2022

Please follow the steps given in Installation Instruction to install Ryzen AI, NPU driver and Visual Studio 2022


### Step 1: Install Packages

Ensure that the Ryzen AI Software is correctly installed. For more details, see the installation instructions.
Use the conda environment created during the installation for the rest of the steps. This example requires a couple of additional packages. Run the following command to install them:

```bash
conda create --name resnet_bf16 --clone ryzen-ai-<version>
conda activate resnet_bf16
set RYZEN_AI_INSTALLATION_PATH = <path/to/RyzenAI/installation>

cd <RyzenAI-SW>\CNN-examples\getting_started_resnet\bf16
python -m pip install -r requirements.txt
```

###  Step 2: Download Model and Dataset

The prepare_model_data.py script downloads the CIFAR-10 dataset in pickle format (for python) and binary format (for C++) in data folder. This dataset will be used in the subsequent steps for BF16 compilation and inference. The script also exports the provided PyTorch model into ONNX format in models folder.

```bash
python prepare_model_data.py
```
### Step 3: Model Compilation

```bash
python compile.py --model models\resnet_trained_for_cifar10.onnx
```

Above script will use the config_file and cache_dir to compile model using VAIML(Vitis AI Model Compiler).

```python
            cache_dir = Path(__file__).parent.resolve()
            cache_dir = os.path.join(cache_dir,'my_cache_dir')
            cache_key   = pathlib.Path(onnx_model).stem
            provider_options_dict = {
                "config_file": config_file,
                "cache_dir":   cache_dir,
                "cache_key":   cache_key
            }
```

### Expected output

The expected output after the model compilation

The expected output after the model compilation

```bash
Compilation Complete
(WARNING:71, CRITICAL-WARNING:0, ERROR:0)
[Vitis AI EP] No. of Operators : VAIML   124
[Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU      1
Done
```

## Model Deployment

In this section uses Python APIs for deployment. By default the ``predict.py`` scripts runs on CPU

```bash
   python predict.py
```

Expected output:

```bash
    execution started on CPU
    Image 0: Actual Label cat, Predicted Label cat
    Image 1: Actual Label ship, Predicted Label ship
    Image 2: Actual Label ship, Predicted Label ship
    Image 3: Actual Label airplane, Predicted Label airplane
    Image 4: Actual Label frog, Predicted Label frog
    Image 5: Actual Label frog, Predicted Label frog
    Image 6: Actual Label automobile, Predicted Label truck
    Image 7: Actual Label frog, Predicted Label frog
    Image 8: Actual Label cat, Predicted Label cat
    Image 9: Actual Label automobile, Predicted Label automobile
```

## Model Deployment on NPU

Run the ``predict.py`` with the ``--ep npu`` switch to run the ResNet model on the RyzenAI NPU:

```bash
    python predict.py --ep npu
```

Expected Output:

```bash
    Image 0: Actual Label cat, Predicted Label cat
    Image 1: Actual Label ship, Predicted Label ship
    Image 2: Actual Label ship, Predicted Label ship
    Image 3: Actual Label airplane, Predicted Label airplane
    Image 4: Actual Label frog, Predicted Label frog
    Image 5: Actual Label frog, Predicted Label frog
    Image 6: Actual Label automobile, Predicted Label truck
    Image 7: Actual Label frog, Predicted Label frog
    Image 8: Actual Label cat, Predicted Label cat
    Image 9: Actual Label automobile, Predicted Label automobile
```

For instruction to deploy the model using C++ APIs refer to[ResNet BF16 C++ Deployment](./docs/README_C++.md)
