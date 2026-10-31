<table class="sphinxhide" width="100%">
 <tr width="100%">
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Image Classification </h1>
   </td>
 </tr>
</table>

## Introduction

This tutorial uses a pre-trained ResNet50 model to demonstrate the process of preparing, quantizing, and deploying a BF16 model using Ryzen AI Software. The quantized model is deployed on NPU using Python API and evaluated the accuracy of the quantized model on ImageNet dataset.

## Overview

The following steps outline how to deploy the BF16 model on an NPU:

- Download the ResNet50 model from torchvision library and save it as ONNX (Opset 17) model.
- The compiler automatically converts to BF16 configuration. 
- Compile and run the model on an NPU using ONNX Runtime with the Vitis AI Execution Provider.

## Setup Model and Dataset

Setup Instructions
--------------

Before running this example, install Ryzen AI using the [Ryzen AI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html). The Ryzen AI MSI installer creates a Python virtual environment at the installation location. The commands below assume Ryzen AI version **1.9.0** and Windows Command Prompt (`cmd.exe`). Replace `ryzen-ai-1.9.0` if your installed version differs.

The Ryzen AI MSI installer automatically creates a Python virtual environment at `C:\Program Files\RyzenAI\ryzen-ai-1.9.0`.

Follow the steps below to create a separate virtual environment for the image classification example using the packages installed by Ryzen AI.

- Prerequisite: Install the Ryzen AI MSI installer before proceeding.

- Open x64 Native Tools Command Prompt for VS 2022, Command Prompt, or Miniforge Prompt and execute the following commands.

Step 1: Activate the Ryzen AI installation environment
```bat
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Step 2: Export the installed Python packages
```bat
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```

This creates a requirements file containing the packages installed in the Ryzen AI environment.

Step 3: Create a virtual environment for image classification
```bat
cd /d <RyzenAI-SW>\CNN-examples\image_classification
python -m venv image_classification
```

Step 4: Deactivate the Ryzen AI installation environment
```bat
deactivate
```

Step 5: Activate the image classification environment
```bat
"image_classification\Scripts\activate.bat"
```

Step 6: Install the Ryzen AI Python dependencies
```bat
python -m pip install -r "%TEMP%\ryzenai_1.9_requirements.txt"
```

Step 7: Install the example-specific dependencies
```bat
cd <RyzenAI-SW>\CNN-examples\image_classification
python -m pip install -r requirements.txt
```
The image_classification virtual environment is now configured for running the example.

ResNet50 Model
--------------

Download the ResNet50 model from torchvision models and export it to ONNX model

```bash
cd models
python download_ResNet.py
```

ImageNet Dataset
----------------
If you already have an ImageNet datasets, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/ILSVRC/imagenet-1k/blob/script/data/val_images.tar.gz. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

```bash
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```

## Model Evaluation

Evaluate the accuracy of the model using ImageNet dataset on CPU/NPU

```bash
python image_classification.py --model_input models\resnet50.onnx --calib_data calib_data --device cpu/npu --evaluate
```

Summary of BF16 model accuracy:

<div align="center">

| ResNet50              | Top-1 Accuracy | Top-5 Accuracy |
|-----------------------|----------------|----------------|
| Float 32              | 0.800          | 0.961          |
| FP32 to BF16 (NPU)    | 0.804          | 0.962          |

</div>
