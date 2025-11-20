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
- Quantize the model to BF16 using the AMD Quark Quantizer.
- Compile and run the model on an NPU using ONNX Runtime with the Vitis AI Execution Provider.

## Setup Model and Dataset

Setup Instructions
--------------

Activate the conda environment created by the RyzenAI installer

```bash
conda create --name image_classification --clone ryzen-ai-<version>
conda activate image_classification
python -m pip install -r requirements.txt
```

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

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/script/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

```bash
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```

## Model Evaluation

Evaluate the accuracy of the model using ImageNet dataset on CPU/NPU

```bash
python image_classification.py --model_input models\resnet50_bf16.onnx --calib_data calib_data --device cpu/npu --evaluate
```

Summary of BF16 model accuracy:

<div align="center">

| ResNet50              | Top-1 Accuracy | Top-5 Accuracy |
|-----------------------|----------------|----------------|
| Float 32              | 0.800          | 0.961          |
| FP32 to BF16 (NPU)    | 0.804          | 0.962          |

</div>

