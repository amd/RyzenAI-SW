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

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:

```bash
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```

## Model Quantization


The 'model_quantization.py' script uses the BF16 quantization configuration from AMD Quark quantizer. It also uses a subset of ImageNet as calibration dataset for quantization.

```python
quant_config = get_default_config("BF16")
quant_config.extra_options["BF16QDQToCast"] = True
config = Config(global_quant_config=quant_config)
print("The configuration of the quantization is {}".format(config))
```
For more details about the BF16 quantization refer to [AMD Quark BF16 Tutorial](https://quark.docs.amd.com/latest/supported_accelerators/ryzenai/tutorial_convert_fp32_or_fp16_to_bf16.html)

The ONNX model can be quantized to BF16 using the 'model_quantization.py' script, as shown:

```bash
python model_quantization.py --model_input models\resnet50.onnx --model_output models\resnet50_bf16.onnx --quantize bf16
```

## Model Evaluation

Evaluate the accuracy of the model using ImageNet dataset on CPU/NPU

```
python image_classification.py --model_input models\resnet50_bf16.onnx --calib_data calib_data --device cpu/npu --evaluate
```

Summary of BF16 model accuracy:

<div align="center">

| ResNet50      | Top-1 Accuracy | Top-5 Accuracy |
|---------------|----------------|----------------|
| Float 32      | 0.800          | 0.961          |
| BF16 (CPU)    | 0.797          | 0.959          |
| BF16 (NPU)    | 0.798          | 0.963          |

</div>

