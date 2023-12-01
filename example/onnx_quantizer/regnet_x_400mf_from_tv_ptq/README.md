# Vitis AI ONNX Quantization Example
This folder contains example code for quantizing a [regnet_x_400mf image classification model](https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py) using VAI_Q_ONNX.

The example has the following parts:

- [Create conda env and Install necessary packages](#Create-conda-env-and-Install-necessary-packages)
- [Prepare the data and the model](#prepare-the-data-and-the-model)
- [Quantize the onnx float model](#quantize-the-onnx-float-model)
- [Evaluate the float model and the quantized model](#evaluate-the-float-model-and-the-quantized-model)


## Create conda env and Install necessary packages
Create conda environment and Install VAI_Q_ONNX from the release wheel package or refer to [README](../../../tutorial/RyzenAI_quant_tutorial/Docs/ONNX_README.md).

Install the necessary python packages:
```
$ conda activate $env_name
$ python -m pip install -r requirements.txt
```

## Prepare the data and the model

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access to ImageNet (ILSVRC) 2012 which is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 50,000 validation images.

If you already have an ImageNet datasets, you can directly use your dataset path.

To prepare the test data, please check the download section of the main website: https://huggingface.co/datasets/imagenet-1k/tree/main/data. You need to register and download **val_images.tar.gz**.

Then, create the validation dataset and calibration dataset:
```
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```
The storage format of the val_data and the calib_data of the ImageNet dataset organized as follows:

- val_data / calib_data
  - n01440764
    - ILSVRC2012_val_00000293.JPEG
    - ILSVRC2012_val_00002138.JPEG
    - ...
  - n01443537
    - ILSVRC2012_val_00000236.JPEG
    - ILSVRC2012_val_00000262.JPEG
    - ...
    - ...
  - n15075141
    - ILSVRC2012_val_00001079.JPEG
    - ILSVRC2012_val_00002663.JPEG
    - ...

Finally, export the pytorch timm model to an onnx float model.
```
mkdir models && python prepare_model.py regnet_x_400mf RegNet_X_400MF_Weights.IMAGENET1K_V1 models/regnet_x_400mf.onnx
```

## Quantize the onnx float model

Quantization tool takes the onnx float32 model and produce a quantized model. For different hardware platforms, distinct quantization settings are required to optimize compatibility with hardware accelerators.
<!-- omit in toc -->
### For IPU:
<!-- omit in toc -->
For IPU, acceleration is achieved using QDQ quantization with uint8 activation, int8 weight types, and power of 2 scale:
```
python quantize_tv_model_ipu.py RegNet_X_400MF_Weights.IMAGENET1K_V1 models/regnet_x_400mf.onnx models/regnet_x_400mf.U8S8.pof2s.onnx calib_data
```
This will generate quantized model using QDQ quant format and uint8 activation type and int8 weight type with pof2s scale to models/regnet_x_400mf.U8S8.pof2s.onnx.
<!-- omit in toc -->
### For CPU:
<!-- omit in toc -->
For CPU, acceleration is achieved using QDQ quantization with uint8 activation, int8 weight types, and float scale:
```
python quantize_tv_model_cpu.py RegNet_X_400MF_Weights.IMAGENET1K_V1 models/regnet_x_400mf.onnx models/regnet_x_400mf.U8S8.fs.onnx calib_data
```
This will generate quantized model using QDQ quant format and uint8 activation type and int8 weight type with float scale to models/regnet_x_400mf.U8S8.fs.onnx.

## Evaluate the float model and the quantized model

Test the accuracy of the float model and the quantized model on ImageNet val dataset with CPUExecutionProvider(default):
```
python onnx_validate.py val_data --weights-name RegNet_X_400MF_Weights.IMAGENET1K_V1 --onnx-float models/regnet_x_400mf.onnx --onnx-quant models/regnet_x_400mf.U8S8.pof2s.onnx
```

Test the accuracy of the float model and the quantized model on ImageNet val dataset with CUDAExecutionProvider:
```
python onnx_validate.py val_data --weights-name RegNet_X_400MF_Weights.IMAGENET1K_V1 --onnx-float models/regnet_x_400mf.onnx --onnx-quant models/regnet_x_400mf.U8S8.pof2s.onnx --gpu
```


|            | Float Model                  | Quantized Model For IPU                      | Quantized Model For CPU          |
|------------|------------------------------|-----------------------------------------|--------------------|
| Model      | models/regnet_x_400mf.onnx | models/regnet_x_400mf.U8S8.pof2s.onnx | models/regnet_x_400mf.U8S8.fs.onnx    |
| Model Size | 21.56 MB                     | 6.02 MB                                 | 5.80 MB   |
| Prec@1     | 72.83 %                      | 71.63 %                                 | 70.03 %   |
| Prec@5     | 90.96 %                      | 90.49 %                                 | 89.40 %   |


<!-- omit in toc -->
## License
<!-- omit in toc -->

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

