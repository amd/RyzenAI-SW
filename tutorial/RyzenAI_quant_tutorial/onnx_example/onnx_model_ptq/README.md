# Vitis AI ONNX Quantization Example
This folder contains example code for quantizing a [Resnet50-v1-12 image classification model](https://github.com/onnx/models/blob/main/vision/classification/reset/model/resnet50-v1-12.onnx) using vai_q_onnx.
The example has the following parts:

1. Prepare data and model
2. Quantization
3. Evaluation

## Pip requirements
Install the necessary python packages:
```
python -m pip install -r requirements.txt
```

## Prepare data and model
Dataset Summary

ILSVRC 2012, commonly known as 'ImageNet'. This dataset provides access to ImageNet (ILSVRC) 2012 which is the most commonly used subset of ImageNet. This dataset spans 1000 object classes and contains 50,000 validation images.

To Prepare the test data, please check the download section of the main website:
https://huggingface.co/datasets/imagenet-1k/tree/main/data

You need to register and download val_images.tar.gz,

Then, create the validation dataset and calibration dataset.

```
mkdir val_data && tar -xzf val_images.tar.gz -C val_data
python prepare_data.py val_data calib_data
```
The storage format of the val_data and the calib_data of the ImageNet dataset organized as follows:

- Dataset Root Directory
  - Category 1 Folder
    - Image1.jpg
    - Image2.jpg
    - ...
  - Category 2 Folder
    - Image1.jpg
    - Image2.jpg
    - ...
    - ...
  - Category N Folder
    - Image1.jpg
    - Image2.jpg
    - ...

If you already have an ImageNet datasets with the above structure, you can also directly use your dataset path instead of val_data.

Finally, download onnx model from the link.
```
mkdir models && wget https://github.com/onnx/models/blob/main/vision/classification/resnet/model/resnet50-v1-12.onnx
```

## Quantization

Quantization tool takes the pre-processed float32 model and produce a quantized model.

For CPU config:
```
python quantize_onnx_model_cpu.py models/resnet50-v1-12.onnx models/resnet50-v1-12.S8S8.fs.onnx calib_data
```
This will generate quantized model using QDQ quant format and Int8 activation type and Int8 weight type with float scale to models/resnet50-v1-12.S8S8.fs.onnx

For IPU config:
```
python quantize_onnx_model_ipu.py models/resnet50-v1-12.onnx models/resnet50-v1-12.S8S8.pof2s.onnx calib_data
```
This will generate quantized model using QDQ quant format and Int8 activation type and Int8 weight type with pof2s scale to models/resnet50-v1-12.S8S8.pof2s.onnx

## Evaluation

Test the accuracy of a float model on ImageNet val dataset (Prec@1 76.130 Prec@5 92.862):
```
python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.onnx
```

For a float model, you can use GPU to accelerate the evaluation.
```
python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.onnx --gpu
```

Test the accuracy of a CPU quantized config model on ImageNet val dataset (Prec@1 73.882 Prec@5 91.716):
```
python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.S8S8.fs.onnx
```

Test the accuracy of a IPU quantized config model on ImageNet val dataset (Prec@1 75.560 Prec@5 92.588):
```
python onnx_validate.py val_data --onnx-input models/resnet50-v1-12.S8S8.pof2s.onnx

|            | Float Model                  | Quantized Model For IPU                      | Quantized Model For CPU          |
|------------|------------------------------|-----------------------------------------|--------------------|
| Model      | models/resnet50-v1-12.onnx   | models/resnet50-v1-12.U8S8.pof2s.onnx   | models/resnet50-v1-12.U8S8.fs.onnx    |
| Model Size | 97.82 MB                     | 24.48 MB                                | 24.96 MB   |
| Prec@1     | 74.11 %                      | 73.52 %                                 | 72.55 %   |
| Prec@5     | 91.71 %                      | 91.29 %                                 | 90.95 %   |

```

## License

Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

