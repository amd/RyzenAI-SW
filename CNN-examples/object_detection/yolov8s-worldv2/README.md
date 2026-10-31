<table class="sphinxhide" width="100%">
 <tr width="100%">
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Object Detection with Yolov8s-Worldv2 </h1>
   </td>
 </tr>
</table>

# Introduction

In this part we will using yolov8s-worldv2 model for object detection, including preparing, quantization and deploying a high precision model using RyzenAI Software.

## Overview

The following steps outline how to deploy the quantized model on an NPU:

- Download the yolov8s-worldv2 model from the Ultralytics and export it to ONNX model
- Quantize the model to `a16w8_adaround` using the AMD Quark Quantization API
- Compile and run the model on NPU using ONNX Runtime with the Vitis AI Execution Provider

## Set Up and Activate Python Virtual Environment

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

Step 3: Create a virtual environment for YOLOv8s-Worldv2

```bat
cd <RyzenAI-SW>\CNN-examples\object_detection\yolov8s-worldv2
python -m venv yoloworld_env
```

Step 4: Deactivate the Ryzen AI installation environment

```bat
deactivate
```

Step 5: Activate the YOLOv8s-Worldv2 environment

```bat
"yoloworld_env\Scripts\activate.bat"
```

Step 6: Install the Ryzen AI Python dependencies

```bat
python -m pip install -r "%TEMP%\ryzenai_1.9_requirements.txt"
```

Install the other required python packages

```bash
pip install opencv-python
pip install pycocotools
pip install torch
pip install ultralytics
pip install --force-reinstall numpy==1.26.4
```

## Prepare dataset

This model was trained by COCO dataset, you can download and install it from official website.  
https://cocodataset.org/#download
After install it, you need to modify 'env.py' so that coco path is consistent.

## Download and export the model

Origin pth model link:
https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt

```bash
mkdir models
.\download.bat
```

Export onnx:

```bash
python .\ultra_yolo_to_onnx.py --pt-model .\models\yolov8s-worldv2 --input-size 640
```

The exported onnx model will be saved to 'models/yolov8s-worldv2.onnx'.

## Onnx model quantization

We will quantize it with exclude post-process. The post-process part will retained in onnx model with this method, which could simplify the complexity of our deployment.

```bash
python quark_quant.py --onnx models\yolov8s-worldv2.onnx --quant A16W8_ADAROUND -exclude-post
```

## Eval onnx model

We can test the mAP of origin onnx model and quantized model

(CPU)
```bash
python eval_on_coco.py --model models\yolov8s-worldv2.onnx --device cpu
```
(NPU)
```bash
python eval_on_coco.py --model models\yolov8s-worldv2-A16W8_ADAROUND-640x640-exclude-post --device npu
```

## Test model performance

(CPU)
```bash
python infer_single.py --model models\yolov8s-worldv2.onnx --image .\images\test.jpg --runtime-seconds 60 --device cpu
```
(NPU)
```bash
python infer_single.py --model .\models\yolov8s-worldv2-A16W8_ADAROUND-640x640-exclude-post.onnx --image .\images\test.jpg --device npu --runtime-seconds 60
```

<div align="center">

|    Yolov8m      | mAP (AP@[IoU=0.50:0.95]) |  mAP50 (AP@IoU=0.50) |  mAP75 (AP@IoU=0.75) |
|-----------------|--------------------------|----------------------|----------------------|
| Float 32        |       35.9               |     49.7             |      39.0            |
| quantized model |       35.6               |     49.6             |      38.7            |
| NPU E2E mAP     |       35.6               |     49.5             |      38.7            |

</div>
