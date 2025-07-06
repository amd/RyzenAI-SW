# Introduction

In this tutorial we will using yolov8m model for object detection, including preparing, quantization and deploying a `BF16` and `XINT8` model using RyzenAI Software.

## Overview

The following steps outline how to deploy the quantized model on an NPU:

- Download the yolov8m model from the Ultralytics and save it as ONNX (Optset 17) model
- Quantize the model to `BF16` or `XINT8` using the AMD Quark Quantization API
- Compile and run the model on NPU using ONNX Runtime with the Vitis AI Execution Provider


## Download and export the model

Python code to download the model from ultralytics

```python
from ultralytics import YOLO

def export_yolov8m_to_onnx():
    model = YOLO("yolov8m.pt")
    print("Number of classes:", model.model.nc)
    model.export(format="onnx", opset=17)  # Exports to yolov8m.onnx
    print("YOLOv8m exported to yolov8m.onnx")

if __name__ == "__main__":
    export_yolov8m_to_onnx()
```
Command to download and export the model from `.pt` to `.onnx`

```bash
cd models
python export_to_onnx.py
```

*Note:* If asked to update the Ultralytics package, it will upgrade the onnx-runtime. To run on NPU, ensure you create a new clone with existing environment created by RyzenAI software installer.

### Create and Activate Conda Environment

```python
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-<version>
conda create --name yolov8m_env --clone %RYZEN_AI_CONDA_ENV_NAME%
conda activate yolov8m_env
```

Install the required python packages for the tutorial:

```bash
pip install -r requirements.txt
``` 

## Model Quantization and Compilation

Model quantization levarages the power of AMD-Quark to optimize the model for significant performance without losing accuracy.
This tutorial will guide through quantizing model for `BF16` configuration as well as `XINT8` configuration

### BF16 quantization

- **Model Quantization** - Model is quantized using AMD-Quark with `BF16` configuration


```bash
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_BF16.onnx \
                         --config BF16
```

- **Compile and Test Model** - Run model inference on the `test_image.jpg` from the COCO dataset

```bash
python run_inference.py --model_input models\yolov8m_BF16.onnx --input_image test_image.jpg --output_image test_output.jpg --device npu-bf16
```

### Evaluate the model accuracy

The `BF16` quantized model accuracy is evaluated on COCO dataset

Use the `prepare_data.py` script to download the COCO dataset

```bash
python prepare_data.py
```

Evaluate the accuracy of the model on COCO dataset, use `--device` options `cpu` or  `npu-bf16` to measure accuracy metrics on CPU/NPU respectively.

```bash
python run_inference --model_input models\yolov8m_BF16.onnx --evaluate --coco_dataset datasets\coco --device npu-bf16
```

<div align="center">

|  Yolov8m      | mAP (AP@[IoU=0.50:0.95]) |  mAP50 (AP@IoU=0.50) |  mAP75 (AP@IoU=0.75) |
|---------------|--------------------------|----------------------|----------------------|
| Float 32      |       44.0               |     57.4             |      47.9            |
| BF16 (CPU)    |       42.5               |     57.2             |      46.6            |
| BF16 (NPU)    |       42.8               |     57.7             |      46.7            |

</div>


- **Sample Output** - Sample outputs generated using `BF16` model

![test_output_bf16](results/test_output_bf16.png)


### XINT8 quantization

- **Model Quantization** - Model is quantized using AMD-Quark with `XINT8` configuration

```bash
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_XINT8.onnx \
                         --config XINT8
```

- **Compile and Test Model** - Run model inference on the `test_image.jpg` from the COCO dataset

```bash
python run_inference.py --model_input models\yolov8m_XINT8.onnx --input_image test_image.jpg --output_image test_output_int8.jpg --device npu-int8
```

- **Sample Output** - Sample outputs generated using `XINT8` configuration

![test_output_int8](results/test_output_int8.jpg)

### Evaluate quantized model accuracy

The `XINT8` quantized model accuracy is evaluated on COCO dataset

```bash
python run_inference --model_input models\yolov8m_XINT8.onnx --evaluate --coco_dataset datasets\coco --device npu-int8
```
*Note:* The evaluation functions fails to detect any objects

## Modification

The model uses concat operations to combine the `confidence` and `bounding boxes` as shown in the below in yolov8m ONNX model. This leads to significant degradation in confidence values, missing most of the bounding boxes.

![image](results/yolov8m_quantized_concat_node.png)


We need to skip the post-processing sub-graph to improve the accuracy of the `XINT8` quantized model. Shown below in the post-processing sub-graph yolov8m model.

![image](results/yolov8m_skip_nodes.png)


## Model Quantization

After the above modifications model is quantized using AMD-Quark with `XINT8` configuration

```bash
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_XINT8.onnx \
                         --config XINT8
                         --exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_5]]"
```

## Sample Output

Sample outputs generated using `XINT8` quantized model with skipped nodes.

![test_output_int8](results/test_output_int8_skip_nodes.jpg)

### Evaluate the model accuracy

The `XINT8` quantized model accuracy is evaluated on COCO dataset

Use the `prepare_data.py` script to download the COCO dataset

```bash
python prepare_data.py
```

Evaluate the accuracy of the model on COCO dataset, use `--device` options `cpu` or  `npu-int8` to measure accuracy metrics on CPU/NPU respectively.

```bash
python run_inference --model_input models\yolov8m_XINT8.onnx --evaluate --coco_dataset datasets\coco --device npu-int8
```

<div align="center">

|  Yolov8m      | mAP (AP@[IoU=0.50:0.95]) |  mAP50 (AP@IoU=0.50) |  mAP75 (AP@IoU=0.75) |
|---------------|--------------------------|----------------------|----------------------|
| Float 32      |       44.0               |     57.4             |      47.9            |
| XINT8 (CPU)   |       38.2               |     52.3             |      41.8            |
| XINT8 (NPU)   |       38.1               |     52.2             |      41.6            |


</div>