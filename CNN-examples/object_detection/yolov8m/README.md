<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Object Detection with Yolov8m </h1>
    </td>
 </tr>
</table>

# Introduction

In this tutorial we deploy a YOLOv8m object detector on the Ryzen AI NPU. Two quantization paths are supported:

- **XINT8** — INT8 quantization via AMD Quark. Higher throughput, slight accuracy loss.
- **BF16** — BFloat16 quantization via AMD Quark with `BF16QDQToCast`. Comparable throughput to XINT8 but closer to FP32 accuracy. VAIML automatically compiles the entire model for NPU (100% operator offload).

## Overview

The following steps outline how to deploy the model on an NPU:

- Export `yolov8m.pt` to ONNX (opset 17)
- Quantize with AMD Quark — either `XINT8` or `BF16`
- Compile and run on the NPU with ONNX Runtime and the Vitis AI Execution Provider


## Environment

Use the Ryzen AI environment from the installer. `requirements.txt` only adds `pycocotools`, `wget`, and `onnxsim`. It does not install `torch`, `numpy`, `opencv`, or `ultralytics`, because replacing the builds already in the Ryzen AI environment breaks the NPU stack.

### Windows

```bat
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-<version>
conda activate %RYZEN_AI_CONDA_ENV_NAME%
cd <RyzenAI-SW>\CNN-examples\object_detection\yolov8m
pip install -r requirements.txt
```

### Linux

Activate the installer venv first. That sets `RYZEN_AI_INSTALLATION_PATH` to the venv directory. Then load XRT, which adds the NPU libraries to `LD_LIBRARY_PATH`. This is the order in the [Linux installation instructions](https://ryzenai.docs.amd.com/en/latest/linux.html).

```sh
source <Ryzen-AI-install>/venv/bin/activate
source /opt/xilinx/xrt/setup.sh
cd <RyzenAI-SW>/CNN-examples/object_detection/yolov8m
python -m pip install -r requirements.txt
```

## Download and export the model

Download the PyTorch model [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt) from Ultralytics. Use the python script to export the model from `yolov8m.pt` to `yolov8m.onnx`. The script assumes the `yolov8m.pt` is in the models directory. 

```python
cd models
python export_to_onnx.py
```

## Option A: Quantize and run XINT8

Quark `XINT8` is the same configuration used by `CNN-examples/torchvision_inference`: `get_default_config("XINT8")`, with no `BF16QDQToCast`. The NPU session is created with the Vitis AI Execution Provider and cache options only. No VAIML config file is passed.

Quantizing the whole graph, including the head that concatenates boxes and class scores, drops detections. Exclude that subgraph. In `models/yolov8m.onnx` it starts at `/model.22/Concat_3` and ends at `/model.22/Concat_10`. There is no `/model.22/Concat_5` in this file.

```bash
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_XINT8.onnx \
                         --config XINT8 \
                         --exclude_subgraphs "[/model.22/Concat_3], [/model.22/Concat_10]"
```

Run it on `test_image.jpg`. Use forward slashes in the model path so the command is the same in bash and in the Windows shell.

```bash
python run_inference.py --model_input models/yolov8m_XINT8.onnx --input_image test_image.jpg --output_image test_output_int8.jpg --device npu-int8
```

The run writes `test_output_int8.jpg`.

![test_output_int8](test_output_int8.jpg)

## Option B: Quantize and run BF16

The BF16 path uses Quark's `BF16` config with `BF16QDQToCast` enabled, which inserts Cast ops instead of QuantizeLinear/DequantizeLinear pairs. VAIML then automatically casts the FP32 model to BFloat16 and compiles the entire graph for the NPU — no subgraph exclusions are needed.

### Quantize to BF16

```bash
python quantize_quark.py --input_model_path models/yolov8m.onnx \
                         --calib_data_path calib_images \
                         --output_model_path models/yolov8m_BF16.onnx \
                         --config BF16
```

Note: unlike XINT8, **no `--exclude_subgraphs` is needed** for BF16. The entire model, including the detection head, compiles and runs correctly on the NPU.

### Run BF16 inference

The BF16 path requires a VAIML config file (`vaiml_config.json`, included in this directory) which tells the Vitis AI EP to use the VAIML partition pass. The `run_inference.py` script selects this config automatically when `--device npu-bf16` is specified.

```bash
python run_inference.py --model_input models/yolov8m_BF16.onnx --input_image test_image.jpg --output_image test_output_bf16.jpg --device npu-bf16
```

On first run, VAIML compiles the model for the NPU (this may take a few minutes). The compilation cache is stored in `modelcachekey/` and reused on subsequent runs. The VAIML compilation summary reports:

- **925 operators — 100% supported by VAIML**
- **82.3 GOPs — 100% offloaded to AIE (NPU)**
- **1 single subgraph** — the entire model runs as one NPU partition with no CPU-NPU data transfers in between
- **0 unsupported ops**

### Evaluate the model accuracy

Use the `prepare_data.py` script to download the COCO dataset

```bash
python prepare_data.py
```

Evaluate the accuracy of the model on COCO dataset. Use `--device` options `cpu`, `npu-int8`, or `npu-bf16` to measure accuracy metrics on CPU/NPU respectively.

```bash
python run_inference.py --model_input models/yolov8m_XINT8.onnx --evaluate --coco_dataset datasets/coco --device npu-int8
python run_inference.py --model_input models/yolov8m_BF16.onnx --evaluate --coco_dataset datasets/coco --device npu-bf16
```

<div align="center">

|  Yolov8m      | mAP (AP@[IoU=0.50:0.95]) |  mAP50 (AP@IoU=0.50) |  mAP75 (AP@IoU=0.75) |
|---------------|--------------------------|----------------------|----------------------|
| Float 32      |       44.0               |     57.4             |      47.9            |
| XINT8 (CPU)   |       38.2               |     52.3             |      41.8            |
| XINT8 (NPU)   |       38.1               |     52.2             |      41.6            |


</div>


## Evaluate the Model Performance across formats

Evaluate the Performance of Model across Float32 (CPU), BF16 and XINT8. Model is benchmarked across 100 inference runs and we measure  
1) Latency - Average time for each Inference run on CPU/NPU
2) Frames per Second - Indicates how many individual images, or frames, are inferred per second 

Use `--benchmark` option while running the Model

```bash
python run_inference.py --model_input models/yolov8m_BF16.onnx --input_image test_image.jpg --output_image test_output_bf16.jpg --device npu-bf16 --benchmark
python run_inference.py --model_input models/yolov8m_XINT8.onnx --input_image test_image.jpg --output_image test_output_int8.jpg --device npu-int8 --benchmark
```


<div align="center">

|  Yolov8m      | Avg time for each Inference (Latency) |  Frames per Second (FPS)| 
|---------------|---------------------------------------|-------------------------|
| Float32 (CPU) |       0.154 sec                       |     6.5                 |
| BF16    (NPU) |       0.029 sec                       |     34.1                |
| XINT8   (NPU) |       0.026 sec                       |     37.8                |

</div>
