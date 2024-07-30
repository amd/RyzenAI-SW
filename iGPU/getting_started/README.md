# Getting started with iGPU 

This is an example showing how to run the ResNet50 model from PyTorch on AMD's integrated GPU. We will use Olive to convert the model to ONNX format, convert it to FP16 precision, and optimize it. We will then use DirectML execution provider to run the model on the iGPU. 

## Activate Ryzen AI conda environment

Activate the conda environment created by the automatic installer: 

```powershell
conda activate ryzenai-1.2
```

## Install Olive 

```powershell
python -m pip install -r requirements.txt
```

## Install additional dependencies for the example 

```powershell
python -m olive.workflows.run --config resnet50_config.json --setup
```

## Optimize the model using Olive 

```powershell
python -m olive.workflows.run --config resnet50_config.json
```

The optimized models will be available in `./torch_to_onnx-float16_conversion-perf_tuning/.`


## Run the generated model on the iGPU 

```powershell
python predict.py
```

**_NOTE:_**  In predict.py, line 15, the iGPU device ID is enumerated as 0. For PCs with multiple GPUs, you may adjust the device_id to target a specific iGPU.
