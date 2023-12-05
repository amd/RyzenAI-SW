# Vision Transformers using Multiple-Overlays with RyzenAI

## Introduction

Use two different overlays, CNN (1x4) and GEMM (2x4) to run a model with both
CNN and GEMM Operators of Vision Transformer models.

## Steps to run

- Open Anaconda Command Prompt & Activate conda environment

  ```
  conda activate ryzenai-transformers
  ```

- Refer to the [manual installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html#) to properlly install the Ryzen AI software.


- Setup environment

  ```
  ## On Anaconda Command Prompt, go to transformers root directory
  cd ..\..

  ## Run setup script
  .\setup.bat

  ## Navigate back to models/vision_transformer_onnx
  cd vision-transformer-onnx

  ## Run setup for examples
  .\setup.bat
  ```

- Run model

  :pushpin: Vitis-AI Execution Provider requires models to be Quantized to accelerate them on RyzenAI devices.

  Use `classify.bat` script to run the model

  ```
  ## Check usage
  .\classify.bat --help

  ## With CPU-EP
  .\classify.bat --model <path-to-onnx-model> --ep cpu --img <path-to-image>

  ## With VAI-EP
  .\classify.bat --model <path-to-onnx-model> --img <path-to-image>
  ```

- Run multiple models

  Use `nightly.py` script to run multiple models

  ```
  ## Use below script to run multiple models
  ## This script calls the "classify.bat" for all the models in provided directory
  python nightly.py --models-dir <path-to-models-directory> --img <path-to-image>
  ```
