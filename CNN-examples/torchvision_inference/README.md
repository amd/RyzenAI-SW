<!--
Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Tutorial </h1>
    </td>
 </tr>
</table>

# Torchvision Models End-to-End Inference with Ryzen AI

## Contents

This folder contains the following files:

- `classification.ipynb`: Jupyter Notebook demonstrating how to grab a classification model from torchvision and port it to run on the Ryzen AI Neural Processing Unit (NPU)
- `classification_utils.py`: This is a Python file contains some sub-functions.
- `README.md`: This file provides an overview of the folder's contents.
- `requirements.txt`: This file contains the necessary dependencies and packages required to run the code in this folder.
- `data/`: This folder contains an input image with corresponding classes info.
- `vaiml_config_bf16.json`: VAIML config that sets `enable_f32_to_bf16_conversion` — compiles an FP32 ONNX model to BF16 on the NPU.
- `vaiml_config_quark.json`: VAIML config for running a Quark-quantized BF16 ONNX model on the NPU (no float conversion, the model is already BF16).

## Getting Started

Before running this example, ensure that you have followed the Ryzen AI installation instructions and activated the Ryzen AI environment:

- **Windows:** follow the [Windows installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html) and activate the conda environment created during installation.
- **Linux:** follow the [Linux installation instructions](https://ryzenai.docs.amd.com/en/1.7.1/linux.html) and activate the Python virtual environment created by the installer:

  ```sh
  source /opt/xilinx/xrt/setup.sh
  source <TARGET-PATH>/venv/bin/activate
  ```

`requirements.txt` only adds `torchvision==0.19.1+cpu`. It does not install `torch`, `numpy`, `onnx`, or `amd-quark`, because replacing the builds already in the Ryzen AI environment breaks the NPU stack. Ryzen AI 1.7.1 ships `torch` 2.4.1 and does not ship `torchvision`. `0.19.1+cpu` is the `torchvision` build for that `torch`.

From this directory, on either operating system:

```sh
python -m pip install -r requirements.txt
```

## Download the Calibration Dataset

Both the notebook and `classification.py` need ImageNet images to calibrate the quantized model.

To obtain the calibration dataset, visit the ImageNet dataset repository on [Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/script/data).

You need to register on Hugging Face and download the following file:
**val_images.tar.gz**.

This file contains a subset of ImageNet images used specifically for calibration.

The dataset is gated, so an anonymous download (for example a plain `wget`) is refused with `401 GatedRepo`. Once per account:

1. Sign in on Hugging Face and accept the access conditions on the [ImageNet dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
2. Create an access token with read permission at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Then download the file from this folder with one of the following.

- **Hugging Face CLI (Windows and Linux):** `hf` is included with `huggingface_hub` in the Ryzen AI environment. `hf auth login` asks for the token once and stores it. The file is saved as `data/val_images.tar.gz`:

  ```sh
  hf auth login
  hf download ILSVRC/imagenet-1k data/val_images.tar.gz --repo-type dataset --revision script --local-dir .
  ```

- **wget (Linux)**, with the token in the `HF_TOKEN` environment variable:

  ```sh
  wget --header="Authorization: Bearer $HF_TOKEN" -O data/val_images.tar.gz \
       https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/script/data/val_images.tar.gz
  ```

- **curl (Windows 10+ Command Prompt)**, with the token in the `HF_TOKEN` environment variable:

  ```bat
  curl -L -H "Authorization: Bearer %HF_TOKEN%" -o data\val_images.tar.gz https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/script/data/val_images.tar.gz
  ```

If you downloaded the file manually in the browser instead, move it to `data/val_images.tar.gz`.

Then run the following commands to extract it into the `val_images` directory. They work in a Linux shell and in a Windows 10+ Command Prompt or PowerShell:

```sh
mkdir val_images
tar -xzf data/val_images.tar.gz -C val_images
```

The notebook and the script then build the `calib_data` directory from `val_images` automatically.

## Running the classification.ipynb

The notebook runs unchanged on Windows and Linux. It detects the platform and NPU type itself. On Linux the iGPU (DirectML) cell is skipped because DirectML is only available on Windows.

Download and extract the calibration dataset first (see [Download the Calibration Dataset](#download-the-calibration-dataset)). Then, from the activated Ryzen AI environment in this directory, install Jupyter and open the notebook. The same commands work on Windows and Linux:

```sh
python -m pip install ipykernel jupyterlab
jupyter lab classification.ipynb
```

On Linux, run them in the same shell where you sourced `/opt/xilinx/xrt/setup.sh` and activated the installer venv. Jupyter then starts with that environment, including `LD_LIBRARY_PATH` and `RYZEN_AI_INSTALLATION_PATH`.

>**:pushpin: NOTE:** If the notebook's Platform Setup cell reports that `VitisAIExecutionProvider` or `RYZEN_AI_INSTALLATION_PATH` is missing, Jupyter was not started from the activated Ryzen AI environment.

## Running the classification.py

Run the notebook first, from this directory, so it writes `models/resnet50_bf16.onnx` or `models/resnet50_xint8.onnx` and compiles that file into `cache/`. Then run the script from the same directory. It loads that ONNX file and reuses the notebook's NPU cache.

```sh
python classification.py
```

On Linux, run it in the same shell where you sourced `/opt/xilinx/xrt/setup.sh` and activated the installer venv. DirectML iGPU inference is skipped there. If neither notebook model exists, the script quantizes to `models/resnet50_quantized.onnx` and compiles its own cache.

## NPU compilation paths: XINT8 vs BF16

The notebook and `classification.py` currently run the **Quark XINT8** path on the NPU. There are two additional BF16 paths that can be run manually with `run_inference.py` from the [yolov8m example](../object_detection/yolov8m/) or adapted into new scripts. All three paths use the Vitis AI Execution Provider.

### Path 1: Quark XINT8 (current default)

The notebook quantizes the FP32 ONNX model with `get_default_config("XINT8")` and runs it on the NPU. No VAIML config file is passed — the EP uses its default DPU partition pass. This is the path used by `classification.ipynb` and `classification.py`.

### Path 2: Quark BF16 + `vaiml_config_quark.json`

Quantize with Quark using `get_default_config("BF16")` with `BF16QDQToCast=True`. This produces a BF16 ONNX model. Then pass `vaiml_config_quark.json` as the `config_file` provider option to compile it with VAIML:

```python
provider_options = [{
    'config_file': 'vaiml_config_quark.json',
    'cache_dir': '<cache_dir>',
    'cache_key': '<cache_key>'
}]
session = onnxruntime.InferenceSession(
    'model_bf16.onnx',
    providers=['VitisAIExecutionProvider'],
    provider_options=provider_options
)
```

### Path 3: FP32 model + `vaiml_config_bf16.json` (no Quark)

Skip Quark entirely. Pass the original FP32 ONNX model directly with `vaiml_config_bf16.json`, which sets `enable_f32_to_bf16_conversion: true`. VAIML casts the model to BFloat16 during compilation:

```python
provider_options = [{
    'config_file': 'vaiml_config_bf16.json',
    'cache_dir': '<cache_dir>',
    'cache_key': '<cache_key>'
}]
session = onnxruntime.InferenceSession(
    'model_fp32.onnx',
    providers=['VitisAIExecutionProvider'],
    provider_options=provider_options
)
```

Both BF16 paths require a VAIML config file on **both** Windows and Linux. The first VAIML compilation takes several minutes; the compiled model is cached in `cache_dir` and reused on later runs.

See the [yolov8m README](../object_detection/yolov8m/README.md) for a fully worked example with all three paths validated on both platforms.

## License

The MIT License (MIT)

Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


<p align="center"><sup>XD106 | © Copyright 2022 Xilinx, Inc.</sup></p>
