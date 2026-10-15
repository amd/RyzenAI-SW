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

- `classification.ipynb`: Jupyter Notebook demonstrating how to grab a classificatioin model from torchvision and port it to run on the Ryzen AI Neural Processing Unit (NPU)
- `classification_utils.py`: This is a Python file contains some sub-functions.
- `README.md`: This file provides an overview of the folder's contents.
- `requirements.txt`: This file contains the necessary dependencies and packages required to run the code in this folder.
- `data\`: This folder conatains an input image with corresponding classes info.
- `vaip_config.json`: This is the default runtime configuration file. It can also be found in the `ryzen-ai-sw` installation package.

## Getting Started

Before running this example, ensure that you have followed the Ryzen AI installation instructions and activated the Ryzen AI environment:

- **Windows:** follow the [Windows installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html) and activate the conda environment created during installation.
- **Linux:** follow the [Linux installation instructions](https://ryzenai.docs.amd.com/en/1.7.1/linux.html) and activate the Python virtual environment created by the installer:

  ```sh
  source /opt/xilinx/xrt/setup.sh
  source <TARGET-PATH>/venv/bin/activate
  ```

>**:pushpin: NOTE:** Do not run `pip install -r requirements.txt` inside the Ryzen AI environment. Its `torch` / `torchvision` pins would replace the versions installed by Ryzen AI, which AMD Quark depends on.

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

### Jupyter Notebook

The notebook runs unchanged on Windows and Linux. It detects the platform and NPU type itself; on Linux the iGPU (DirectML) cell is skipped because DirectML is only available on Windows.

Steps for both platforms:

1. Download and extract the calibration dataset (see [Download the Calibration Dataset](#download-the-calibration-dataset)).
2. Open the notebook and select the Ryzen AI kernel for your platform (see below).
3. Run all the cells in the notebook.

#### Windows

1. Copy the ***vaip_config.json*** file to here from the Ryzen AI installation folder.
2. Launch the Jupyter Notebook.
3. Ensure that you've pointed the Jupyter Notebook to the correct Python environment. To do this, in the top-right corner of the notebook, click "Select Kernel" and provide the path to the conda environment.

>**:pushpin: NOTE (Windows only):** If you can't find the kernel from the top-right dropdown menu, please exit the notebook and install it as below. On Linux, use the kernel setup in the Linux section instead.

```python
python -m ipykernel install --user --name [CONDA_ENV_NAME]
```

#### Linux

`vaip_config.json` is not needed on Linux.

Do not register the kernel with `python -m ipykernel install` on Linux. The Ryzen AI venv does not include `ipykernel`, and a kernel registered that way starts the venv's Python without running `setup.sh` and `activate`. Then `LD_LIBRARY_PATH`, `RYZEN_AI_INSTALLATION_PATH` and the other variables the NPU libraries need are missing, and they cannot be set from inside a running kernel.

Choose one of the following:

- **Jupyter in the browser:** install `ipykernel` and JupyterLab into the venv, then start JupyterLab from the shell where you ran the two `source` commands from Getting Started. Its kernels inherit that environment:

  ```sh
  python -m pip install ipykernel jupyterlab
  jupyter lab
  ```

- **VS Code / Cursor:** these editors start kernels without running `activate`, so register a kernel that runs it first. Run this block once from the shell where you ran the two `source` commands from Getting Started; it fills in your paths automatically:

  ```sh
  python -m pip install ipykernel
  KDIR=~/.local/share/jupyter/kernels/ryzen-ai-1.7.1
  mkdir -p "$KDIR"
  cat > "$KDIR/launch.sh" <<EOF
  #!/usr/bin/env bash
  source /opt/xilinx/xrt/setup.sh >/dev/null
  source "$VIRTUAL_ENV/bin/activate"
  exec python -m ipykernel_launcher "\$@"
  EOF
  chmod +x "$KDIR/launch.sh"
  cat > "$KDIR/kernel.json" <<EOF
  {
    "argv": ["$KDIR/launch.sh", "-f", "{connection_file}"],
    "display_name": "Ryzen AI 1.7.1 (Linux)",
    "language": "python"
  }
  EOF
  ```

  Then reload the editor and select **Ryzen AI 1.7.1 (Linux)** as the notebook kernel.

>**:pushpin: NOTE:** If the notebook's Platform Setup cell reports that `VitisAIExecutionProvider` or `RYZEN_AI_INSTALLATION_PATH` is missing, the kernel was not started from the activated Ryzen AI environment. Select the kernel described above.

## Running the classification.py

>**:pushpin: NOTE:** `classification.py` currently supports Windows only. On Linux, use the notebook.

To run the classification.py script, follow these steps:

**Step 1:** Ensure Environment Setup
Before running the script, ensure you have activated the Ryzen AI conda environment
```sh
conda activate ryzen-ai-<version>
```
**Step 2:** Download the Dataset.
Download and extract the calibration dataset as described in [Download the Calibration Dataset](#download-the-calibration-dataset).

**Step 3:** Run the Quark-Based Example
To execute the hello_world_quark.py script, use the following command:
```sh
python classification.py
```

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
