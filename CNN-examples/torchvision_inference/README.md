<!--
Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

<table class="sphinxhide" width="100%">
  <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Tutorial </h1></td>
  </tr>
</table>

# Torchvision Models End-to-End Inference with Ryzen AI

## Contents

This folder contains the following files:

- `classification.ipynb`: Jupyter Notebook demonstrating how to obtain a classification model from Torchvision and run it on the Ryzen AI Neural Processing Unit (NPU).
- `classification_utils.py`: Python helper functions used by the example.
- `README.md`: Instructions for setting up and running the example.
- `requirements.txt`: Python dependencies required for this example.
- `data/`: Sample input image and corresponding class information.
- `vaip_config.json`: Default runtime configuration file, also available in the Ryzen AI installation package.

## Getting Started

Before running this example, install Ryzen AI using the [Ryzen AI installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html). The Ryzen AI MSI installer creates a Python virtual environment at the installation location. The commands below assume Ryzen AI version **1.9.0** and Windows Command Prompt (`cmd.exe`). Replace `ryzen-ai-1.9.0` if your installed version differs.

**Step 1: Activate the Ryzen AI installation virtual environment**

```bash
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

**Step 2: Navigate to the example directory**

```bash
cd /d "<RyzenAI-SW>\CNN-examples\torchvision_inference"
```

Replace `<RyzenAI-SW>` with the path to your local repository checkout.

**Step 3: Install the example-specific dependencies**

```bash
python -m pip install -r requirements.txt
```

**Step 4: Verify the active Python interpreter**

```bash
python -c "import sys; print(sys.executable)"
```

The interpreter should be located under `C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\`.

**Note:** This example uses the Ryzen AI installer-provided virtual environment directly; it does not require creating a separate environment. Installing additional packages may modify the installer environment.

## Running the classification.ipynb

### Jupyter Notebook

1. Copy `vaip_config.json` from the Ryzen AI installation folder into this example directory if the file is not already present.
2. With the Ryzen AI virtual environment activated, install Jupyter and the kernel package if needed:

   ```bash
   python -m pip install notebook ipykernel
   ```

3. Register the environment as a Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name ryzen-ai-1.9.0 --display-name "Python (Ryzen AI 1.9.0)"
   ```

4. Launch Jupyter Notebook from the example directory:

   ```bash
   python -m notebook
   ```

5. Open `classification.ipynb`. In the top-right corner, select **Select Kernel** and choose **Python (Ryzen AI 1.9.0)**.
6. Run all cells in the notebook.

> **📌 NOTE:** If the kernel is missing from the dropdown, close Jupyter Notebook, rerun the `ipykernel` installation and registration commands above, restart Jupyter, and select the registered kernel. You can confirm the notebook interpreter by running `import sys; print(sys.executable)` in a cell.

## Running the classification.py

To run the `classification.py` script, follow these steps. Make sure `classification.py` is present in your working checkout; it is referenced by the upstream README but is not included in its file inventory.

**Step 1: Ensure Environment Setup**

Activate the Ryzen AI installation virtual environment and set the installation path:

```bash
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
cd /d "<RyzenAI-SW>\CNN-examples\torchvision_inference"
```

**Step 2: Download the Dataset**

To obtain the calibration dataset, visit the [ImageNet dataset repository on Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/script/data). You may need to register and accept the dataset access terms. Download `val_images.tar.gz` and place it in your working directory.

Extract the archive into the `val_images` directory:

```bash
mkdir val_images
tar -xzf val_images.tar.gz -C val_images
```

**Step 3: Run the Classification Script**

```bash
python classification.py
```

**Note:** The upstream README refers to `hello_world_quark.py` in this step but shows `python classification.py` as the command. The command above retains the actual command shown in the README. Verify that the script exists in your checkout before running it.

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
