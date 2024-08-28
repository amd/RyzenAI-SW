<!--
Copyright © 2024 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Author: Fan Zhang, AMD-Xilinx
-->

<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> AIG - Technical Marketing </h1>
    </td>
 </tr>
</table>

# Yolov8 Python Implementation with Webcam Input


## Current Status

- Last update:  29 July 2024
- Author:       Fan Zhang
- Release:      NPU driver (32.0.201.204), VOE ryzen-ai-1.2
- Device:       (RyzenAI) HPT

## Contents

This folder contains the following files:

- `yolov8.ipynb`: Jupyter Notebook demonstrating how to grab a classificatioin model from torchvision and port it to run on the Ryzen AI Neural Processing Unit (NPU)
- `yolov8_utils.py`: This is a Python file contains some sub-functions.
- `README.md`: This file provides an overview of the folder's contents.
- `requirements.txt`: This file contains the necessary dependencies and packages required to run the code in this folder.
- `sample_yolov8.jpg`: Input image.
- `coco.names`: Coco datasets classes.
- `anchors.npy`: anchors data of the model.
- `strides.npy`: strides data of the model.
- `vaip_config.json`: This is the default runtime configuration file. It can also be found in the `ryzen-ai-sw` installation package.

## Getting Started

Before running this example, ensure that you have followed the Ryzen AI Installation instructions found [here](https://ryzenai.docs.amd.com/en/latest/inst.html) and that you have activated the conda environment created during installation.

## Running the Example

### Jupyter Notebook

1. Launch the Jupyter Notebook.
2. Ensure that you've pointed the Jupyter Notebook to the correct Python environment. To do this, in the top-right corner of the notebook, click "Select Kernel" and provide the path to the conda environment.
3. Run all the cells in the notebook.

>**:pushpin: NOTE:** If you can't find the kernel from the top-right dropdown menu, please exit the notebook and install it as below:

```
python -m ipykernel install --user --name [CONDA_ENV_NAME]
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
