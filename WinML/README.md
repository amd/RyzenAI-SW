<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI WinML Tutorial </h1>
    </td>
 </tr>
</table>

# Introduction

Windows Machine Learning (WinML) enables developers to run ONNX AI models on PC via ONNX runtime, with automatic execution provider management for different hardwares i.e. CPUs, GPUs and NPUs. For more details [Microsoft Windows ML Documentation](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview)

WinML is a lightweight, efficient AI model runtime designed for dynamic execution and broad hardware compatibility.

Key Features:
-  Dynamic Loading - Automatically fetches latest execution providers (EPs) at runtime
-  Shared ONNX Runtime - Reduces application size by eliminating redundant dependencies
-  Optimized Distribution - Smaller downloads and streamlined installations
-  Broader Hardware Support - Seamless compatibility across different vendors and device types

In this document, we discuss how to enable AMD hardware through WinML APIs. This tutorial uses Windows ML APIs to run CNN, Transformer and LLM models on AMD NPU.

## System Requirements

- Windows 11 with a supported AMD Ryzen AI NPU and the current NPU driver
- Python 3.12 for the Python examples (no Conda or Miniforge required)
- Visual Studio 2022 or newer with the **Desktop development with C++** workload for the C++ example
- CMake 3.23+ on PATH — for the C++ example
- A compatible Windows App SDK runtime (see the setup sections below)

# Model Support

The VitisAI EP within WinML supports input models in the following formats:

  - CNN Models
    - Original float (FP32) model with automatically converted to BF16 during compilation
    - Quantized QDQ model using A8W8 configuration
  - Transformer Models
    - Original float (FP32) model with automatically converted to BF16 during compilation
    - Quantized QDQ model using A16W8 configuration
  - LLM Models:
    - Quantized and pre-compiled LLM models
    - Support for custom models through Olive recipe and Windows ML + OGA APIs

# Python Setup

Install the required python packages in the python virtual environment `winml_env`

Step 1: Activate the Ryzen AI installation environment
```bash
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Step 2: Export the installed Python packages
```bash
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```

Step 3: Create a virtual environment for winml_env
```bash
py -3.12 -m venv "%USERPROFILE%\winml_env"
```

Step 4: Deactivate the Ryzen AI installation environment
```bash
deactivate
```

Step 5: Activate the example environment
```bash
"%USERPROFILE%\winml_env\Scripts\activate.bat"
```

Step 6: Install environment dependencies
```bash
python -m pip install --upgrade pip
```

Step 7: Install the example-specific dependencies
```bash
cd <RyzenAI-SW>\WinML
python -m pip install -r requirements.txt
```

The `wasdk` pip packages require a matching **Windows App SDK runtime** installed on the
machine. Check the installed pip version:

```sh
python -m pip show wasdk
```

Then install the Windows App SDK runtime whose version matches that `wasdk` version from the
official [Windows App SDK downloads](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/downloads)
page (choose the runtime installer for your architecture, e.g. x64).

> **Note:** All stable 2.x runtimes share the `Microsoft.WindowsAppRuntime.2` package family.
> A newer runtime (e.g. 2.2.0) supersedes an older one (e.g. 2.1.3) in place and remains
> compatible with the `wasdk` pip package.


# C++ Setup

Each `CMakeLists.txt` fetches its NuGet packages automatically on the first configure (pinned
versions).

> If `cmake` isn't found, run from the **Developer PowerShell for VS**, which puts the
> VS-bundled CMake on PATH.

See the [ResNet C++ README](./CNN/ResNet/cpp/README.md) for the build and run commands.

# WinML examples

For detailed step by step tutorials:

- [Getting Started ResNet Tutorial](./CNN/ResNet/README.md)
- [Transformer Tutorial using Google BERT](./Transformers/GoogleBert/README.md)
- [LLM Examples](./LLM/README.md)

# References

- [Windows ML Documentation](https://learn.microsoft.com/en-us/windows/ai/windows-ml/)
- [Foundry Toolkit (VS Code)](https://code.visualstudio.com/docs/intelligentapps/modelconversion)
- [Windows App SDK](https://learn.microsoft.com/en-us/windows/apps/windows-app-sdk/)
