<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI LLM Tutorial </h1>
    </td>
 </tr>
</table>

# Introduction

In this example we will show different ways to deploy LLM  on AMD NPU through `Foundry Local` and 'Windows ML` APIs.

Windows ML enables support for Foundry Local models for on-device AI inference solutions that provide privacy and performance. Currently, Foundry Local is available in preview mode. It automatically detects NPU and downloads the compatible model for the NPU device.

# Prerequisites

Make sure the following requirements are met before proceeding with the tutorial:
- Operating System: Windows 10 (x64), Windows 11 (x64/ARM), Windows Server 2025, macOS.
- Hardware: Minimum 8 GB RAM and 3 GB free disk space. Recommended 16 GB RAM and 15 GB free disk space.
- Network: Internet connection to download the initial model (optional for offline use).
- Acceleration (optional): AMD GPU (6,000 series or newer) or AMD NPU

# Running LLM models on AMD NPU

The LLM models can be run on AMD NPU using Foundry Local or Windows ML APIs. Foundry Local provides an easy-to-use interface for running LLM models on AMD NPU, while Windows ML APIs allow for more customization and control over the inference process. Here are the different options to run LLM models on AMD NPU:

- **Running LLM using Foundry Local**: This is the recommended option for most users as it provides a simple and efficient way to run LLM models on AMD NPU without needing to manage dependencies or optimize the model manually.
- **Running custom LLM model using Windows ML APIs**: This option allows users to run LLM models on AMD NPU using Windows ML APIs. This option is suitable for users who want more control over the inference process and are comfortable managing dependencies and model management manually.
- **Running custom LLM model using Foundry Local**: This option allows users to run LLM models on AMD NPU using Foundry Local. This option is suitable for users who want to bring custom LLM model and use Foundry Local for managing dependencies and model management.


# Running LLM using Foundry Local

## Install Foundry Local

Download and install Foundry Local from the official AMD website: [Foundry Local Download](https://www.amd.com/en/technologies/foundry-local).

You can also open a terminal and run the following command:

```bash
winget install Microsoft.FoundryLocal
```

To upgrade an existing installation of Foundry Local, use the following command:

```bash
winget upgrade --id Microsoft.FoundryLocal
```

```bash
Found an existing package already installed. Trying to upgrade the installed package...
Found Foundry Local [Microsoft.FoundryLocal] Version 0.8.94.38902
This application is licensed to you by its owner.
Microsoft is not responsible for, nor does it grant any licenses to, third-party packages.
This package requires the following dependencies:
  - Packages
      Microsoft.VCLibs.Desktop.14 [>= 14.0.33728.0]
Successfully verified installer hash
Starting package install...
  ████████████████████████████▌   95%
Successfully installed. Restart the application to complete the upgrade.
```

## Run LLM model using Foundry Local

Run the model supported in Foundry local using the following command:

```bash
foundry model run phi-4-mini
```

```bash
Downloading phi-4-mini-instruct-vitis-npu:2...
[####################################] 100.00 % [Time remaining: about 0s]        53.7 MB/s
Unloading existing models. Use --retain true to keep additional models loaded.
🕘 Loading model... 
🟢 Model phi-4-mini-instruct-vitis-npu:2 loaded successfully

Interactive Chat. Enter /? or /help for help.
Press Ctrl+C to cancel generation. Type /exit to leave the chat.

Interactive mode, please enter your prompt
> What is an AI accelerator ?
🧠 Thinking...
🤖 An AI accelerator is specialized hardware designed to enhance the training and inference tasks of AI and machine learning models. Just as GPUs (Graphics Processing Units) are specialized hardware used to improve the training and execution of deep learning tasks, AI accelerators are similarly specialized and optimized hardware designed to provide faster and more energy-efficient training and inference operations for AI tasks.

These accelerators are often integrated into data centers and deployed on cloud platforms, and they are optimized through hardware and algorithms specially designed to process AI workloads more effectively than general-purpose hardware such as CPUs and GPUs. Some common types of AI accelerators include TPUs (Tensor Processing Units) from Google, which are specialized hardware units designed to train and run large-scale machine learning models, and the Apple-designed Apple Neural Engine, which is integrated into their hardware products to provide on-device AI and ML tasks.

The primary benefits of AI accelerators include faster training times, lower energy consumption, and the ability to process large and complex datasets more efficiently, making them an essential tool for modern AI and machine learning applications.
```

This command will download the compatable model, then runs it. It allows user to interact through the CLI interface.

Check a model list supported by Foundry local using the NPU device

```bash
foundry model list --filter device=NPU
```

```bash
🟢 Service is Started on http://127.0.0.1:55051/, PID 15588!
Alias                          Device     Task               File Size    License      Model ID
-----------------------------------------------------------------------------------------------
phi-3-mini-128k                NPU        chat-completion    2.13 GB      MIT          phi-3-mini-128k-instruct-vitis-npu:1
---------------------------------------------------------------------------------------------------------------------------
phi-3-mini-4k                  NPU        chat-completion    2.13 GB      MIT          Phi-3-mini-4k-instruct-vitis-npu:1
-------------------------------------------------------------------------------------------------------------------------
mistral-7b-v0.2                NPU        chat-completion    4.07 GB      apache-2.0   Mistral-7B-Instruct-v0-2-vitis-npu:1
---------------------------------------------------------------------------------------------------------------------------
deepseek-r1-7b                 NPU        chat-completion    5.58 GB      MIT          DeepSeek-R1-Distill-Qwen-7B-vitis-npu:1
------------------------------------------------------------------------------------------------------------------------------
qwen2.5-0.5b                   NPU        chat-completion    0.52 GB      apache-2.0   qwen2.5-0.5b-instruct-vitis-npu:2
------------------------------------------------------------------------------------------------------------------------
qwen2.5-1.5b-instruct-test-vitis-npu NPU        chat-completion    0.63 GB      apache-2.0   qwen2.5-1.5b-instruct-test-vitis-npu:1
-----------------------------------------------------------------------------------------------------------------------------
qwen2.5-coder-0.5b             NPU        chat-completion    0.52 GB      apache-2.0   qwen2.5-coder-0.5b-instruct-vitis-npu:1
------------------------------------------------------------------------------------------------------------------------------
qwen2.5-coder-7b               NPU        chat-completion    4.73 GB      apache-2.0   qwen2.5-coder-7b-instruct-vitis-npu:1
----------------------------------------------------------------------------------------------------------------------------
qwen2.5-coder-1.5b             NPU        chat-completion    1.25 GB      apache-2.0   qwen2.5-coder-1.5b-instruct-vitis-npu:1
------------------------------------------------------------------------------------------------------------------------------
phi-4-mini                     NPU        chat-completion    3.60 GB      MIT          phi-4-mini-instruct-vitis-npu:1
----------------------------------------------------------------------------------------------------------------------
phi-4-mini-reasoning           NPU        chat-completion    2.78 GB      MIT          Phi-4-mini-reasoning-vitis-npu:1
-----------------------------------------------------------------------------------------------------------------------
qwen2.5-7b                     NPU        chat-completion    5.20 GB      apache-2.0   qwen2.5-7b-instruct-vitis-npu:1
```

For more details abou the Foundry Local, please refer to the official documentation: [Foundry Local Documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/get-started)


# Run custom LLM model

In this section, we will show how to run custom LLM models in three steps:
1. Convert and optimize the model using Olive
2. Run with ONNX Generation API (OGA) and Windows ML
3. Run with Foundry Local

## Model conversion with Olive

First, we need to convert and optimize the model using [Olive](https://github.com/microsoft/olive-recipes/). This step will download, quantize, and optimize the model for AMD NPU.

You will need a separate python environment for model conversion due to specific package requirements:
  - Model Quantization with Olive/AMD-Quark: Needs specific older versions (transformers 4.50.0, onnx 1.18.0/1.19.0, standard onnxruntime)

Set up the python environment and install dependencies:

```bash
conda create -n winml_olive python=3.10
conda activate winml_olive
pip install -r requirements_olive.txt
```

Install PyTorch

For Windows (CPU):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For Windows (CUDA):
```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

For Linux:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

For `Phi-4-mini-Instruct` model, download and quantize / optimize the model using [olive recipe](https://github.com/microsoft/olive-recipes/blob/main/microsoft-Phi-4-mini-instruct/VitisAI/Phi-4-mini-instruct_quark_vitisai_llm.json):

```bash
olive run --config Phi-4-mini-instruct_quark_vitisai_llm.json
```

The converted model will be saved in the output directory specified in the Olive configuration file.

## Run custom LLM model with Windows ML and OGA API

After converting the model with Olive, you can run it using Windows ML APIs with ONNX Runtime GenAI. This option is suitable for users who want more control over the inference process and are comfortable managing dependencies manually.

Create a separate environment for running the inference with Windows ML APIs:

```bash
conda create -n winml_llm python=3.11
conda activate winml_llm
pip install -r .\requirements.txt
```

```bash
python run_genai_llm.py --model models\phi-4-mini-instruct-vai-npu --interactive
```

Expected Output:

```bash
============================================================
Registering Execution Providers
============================================================
[INFO] Initializing WinAppSDK 2.0-experimental5...
[INFO] WinAppSDK 2.0-experimental5 initialized

[INFO] Available Execution Providers in WinML catalog:
       (Note: CPU EP is built-in to ONNX Runtime and not shown here)
  1. VitisAIExecutionProvider (Status: NOT_READY)

[INFO] Ensuring VitisAIExecutionProvider (state: 1)...
[INFO] VitisAIExecutionProvider is ready
[INFO] Registered VitisAIExecutionProvider to ONNX GenAI
[INFO] Library path: C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.AMD.NPU.EP.1.8_1.8.51.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_vitisai.dll
C:\Users\dwchenna\github\dwchenna\RyzenAI-SW\WinML\LLM\run_genai_llm.py:347: RuntimeWarning: Shutdown object was not called before being garbage collected.
  if not register_vitisai_ep():

============================================================
Loading model from: models\Phi-4-mini-instruct-vai-npu
============================================================
[INFO] Using VitisAI Execution Provider (configured in genai_config.json)
[INFO] Loading model (this may take a minute)...
[INFO] ✓ Model loaded successfully
[INFO] Creating tokenizer...
[INFO] ✓ Tokenizer created

============================================================
Interactive Mode
============================================================
Type your prompt and press Enter
Type 'quit', 'exit', or 'q' to exit
Press Enter without text for default prompt
============================================================

Prompt: What is AI accelerator?

Response:  An AI accelerator is specialized hardware designed to enhance the efficiency and speed of training and deploying AI and deep learning models. Traditional CPUs may not be optimized for the specific tasks involved in AI, so an AI accelerator takes over these tasks and devotes specialized hardware and specialized hardware
------------------------------------------------------------
```

## Run custom LLM model with Foundry Local

After converting your custom LLM model with Olive, you can use Foundry Local cache to import and run the model.

Change the Foundry cache directory to your model output location:

```bash
foundry cache cd <path-to-your-olive-output-directory>
```

List the available models in the cache:

```bash
foundry cache ls
```

Expected Output:

```
Models cached on device:
   Alias                                             Model ID
💾 DeepSeek-R1-Distill-Qwen-1.5B:1                   DeepSeek-R1-Distill-Qwen-1.5B:1
```

Run your custom model with Foundry:

```bash
foundry model run <your-model-name>
```

Expected Output:
```
🟢 Service is Started on http://127.0.0.1:61301/, PID 8460!
🕑 Downloading complete!...
Successfully downloaded and registered the following EPs: VitisAIExecutionProvider.
Valid EPs: CPUExecutionProvider, WebGpuExecutionProvider, VitisAIExecutionProvider
🕕 Loading model...
🟢 Model DeepSeek-R1-Distill-Qwen-1.5B:1 loaded successfully

Interactive Chat. Enter /? or /help for help.
Press Ctrl+C to cancel generation. Type /exit to leave the chat.

Interactive mode, please enter your prompt
>
```

The model will load and you can interact with it through the CLI interface, similar to running pre-packaged Foundry models.