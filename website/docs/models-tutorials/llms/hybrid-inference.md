# OnnxRuntime GenAI (OGA) Flow

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# OnnxRuntime GenAI (OGA) Flow

<CIStatus validated={false} />

Ryzen AI Software supports deploying LLMs on Ryzen AI PCs using the native ONNX Runtime Generate (OGA) C++ or Python API. The OGA API is the lowest-level API available for building LLM applications on a Ryzen AI PC. It supports the following execution modes:

- **Hybrid execution mode**: This mode uses both the NPU and iGPU to achieve the best TTFT and TPS during the prefill and decode phases.
- **NPU-only execution mode**: This mode uses the NPU exclusively for both the prefill and decode phases.

## Supported Configurations

The Ryzen AI OGA flow supports Strix and Krackan Point processors. Phoenix (PHX) and Hawk (HPT) processors are not supported.

## Requirements

- Install NPU Drivers and Ryzen AI MSI installer. See [Installation](/getting-started/installation) for more details.
- Install GPU device driver: Ensure GPU device driver https://www.amd.com/en/support is installed
- Install Git for Windows (needed to download models from HF): https://git-scm.com/downloads

## Pre-optimized Models

AMD provides a set of pre-optimized LLMs ready to be deployed with Ryzen AI Software and the supporting runtime for hybrid and/or NPU-only execution. These include popular architectures such as Llama-2, Llama-3, Mistral, DeepSeek Distill models, Qwen-2, Qwen-2.5, Qwen-3, Gemma-2, Gemma-3, GPT-OSS, Phi-3, Phi-3.5, and Phi-4. For the detailed list of supported models, visit [Supported Models](/models-tutorials/llms/supported-models)

- Hugging Face collection of hybrid models: https://huggingface.co/collections/amd/ryzen-ai-17-hybrid-llm
- Hugging Face collection of NPU models: https://huggingface.co/collections/amd/ryzen-ai-17-npu-llm

Each OGA model folder contains a `genai_config.json` file. This file contains various configuration settings for the model. The `session_option` section is where information about specific runtime dependencies is specified.

## Changes Compared to Previous Release

- OGA version is updated to v0.11.2 (Ryzen AI 1.7) from v0.9.2.2 (Ryzen AI 1.6.1).
- For 1.7 release, a new set of hybrid and NPU models is published. Models from earlier releases are not compatible with this version. If you are using Ryzen AI 1.7, please download the updated models.
- Context length upto 4K tokens (combined input and output) is supported for NPU Models. Extended context length more than 4K tokens is supported for Hybrid models.

## Compatible OGA APIs

Pre-optimized hybrid or NPU LLMs can be executed using the official OGA C++ and Python APIs. The current release is compatible with OGA version 0.11.2.
For detailed documentation and examples, refer to the official OGA repository:
https://github.com/microsoft/onnxruntime-genai/tree/rel-0.11.2

## LLMs Test Programs

The Ryzen AI installation includes test programs (in C++ and Python) that can be used to run LLMs and understand how to integrate them in your application.

The steps for deploying the pre-optimized models using the sample programs are described in the following sections.

### Steps to run C++ program and sample python script

#### 1. (Optional) Enable Performance Mode

To run LLMs in best performance mode, follow these steps:

- Go to `Windows` → `Settings` → `System` → `Power`, and set the power mode to **Best Performance**.
- Open a terminal and run:

```bat
cd C:\Windows\System32\AMD
xrt-smi configure --pmode performance
```

#### 2. Activate the Ryzen AI Conda Environment and install `torch` library

Run the following commands:

```bat
conda activate ryzen-ai-<version>
pip install torch==2.7.1
```

This step is required for running the Python script.

:::info
For the C++ program, if you choose not to activate the Conda environment, open a Windows Command Prompt and manually set the environment variable before continuing:

`set RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\<version>`
:::

### C++ Program

Use the `model_benchmark.exe` executable to test LLMs and identify DLL dependencies for C++ applications.

#### 1. Set Up a working directory and copy required Files

```bat
mkdir llm_run
cd llm_run

:: Copy the sample C++ executable
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\model_benchmark.exe" .

:: Copy the sample prompt file
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\amd_genai_prompt.txt" .

:: Copy required DLLs
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\deployment\." .
```

#### 2. Download model from Hugging Face

```bat
:: Install Git LFS if you haven't already: https://git-lfs.com
git lfs install

:: Clone the model repository
git clone https://huggingface.co/amd/Llama-2-7b-chat-hf-onnx-ryzenai-hybrid
```

#### 3. Run `model_benchmark.exe`

```bat
.\model_benchmark.exe -i <path_to_model_dir> -f <prompt_file> -l <list_of_prompt_lengths>

:: Example:
.\model_benchmark.exe -i Llama-2-7b-chat-hf-onnx-ryzenai-hybrid -f amd_genai_prompt.txt -l "1024"
```

### Long Context Support

Ryzen AI now supports token counts up to the model's context length for **hybrid models**. If the total number of tokens exceed 4096, follow the below steps.

**Steps to run long context:**

1. Make the following changes in `genai_config.json` file.

- Add `"hybrid_opt_chunk_context": "1"` under `model.decoder.session_options.provider_options.RyzenAI`.

```json
{
  "model": {
    "bos_token_id": 1,
    "context_length": 16384,
    "decoder": {
      "session_options": {
        "log_id": "onnxruntime-genai",
        "provider_options": [
          {
            "RyzenAI": {
              "external_data_file": "model_jit.pb.bin",
              "hybrid_opt_free_after_prefill": "1",
              "hybrid_opt_max_seq_length": "4096",
              "hybrid_opt_chunk_context": "1"
            }
          }
        ]
      }
    }
  }
}
```

- Add `"chunk_size":2048` under `search`.

```json
"search": {
  "diversity_penalty": 0.0,
  "do_sample": false,
  "chunk_size": 2048,
  ...
}
```

2. Copy the `amd_genai_prompt_long.txt` into your working directory.

```bat
xcopy /Y "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\amd_genai_prompt_long.txt" .
```

3. Run the model using `model_benchmark.exe` using the `amd_genai_prompt_long.txt` prompt file.

```bat
.\model_benchmark.exe -i <path_to_model_dir> -f amd_genai_prompt_long.txt -l "16000"
```

:::info
The sample test application model_benchmark.exe accepts -l for input token length and -g for output token length. In Ryzen AI 1.7, **NPU models** support up to 4096 tokens in total (input + output). By default, -g is set to 128. If the input length is close to 4096, you must adjust -g so the sum of input and output tokens does not exceed 4096. For example, -l 4000 -g 96 is valid (4000 + 96 ≤ 4096), while -l 4000 -g 128 will exceed the limit and result in an error.

For **Hybrid models**, the combined number of input and output tokens must not exceed the model's `context_length`. You can verify the `context_length` in the `genai_config.json` file. For example, if a model's `context_length` is 8,000, the total token count (input + output) must not exceed 8,000.

The long context feature has been tested for hybrid models upto 16,000 tokens.
:::

### Python Script

#### 1. Navigate to your working directory and download model

```bat
:: Install Git LFS if you haven't already: https://git-lfs.com
git lfs install

:: Clone the model repository
git clone https://huggingface.co/amd/Llama-2-7b-chat-hf-onnx-ryzenai-hybrid
```

#### 2. Run sample python script

```bat
python "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\run_model.py" -m <model_folder> -l <max_length>

:: Example command
python "%RYZEN_AI_INSTALLATION_PATH%\LLM\example\run_model.py" -m "Llama-2-7b-chat-hf-onnx-ryzenai-hybrid" -l 256
```

:::info
Some models may return non-printable characters in their output (for example, Qwen models), which can cause a crash while printing the output text. To avoid this, modify the provided script %RYZEN_AI_INSTALLATION_PATH%\LLM\example\run_model.py by adding a text sanitization function and updating the print statement as shown below.

Add sanitize_string function:

```
def sanitize_string(input_string):
   return input_string.encode("charmap", "ignore").decode("charmap")
```

Update line 80 to print sanitized output:

```
print("Output:", sanitize_string(output_text))
```

This sanitization fix will be included in the run_model.py script in the next release.
:::

### Python Script (with Chat Template)

For models that use chat templates, the sample [model_chat.py](/models-tutorials/llms/oga-inference) script may provide better output quality. The script and usage instructions are available in the [RyzenAI-SW repository](/models-tutorials/llms/oga-inference).

This script automatically loads and applies the chat template from the model folder during inference, which can improve output quality for models that use a chat template.

It is highly recommended to use [model_chat.py](/models-tutorials/llms/oga-inference) for the [GPT-OSS-20B NPU model](https://huggingface.co/amd/gpt-oss-20b-onnx-ryzenai-npu).

## Vision Language Model (VLM)

AMD provides a pre-optimized Gemma-3-4b-it multimodal model ready to be deployed with Ryzen AI Software. Support for this model is available starting with the current Ryzen AI 1.7 release.

Model: [Gemma-3-4b-it-mm-onnx-ryzenai-npu](https://huggingface.co/amd/Gemma-3-4b-it-mm-onnx-ryzenai-npu)

VLM inference requires a dedicated Python script. The python script and usage instructions are available in the RyzenAI-SW repository: [VLM](/models-tutorials/llms/vlm)

## Building C++ Applications

A complete example including C++ source and build instructions is available in the RyzenAI-SW repository: [LLM-examples/oga_api](/models-tutorials/llms/oga-api)

## Using Fine-Tuned Models

It is also possible to run fine-tuned versions of the pre-optimized OGA models.

To do this, the fine-tuned models must first be prepared for execution with the OGA flow. For instructions on how to do this, refer to the page about [ONNX Model Preparation](/develop/onnx-model-preparation).

After a fine-tuned model has been prepared for execution, it can be deployed by following the steps described previously in this page.

## Running LLM via pip install

In addition to the full RyzenAI software stack, we also provide standalone wheel files for the users who prefer using their own environment. To prepare an environment for running the Hybrid and NPU-only LLM independently, perform the following steps:

1. Create a new python environment and activate it.

```bash
conda create -n <env_name> python=3.12 -y
conda activate <env_name>
```

2. Install onnxruntime-genai wheel file.

```bash
pip install onnxruntime-genai-directml-ryzenai==0.11.2 --extra-index-url=https://pypi.amd.com/simple
pip install model-generate==1.7.0 --extra-index-url=https://pypi.amd.com/simple
```

3. Navigate to your working directory and download the desired Hybrid/NPU model

```bash
cd working_directory
git clone <link_to_model>
```

4. Run the Hybrid or NPU model.
