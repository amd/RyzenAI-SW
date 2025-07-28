# Ryzen AI Hybrid: meta-llama/Meta-Llama-3-8B (int4)

This guide contains all of the instructions necessary to get started with the model [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on Ryzen AI Hybrid in the int4 data type.

Hybrid execution mode optimally partitions the model such that different operations are scheduled on NPU vs. iGPU. This minimizes time-to-first-token (TTFT) in the prefill-phase and maximizes token generation (tokens per second, TPS) in the decode phase.

The commands and scripts in this guide leverage the [Lemonade SDK](https://github.com/lemonade-sdk/lemonade), which provides everything you need to get up and running with LLMs on the OnnxRuntime GenAI (OGA) framework.

# Checkpoint

The Ryzen AI Hybrid implementation of [`meta-llama/Meta-Llama-3-8B`](https://huggingface.co/meta-llama/Meta-Llama-3-8B) uses the [`amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid`](https://huggingface.co/amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid) checkpoint on Hugging Face. This checkpoint has been quantized for int4 with AMD Quark using the recipe provided on the model card.

# Setup

To get started with the [Lemonade SDK](https://github.com/lemonade-sdk/lemonade) in a Python environment, follow these instructions.

### System-level pre-requisites

You only need to do this once per computer:

1. Make sure your system has the recommended Ryzen AI driver installed as described in the [Ryzen AI Installation Guide](https://ryzenai.docs.amd.com/en/latest/inst.html#install-npu-drivers).
1. [Download and install miniconda for Windows](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe). 
1. Launch a terminal and call `conda init`

### Lemonade Installation

To create and set up an environment, run these commands in your terminal:

1. Create environment.
    ```bash
    conda create -n ryzenai-llm python=3.10
    ```

2. Activate environment.
    ```bash
    conda activate ryzenai-llm
    ```

3. Install the Lemonade SDK to get access to the LLM tools and APIs.
    ```bash
    pip install lemonade-sdk[llm-oga-hybrid]
    ```

4. Install support for Ryzen AI Hybrid LLMs.
    ```bash
    lemonade-install --ryzenai hybrid
    ```

# Validation Tools

You can use the following `lemonade` CLI commands to test out your model, in your activated `ryzenai-llm` environment.

## Prompting

To prompt the model and get a response (where `PROMPT` is a prompt of your choosing):

```bash
lemonade -i amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid oga-load --device hybrid --dtype int4 llm-prompt --max-new-tokens 64 -p PROMPT
```

## Responsiveness

To measure the model's time-to-first-token (TTFT) and tokens/second with a sequence length of 1024 and 64 output tokens:

```bash
lemonade -i amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid oga-load --device hybrid --dtype int4 oga-bench --warmup-iterations 5 --iterations 10 --prompt 1024 --output-tokens 64
```

## Task Performance

To measure the model's accuracy on the [MMLU test](https://github.com/lemonade-sdk/lemonade/blob/main/docs/mmlu_accuracy.md) `management` subject, run:

```bash
lemonade -i amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid oga-load --device hybrid --dtype int4 accuracy-mmlu --tests management
```

# Application Integration

## Blocking

To integrate blocking generation into your application (i.e., the entire response is delivered at once), replace your existing LLM invocation with this code:

```python
from lemonade.api import from_pretrained

model, tokenizer = from_pretrained(
    "amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid", recipe="oga-hybrid"
)

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)

print(tokenizer.decode(response[0]))
```

## Streaming

To integrate streaming generation into your application (i.e., each token of the response is streamed back to the main thread as soon as it becomes available), replace your existing LLM invocation with this code:

```python
from threading import Thread
from lemonade.api import from_pretrained
from lemonade.tools.ort_genai.oga import OrtGenaiStreamer

model, tokenizer = from_pretrained(
    "amd/Llama-3-8B-awq-g128-int4-asym-fp16-onnx-hybrid", recipe="oga-hybrid"
)
input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids

streamer = OrtGenaiStreamer(tokenizer)
generation_kwargs = {
    "input_ids": input_ids,
    "streamer": streamer,
    "max_new_tokens": 30,
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# Generate the response using streaming
for new_text in streamer:
    print(new_text)

thread.join()
```

## Application Example

See the [Chat Demo](https://github.com/lemonade-sdk/lemonade/blob/main/examples/demos/chat/chat_hybrid.py) for an example application that demonstrates streaming, multi-threading, and response interruption.

# Server Interface (REST API)

To launch a server process from your Python environment, run the command:

```bash
lemonade serve
```

Once launched, the APIs for accessing the server are the same, regardless of which LLM and device was used to launch the server. Check out the [Sever Interface documentation](https://ryzenai.docs.amd.com/en/latest/llm/server_interface.html) to understand how to interact with the server process.

# Next Steps

This guide provided instructions for testing and deploying an LLM on a target device using the Lemonade SDK's tools and APIs. 

- Visit the [Lemonade LLM examples table](../README.md) to learn how to do this for any of the supported combinations of LLM and device.
- Visit the [overall Ryzen AI LLM documentation](https://ryzenai.docs.amd.com/en/latest/llm/overview.html#) to learn about other deployment options, such as native C++ libraries.
- Visit the [Lemonade SDK repository](https://github.com/lemonade-sdk/lemonade) to learn about more tools and features.

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.