# Hugging Face CPU: microsoft/Phi-3-mini-4k-instruct (bfloat16)

This guide contains all of the instructions necessary to get started with the model `microsoft/Phi-3-mini-4k-instruct` on Hugging Face CPU in the bfloat16 data type.

The CPU implementation in this guide is designed to run on most PCs. However, for optimal performance on Ryzen AI 300-series PCs, try the [hybrid execution mode](../hybrid/Phi_3_mini_4k_instruct.md).

The commands and scripts in this guide leverage the `lemonade` SDK from the [ONNX TurnkeyML](https://github.com/onnx/turnkeyml) project. The `lemonade` SDK provides everything you need to get up and running with LLMs on the OnnxRuntime GenAI (OGA) framework, as well as the support for Hugging Face `transformers` baselines leveraged in this guide.

# Checkpoint

The Hugging Face CPU implementation of `microsoft/Phi-3-mini-4k-instruct` uses the ONNX Runtime GenAI Model Builder tool, via `lemonade` to export an ONNX model for use in inference.

# Setup

To get started with the `lemonade` SDK in a Python environment, follow these instructions.

### System-level pre-requisites

You only need to do this once per computer:

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

3. Install ONNX TurnkeyML to get access to the LLM tools and APIs.
    ```bash
    pip install turnkeyml[llm]
    ```

# Validation Tools

You can use the following `lemonade` commands to test out your model, in your activated `ryzenai-llm` environment.

## Prompting

To prompt the model and get a response (where `PROMPT` is a prompt of your choosing):

```bash
lemonade -i microsoft/Phi-3-mini-4k-instruct huggingface-load --device cpu --dtype bfloat16 llm-prompt --max-new-tokens 64 -p PROMPT
```

## Responsiveness

To measure the model's time-to-first-token (TTFT) and tokens/second with a sequence length of 1024 and 64 output tokens:

```bash
lemonade -i microsoft/Phi-3-mini-4k-instruct huggingface-load --device cpu --dtype bfloat16 huggingface-bench --warmup-iterations 10 --iterations 20 --prompt 1024 --output-tokens 64
```

## Task Performance

To measure the model's accuracy on the [MMLU test](https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/mmlu_accuracy.md) `management` subject, run:

```bash
lemonade -i microsoft/Phi-3-mini-4k-instruct huggingface-load --device cpu --dtype bfloat16 accuracy-mmlu --tests astronomy philosophy management
```

# Application Integration

## Blocking

To integrate blocking generation into your application (i.e., the entire response is delivered at once), replace your existing LLM invocation with this code:

```python
from lemonade.api import from_pretrained

model, tokenizer = from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", recipe="hf-cpu"
)

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)

print(tokenizer.decode(response[0]))
```

## Streaming

To integrate streaming generation into your application (i.e., each token of the response is streamed back to the main thread as soon as it becomes available), replace your existing LLM invocation with this code:

```python
from threading import Thread
from transformers import TextIteratorStreamer
from lemonade.api import from_pretrained

model, tokenizer = from_pretrained("microsoft/Phi-3-mini-4k-instruct", recipe="hf-cpu")

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
)
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



# Server Interface (REST API)

To launch a server process from your Python environment, run the command:

```bash
lemonade serve
```

Once launched, the APIs for accessing the server are the same, regardless of which LLM and device was used to launch the server. Check out the [Sever Interface documentation](https://ryzenai.docs.amd.com/en/latest/llm/server_interface.html) to understand how to interact with the server process.

# Next Steps

This guide provided instructions for testing and deploying an LLM on a target device using the `lemonade` SDK's tools and APIs. 

- Visit the table of [Supported LLMs table](https://ryzenai.docs.amd.com/en/latest/llm/overview.html#supported-llms) in the documentation to learn how to do this for any of the supported combinations of LLM and device.
- Visit the [overall LLM documentation](https://ryzenai.docs.amd.com/en/latest/llm/overview.html#) to learn about other deployment options, such as native C++ libraries.
- Visit the [lemonade SDK repository](https://github.com/onnx/turnkeyml/blob/main/docs/lemonade/getting_started.md) to learn about more tools and features.

# Copyright

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.