# High-Level Python SDK

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# High-Level Python SDK

<CIStatus validated={false} />

A Python environment offers flexibility for experimenting with LLMs, profiling them, and integrating them into Python applications. We use the [Lemonade SDK](https://github.com/lemonade-sdk/lemonade) to get up and running quickly.

To get started, follow these instructions.

## System-level pre-requisites

You only need to do this once per computer:

1. Make sure your system has the recommended Ryzen AI driver installed as described in the [installation guide](/getting-started/installation).
2. Download and install [Miniconda for Windows](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe) or [Miniforge for Windows](https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Windows-x86_64.exe).
3. Launch a terminal and call `conda init`.

## Environment Setup

To create and set up an environment, run these commands in your terminal:

```bash
conda create -n ryzenai-llm python=3.12
conda activate ryzenai-llm
pip install lemonade-sdk[dev,oga-ryzenai] --extra-index-url=https://pypi.amd.com/simple
```

## Validation Tools

Now that you have completed installation, you can try prompting an LLM like this (where `PROMPT` is any prompt you like).

Run this command in a terminal that has your environment activated:

```bash
lemonade -i amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid oga-load --device hybrid --dtype int4 llm-prompt --max-new-tokens 64 -p PROMPT
```

For more details on validation commands, see the [Lemonade Server CLI Guide](https://lemonade-server.ai/docs/server/lemonade-server-cli/).

## Python API

You can also run this code to try out the high-level Lemonade API in a Python script:

```python
from lemonade.api import from_pretrained

model, tokenizer = from_pretrained(
    "amd/Llama-3.2-1B-Instruct-onnx-ryzenai-hybrid", recipe="oga-hybrid"
)

input_ids = tokenizer("This is my prompt", return_tensors="pt").input_ids
response = model.generate(input_ids, max_new_tokens=30)

print(tokenizer.decode(response[0]))
```

## Next Steps

From here, you can explore additional validation tools for measuring speed and accuracy, streaming responses with the API, and launching the server interface. See the [Supported LLMs](/models-tutorials/llms/supported-models) for model-specific examples, or the [Lemonade Server CLI Guide](https://lemonade-server.ai/docs/server/lemonade-server-cli/) for full CLI documentation.
