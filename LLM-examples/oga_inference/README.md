# OGA Inference with Chat Template

Inference script with chat template support for ONNX Runtime GenAI models.

## When to use this?
Use this for models that require chat templates (e.g., GPT-OSS-20B) for better output quality.

Based on Microsoft OGA [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/rel-0.11.2/examples/python/model-chat.py), modified for Ryzen AI.

## Prerequisites
- Ryzen AI Software installed (see [Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html))
- Activate the Python virtual environment created by the MSI installer:
```bat
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```
- For more details on running LLMs with OGA, see [OnnxRuntime GenAI (OGA) Flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html).

## Usage - Chat and Context Cache Testing (model_chat.py)

Interactive chat, single-prompt, and multi-turn context cache testing with timing.

This example is based on [Microsoft's OGA example](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-chat.py).

### Usage

```bash
# Single prompt (fits within 4096 context by default)
python model_chat.py -m "path\to\model" -pr amd_genai_prompt.txt --timings

# Read only the first 256 tokens from the input prompt file
python model_chat.py -m "path\to\model" -pr amd_genai_prompt.txt -l 256

# Long context support (increase context window to e.g. 16k)
python model_chat.py -m "path\to\model" -pr amd_genai_prompt_long.txt -mpt 16000

# Read first 256 tokens from the input prompt file, and increase context window e.g. 16k
python model_chat.py -m "path\to\model" -pr amd_genai_prompt_long.txt -l 256 -mpt 16000

# Multi-turn context cache test (prompts sent sequentially, no rewind)
python model_chat.py -m "path\to\model" -j questions.json

# Interactive chat (use -r to clear KV cache between turns)
python model_chat.py -m "path\to\model"
python model_chat.py -m "path\to\model" -r

# Get all options
python model_chat.py -h
```

### Multi-Turn JSON Format

The `-j` flag accepts a JSON file where prompts are sent sequentially to test context cache reuse:

```json
[
  {"id": 1, "prompt": "Can you explain why the sky is blue?"},
  {"id": 2, "prompt": "What about sunsets?"},
  {"id": 3, "prompt": "Summarize what we discussed."}
]
```

### Key Options

| Flag | Long | Default | Description |
|------|------|---------|-------------|
| `-m` | `--model_path` | — | **(required)** Path to ONNX model folder |
| `-pr` | `--prompt_file` | — | Single prompt `.txt` file |
| `-j` | `--json_file` | — | Multi-turn prompts `.json` file |
| `-l` | `--prompt_length` | — | Read only the first N tokens from the prompt file. |
| `-mpt` | `--max_prompt_tokens` | 4096 | Model context window. Sets context cache size. Increase for long context (e.g. `-mpt 16000`). |
| `-or` | `--output_reserve` | 256 | Tokens reserved for generated output. |
| `-e` | `--execution_provider` | `follow_config` | `cpu`, `cuda`, `dml`, or `follow_config` |
| `-s` | `--system_prompt` | `You are a helpful AI assistant.` | System prompt |
| `-r` | `--rewind` | off | Clear KV cache and rewind to system prompt between turns (interactive mode) |
| `-v` | `--verbose` | off | Detailed debug output |
| `-tm` | `--timings` | off | Print TTFT, prefill tps, decode tps |
| `-gtype` | `--guidance_type` | `none` | Guided generation: `json_schema`, `regex`, `lark_grammar` |
| `-ginfo` | `--guidance_info` | — | Guidance payload |
