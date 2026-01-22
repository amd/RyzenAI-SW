# OGA Inference with Chat Template

Inference script with chat template support for ONNX Runtime GenAI models.

## When to use this?
Use this for models that require chat templates (e.g., GPT-OSS-20B) for better output quality.

Based on Microsoft OGA [model-chat.py](https://github.com/microsoft/onnxruntime-genai/blob/rel-0.11.2/examples/python/model-chat.py), modified for Ryzen AI.

## Prerequisites
- Ryzen AI Software installed (see [Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html))
- Activate the conda environment created by the MSI installer:
```bash
  conda activate ryzen-ai-<version>
```
- For more details on running LLMs with OGA, see [OnnxRuntime GenAI (OGA) Flow](https://ryzenai.docs.amd.com/en/latest/hybrid_oga.html).

## Usage
```bash
python model_chat.py -m <model_path> -pr <prompt_file> -ipl <input_tokens> -tm
```

## Arguments
| Argument | Description |
|----------|-------------|
| `-m` | Path to ONNX model folder |
| `-pr` | Prompt file (.txt) |
| `-ipl` | Input prompt length in tokens (auto-caps to fit context) |
| `-tm` | Show timing info |
| `-v` | Verbose output |

## Example
```bash
python model_chat.py -m ./gpt-oss-20b-onnx-ryzenai-npu -pr prompt.txt -ipl 256 -tm
```