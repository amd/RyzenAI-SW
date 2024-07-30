# LLMs on RyzenAI with ONNXRuntime

The following models are supported on RyzenAI with the 4 bit Blockwise quantization.

Models: `facebook/opt-125m`, `facebook/opt-1.3b`, `facebook/opt-2.7b`, `facebook/opt-6.7b`, `meta-llama/Llama-2-7b-hf`, `Qwen/Qwen1.5-7B-Chat`, `THUDM/chatglm3-6b`, `codellama/CodeLlama-7b-hf`

:pushpin: [Blockwise WOQ](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/matmul_4bits_quantizer.py)

:pushpin: We recommend to use minimum 64 GB RAM
## Prerequisites

:pushpin: For some models, the user needs to have Hugging Face account and should use huggingface-cli as per their documentation: https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login

:pushpin: To request access for Llama-2,
visit [Meta's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
and accept [Huggingface license](https://huggingface.co/meta-llama/Llama-2-7b-hf).

:pushpin: Conda environment with python 3.10

Create conda environment:
```powershell
cd <transformers/models/llm_onnx>
conda update -n base -c defaults conda -y
conda env create --file=env.yaml
conda activate llm_onnx
```

### Install ONNX EP for running ONNX based flows

Install the ONNX EP and required version of libraries using the python wheel files 

```
pip install --upgrade --force-reinstall onnxruntime_vitisai-1.18.0-cp310-cp310-win_amd64.whl
pip install --upgrade --force-reinstall voe-1.2.0-cp310-cp310-win_amd64.whl
pip install numpy==1.26.4
```

### Set environment


##### For PHX
```
cd <transformers>
.\setup_phx.bat
```

##### For STX
```
cd <transformers>
.\setup_stx.bat
```

## Steps to run the models

### Prepare the model

Use "prepare_model.py" script to export, optimize and quantize the LLMs. You can also optimize or quantize an existing ONNX model by providing the path to the model directory.

Check script usage
```powershell
python prepare_model.py --help

usage: prepare_model.py [-h]
                        [--model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,llama-2-7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}]
                        [--groupsize {32,64,128}] --output_model_dir OUTPUT_MODEL_DIR [--input_model INPUT_MODEL] [--only_onnxruntime]
                        [--opt_level {0,1,2,99}] [--export] [--optimize] [--quantize]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,llama-2-7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                        model name
  --groupsize {32,64,128}
                        group size for blockwise quantization
  --output_model_dir OUTPUT_MODEL_DIR
                        output directory path
  --input_model INPUT_MODEL
                        input model path to optimize/quantize
  --only_onnxruntime    optimized by onnxruntime only, and no graph fusion in Python
  --opt_level {0,1,2,99}
                        onnxruntime optimization level. 0 will disable onnxruntime graph optimization. Level 2 and 99 are intended for --only_onnxruntime.
  --export              export float model
  --optimize            optimize exported model
  --quantize            quantize float model
```

#### Export, Optimize and quantize the model

```powershell

python .\prepare_model.py --model_name <model_name> --output_model_dir <output directory> --export --optimize --quantize
```
#### Optimize and quantize existing model

```powershell
python .\prepare_model.py --model_name <model_name> --input_model <input model path> --output_model_dir <output directory> --optimize --quantize
```

#### Quantize existing model

```powershell
python .\prepare_model.py --input_model <input model path> --output_model_dir <output directory> --quantize
```

### Running Inference

Check script usage
```powershell
python infer.py --help

usage: infer.py [-h] --model_dir MODEL_DIR [--draft_model_dir DRAFT_MODEL_DIR] --model_name
                {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                [--tokenizer TOKENIZER] [--dll DLL] [--target {cpu,aie}] [--task {decode,benchmark,perplexity}] [--seqlen SEQLEN [SEQLEN ...]]
                [--max_new_tokens MAX_NEW_TOKENS] [--ort_trace] [--view_trace] [--prompt PROMPT] [--max_length MAX_LENGTH] [--profile] [--power_profile]
                [-v]

LLM Inference on Ryzen-AI

optional arguments:
  -h, --help            show this help message and exit
  --model_dir MODEL_DIR
                        Model directory path
  --draft_model_dir DRAFT_MODEL_DIR
                        Draft Model directory path for speculative decoding
  --model_name {facebook/opt-125m,facebook/opt-1.3b,facebook/opt-2.7b,facebook/opt-6.7b,meta-llama/Llama-2-7b-hf,Qwen/Qwen1.5-7B-Chat,THUDM/chatglm3-6b,codellama/CodeLlama-7b-hf}
                        model name
  --tokenizer TOKENIZER
                        Path to the tokenizer (Optional).
  --dll DLL             Path to the Ryzen-AI Custom OP Library
  --target {cpu,aie}    Target device (CPU or Ryzen-AI)
  --task {decode,benchmark,perplexity}
                        Run model with a specified task
  --seqlen SEQLEN [SEQLEN ...]
                        Input Sequence length for benchmarks
  --max_new_tokens MAX_NEW_TOKENS
                        Number of new tokens to be generated
  --ort_trace           Enable ORT Trace dump
  --view_trace          Display trace summary on console
  --prompt PROMPT       User prompt
  --max_length MAX_LENGTH
                        Number of tokens to be generated
  --profile             Enable profiling summary
  --power_profile       Enable power profiling via AGM
  -v, --verbose         Enable argument display
```

> As we are using an `int4` quantized model, responses might not be as accurate
as `float32` model. The quantizer used is `MatMul4BitsQuantizer` from onnxruntime

 > As for the optimizer , ORT optimizer is used.
### Using ONNX Runtime Interface
 
Copy the 'model.onnx.data' file from output_model_dir to the models/llm_onnx/ folder.

**Note:** Each run generates a log file in `./logs` directory with name `log_<model_name>.log`.
```powershell
python .\infer.py --model_name <model_name> --target <cpu/aie> --model_dir <input model directory> --profile --task decode
```

Example Usage: OPT-125m Model
```
python .\prepare_model.py --model_name facebook/opt-125m --output_model_dir opt125m --export --optimize --quantize
python .\infer.py --model_name facebook/opt-125m --target aie --model_dir opt125m/quant --profile --task decode
```

Example Usage: llama-2-7b
```
This will need a HF auth token with access to https://huggingface.co/meta-llama/Llama-2-7b-hf

python .\prepare_model.py --model_name meta-llama/Llama-2-7b-hf --output_model_dir llama2 --export --optimize --quantize
python .\infer.py --model_name meta-llama/Llama-2-7b-hf --target aie --model_dir llama2/quant --profile --task decode
```
