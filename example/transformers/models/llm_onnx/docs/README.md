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

Setup the environment variable:
```powershell
cd <transformers>
set TRANSFORMERS_ROOT=%CD%
```

Create conda environment:
```powershell
cd %TRANSFORMERS_ROOT%\models\llm_onnx
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
cd %TRANSFORMERS_ROOT%
.\setup_phx.bat
```

##### For STX
```
cd %TRANSFORMERS_ROOT%
.\setup_stx.bat
```

## Steps to run the models

### Prepare the model

Use "prepare_model.py" script to export, optimize and quantize the LLMs. You can also optimize or quantize an existing ONNX model by providing the path to the model directory.

Check script usage

```powershell
cd %TRANSFORMERS_ROOT%\models\llm_onnx
python prepare_model.py --help

```python prepare_model.py --help```

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

```python prepare_model.py --help```

> As we are using an `int4` quantized model, responses might not be as accurate
as `float32` model. The quantizer used is `MatMul4BitsQuantizer` from onnxruntime

 > As for the optimizer , ORT optimizer is used.
### Using ONNX Runtime Interface
 
**Note:** Copy the 'model.onnx.data' file from output_model_dir to the models/llm_onnx/ folder.

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
