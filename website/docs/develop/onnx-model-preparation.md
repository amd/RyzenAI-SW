# ONNX Model Preparation

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Preparing OGA Models

<CIStatus validated={false} />

This section describes the process for preparing LLMs for deployment on a Ryzen AI PC using the hybrid or NPU-only execution mode. Currently, the flow supports only fine-tuned versions of the models already supported (as listed in the [hybrid OGA](/models-tutorials/llms/supported-models) page). For example, fine-tuned versions of Llama2 or Llama3 can be used. However, different model families with architectures not supported by the hybrid flow cannot be used.

For fine-tuned models that introduce architectural changes requiring new operator shapes not available in the Ryzen AI runtime, refer to the [Operator Preparation](/develop/operator-preparation) page.

Preparing a LLM for deployment on a Ryzen AI PC involves 2 steps:

1. **Quantization**: The pretrained model is quantized to reduce memory footprint and better map to compute resources in the hardware accelerators
2. **Postprocessing**: During the postprocessing the model is exported to OGA followed by NPU-only or Hybrid execution mode specific postprocess to obtain the final deployable model.

## Quantization

### Prerequisites

Linux machine with AMD (e.g., AMD Instinct MI Series) or Nvidia GPUs

### Setup

1. Create and activate Conda Environment

```bash
conda create --name <conda_env_name> python=3.11
conda activate <conda_env_name>
```

2. If Using AMD GPUs, update PyTorch to use ROCm

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
python -c "import torch; print(torch.cuda.is_available())" # Must return `True`
```

3. Download [AMD Quark 0.11](https://download.amd.com/opendownload/Quark/amd_quark-0.11.zip) and unzip the archive

4. Install Quark:

```bash
cd <extracted amd quark-version>
pip install amd_quark-<version>+<>.whl
```

5. Install other dependencies

```bash
pip install datasets
pip install transformers
pip install accelerate
pip install evaluate
pip install nltk
```

Some models may require a specific version of `transformers`. For example, ChatGLM3 requires version 4.44.0.

### Generate Quantized Model

Use following command to run Quantization. In a GPU equipped Linux machine the quantization can take about 30-60 minutes.

```bash
cd examples/torch/language_modeling/llm_ptq/

python quantize_quark.py \
     --no_trust_remote_code \
     --model_dir "meta-llama/Llama-2-7b-chat-hf"  \
     --output_dir <quantized safetensor output dir>  \
     --quant_scheme w_uint4_per_group_asym \
     --group_size 128 \
     --num_calib_data 128 \
     --seq_len 512 \
     --quant_algo awq \
     --dataset pileval_for_awq_benchmark \
     --model_export hf_format \
     --data_type <datatype> \
     --exclude_layers []
```

- Use `--data_type bfloat16` for bf16 pretrained model. For fp32/fp16 pretrained model use `--datatype float16`
- Not using `--exclude_layers` parameter may result in model-specific defaults which may exclude certain layers like output layers.

The quantized model is generated in the &lt;quantized safetensor output dir&gt; folder.

:::info
**Note:** For the Phi-4 model, the following quantization recipe is recommended for better accuracy:

- Use `--quant_algo gptq`
- Add `--group_size_per_layer lm_head 32`
:::

:::info
**Note:** Currently the following files are not copied into the quantized model folder and must be copied manually:

- For Phi-4 models: `configuration_phi3.py`
- For ChatGLM-6b models: `tokenizer.json`
:::

## Postprocessing

Copy the quantized model to the Windows PC with Ryzen AI installed, activate the Ryzen AI Conda environment.

```bash
conda activate ryzen-ai-<version>
pip install onnx_ir
pip install torch==2.7.1
```

Generate the final model for Hybrid execution mode:

```bash
conda activate ryzen-ai-<version>

model_generate --hybrid <output_dir> <quantized_model_path>
```

Generate the final model for NPU execution mode:

```bash
conda activate ryzen-ai-<version>

model_generate --npu <output_dir> <quantized_model_path>  --optimize decode
```

Generate model for hybrid execution mode (prefill fused version)

```bash
conda activate ryzen-ai-<version>

model_generate --hybrid <output_dir> <quantized_model_path>  --optimize prefill
```

- Prefill fused hybrid models are only supported for Phi-3.5-mini-instruct and Mistral-7B-Instruct-v0.2
- Edit `genai_config.json` with the following entries

```json
"decoder": {
    "session_options": {
        "log_id": "onnxruntime-genai",
        "custom_ops_library": "onnx_custom_ops.dll",
        "external_data_file": "token.pb.bin",
        "custom_allocator": "ryzen_mm",
        "config_entries": {
            "dd_cache": "",
            "hybrid_opt_token_backend": "gpu",
            "hybrid_opt_max_seq_length": "4096",
            "max_length_for_kv_cache": "4096"
        },
        "provider_options": []
    },
    "filename": "fusion.onnx"
}
```

:::info
**Note**: During the `model_generate` step, the quantized model is first converted to an OGA model using ONNX Runtime GenAI Model Builder (version 0.9.2). It is possible to use a standalone environment for exporting an OGA model, refer to the official [ONNX Runtime GenAI Model Builder documentation](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models). Once you have an exported OGA model, you can pass it directly to the `model_generate` command, which will skip the export step and perform only the post-processing.
:::

Here are simple commands to export OGA model from quantized model using a standalone environment

```bash
conda create --name oga_builder_env python=3.10
conda activate oga_builder_env

pip install onnxruntime-genai==0.9.2
pip install torch transformers onnx numpy

python3 -m onnxruntime_genai.models.builder -m <input quantized model> -o <output OGA model> -p int4 -e dml
```
