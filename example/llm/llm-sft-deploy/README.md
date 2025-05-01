# Accelerate Finetuned LLMs Locally on NPU + iGPU Ryzen AI Processor

This repo provides supplemental code to the AMD Blog [Accelerate Finetuned LLMs Locally on NPU + iGPU Ryzen AI processor](https://www.amd.com/fr/developer/resources/technical-articles/accelerate-llms-locally-on-amd-ryzen-ai-npu-and-igpu.html). Code is provided for LoRA finetuning on MI300X and then running inference of finetuned model on Ryzen AI. 

# Finetuning LLMs

## Getting Started
1. Install miniconda/anaconda and create a new conda environment for training/inference on GPUs
2. Install requirements.txt using ```pip install -r requirements.txt```
3. Set Huggingface API Tokens by ```export={HUGGINGFACE_API_TOKEN}``` in terminal. Needed for accessing gated models and saving to Huggingface. 

## Finetune

We provide ``train.py`` to do LoRA finetuning. Training can be saved locally or directly to huggingface and wandB can be utilized to track training <br/>
Set ``--hf_dir local`` to save locally and bypass huggingface and wandB setup. 

The training script LoRA finetunes Llama3.2 1B on the [Volve Alpaca Dataset](https://huggingface.co/datasets/bengsoon/volve_alpaca), an application for the oil & rigging industry. 

### Finetuning Adapter (Save Locally)
```python train.py --lora --lora_qv --hf_dir local```

### Finetuning Adapter (Save to HF)
```python train.py --lora --lora_qv --hf_dir <HF_username/repo-name>```

### Merging Adapter 
After finetuning the adapter, merge adapter with base LLM through the following:
```python train.py --merge_model --model_name meta-llama/Llama-3.2-1B --adapter_model_dir <adapter path>```

## LLM Inference of Finetuned models on GPU
Use: ``inference.py`` to run inference on GPU . <br/>
Set ``--inference_filename`` to a ".json" filename in which model predictions will be stored.

#### Inference on Finetuned (merged) model 
```python inference.py --fp --model_dir amd/volve-llama3.2-1b --inference_filename "volve-llama3_1B.json"```

#### Inference on Quark Quantized model (safetensors) 
- Install Quark from wheel file [here](https://quark.docs.amd.com/latest/install.html#install-quark-quark-examples-from-download). <br/>
- Inside the zip folder are example scripts. Use the following for AWQ quantization: <br/>

```
cd examples/torch/language_modeling/llm_ptq/
python quantize_quark.py \
     --model_dir <finetuned model>  \
     --output_dir <quantized safetensor output dir>  \
     --quant_scheme w_uint4_per_group_asym \
     --num_calib_data 128 \
     --quant_algo awq \
     --dataset pileval_for_awq_benchmark \
     --model_export hf_format \
     --data_type float16 \
     --custom_mode awq
```

- Run the following for inference: <br/>
```python inference.py --quark_safetensors --quant_model_dir <path to quant model> --inference_filename "quantized_model.json"```


# Deploy on Ryzen AI 

### Quantize the full-precision, finetuned model using the quantization strategy menentioned [here](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html#generate-quantized-model)

### Install RyzenAI and prerequisites accoring to instructions [here](https://ryzenai.docs.amd.com/en/latest/inst.html).

### Transform the quantized model to run on Hybrid approach within Ryzen AI, utilizing both the iGPU and NPU by running the following. See reference [here](https://ryzenai.docs.amd.com/en/latest/oga_model_prepare.html#postprocessing).

### Run inference on RyzenAI with the following: <br/>
``python inference_oga.py --model_dir "<hybrid-model-path>" --inference_filename hybrid_ft_model.json``

Please check the blog post for comprehensive instructions on additional packages needed within the ryzen-ai conda environment.
