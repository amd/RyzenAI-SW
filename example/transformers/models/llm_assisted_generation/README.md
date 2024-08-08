# Assisted generation

Assisted generation is similar to speculative decoding, released by [HuggingFace](https://huggingface.co/blog/assisted-generation).

# Setup

Complete the setup instructions for [LLMs on RyzenAI with Pytorch](../llm/docs/README.md) before trying out these steps.

## Step 1: Generate AWQ checkpoints
Generate the quantized checkpoints by running ```models/llm/run_awq.py``` for target models. 

* To generate the quantized target model for OPT-6.7b run the below command from ```models/llm``` directory
```
python run_awq.py --model_name facebook/opt-6.7b --task quantize
```
* To generate the quantized target model for llama-2-7b run the below command from ```models/llm``` directory
```
python run_awq.py --model_name llama-2-7b --task quantize
```
## Step 2: Get draft models
* For OPT-6.7b, ```facebook/opt-125m``` is used as the draft model. 
* For llama-2-7b, ```JackFram/llama-160m``` is used as the draft model. 
* Draft models are used with bf16 precision on CPU.
* Speed up observed is 2-3x in token time. 

|   Target Model                | Assistant Model               |
|-------------------------------|-------------------------------|
| OPT-6.7b (AWQ-w4abf16)        | OPT-125M (bf16/SQ-w8a8)       |
| llama-2-7b (AWQ-w4abf16)	    | JackFram/llama-160m (bf16)    |

# Instructions to run the LLM models
## 1. OPT-6.7b
### a. OPT-6.7b without assisted generation
```
python assisted_generation.py --model_name opt-6.7b --task benchmark
```
### b. OPT-6.7b with OPT-125M assistant model
```
python assisted_generation.py --model_name opt-6.7b --task benchmark --assisted_generation
```
## 2. llama-2-7b
### a. llama-2-7b without assisted generation, with fast attention
```
python assisted_generation.py --model_name llama-2-7b --task benchmark --fast_attention
```
### b. llama-2-7b with llama-160m assistant model, with fast attention
```
python assisted_generation.py --model_name llama-2-7b --task benchmark --assisted_generation --fast_attention
```
**Note:**
- fast_attention argument is only supported with llama-2-7b, and llama-2-7b-chat models in this release.
- Know issue related to kernel driver shows up when using assisted_generation with Llama-2-7b.
