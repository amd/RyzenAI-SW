# OPT - Pytorch

**_NOTE:_**  Please ensure that you followed the environment setup instructions from the [Transformer folder readme](../../README.md) before following the steps here.

OPT (Open Pre-trained Transformer) class of language models are released by Meta and are available through [Huggingface](https://huggingface.co/facebook/opt-1.3b)

This example shows how to deploy OPT models on AIE and custom AIE+CPU backend in **eager mode**. 

### Model Structure 

This image shows model structure after quantizing and replacing Linear node with AIE custom linear node.
![Model Structure](opt1.3b.png)

### Model Variants

Ryzen-AI supports opt-125m, opt-350m, opt-1.3b up to a sequence length of 2048 tokens.


### Model preparation steps

The model preparation steps consist of: 

1. Load FP32 or BFloat16 OPT model
2. Perform [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) to condition weights before quantization.
3. Quantize the model usign PyTorch's dynamic quantization
  1. Optionally load from checpoint. 
4. Idenfiy quantized linear nodes in the model
5. Replace each quantized linear node with a custom Linear node (QLinear)
6. Initialize weights through Torch's PackedParams
7. Pad, tile and cache weights as initialization step
8. Run inferece - benchmark, decode or perplexity

In this flow, compute primitives are chosen to optimize prompt/prefill phase and token phase, both independently.

The flow below provides two precision mode options for quantization and subsequent deployment: w4abf16 and w8a8

## Precision w4abf16

### Quantization 

Go to the opt folder for PyTorch

```
cd models/opt
```

Quantize w4abf16 using AWQ

```
python run_awq.py --task quantize --model_name opt-1.3b
```

### Deployment 

Run OPT Model   
```
python run_awq.py --task decode --target aie --model_name opt-1.3b
```

Example output


```

****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?
I think it's a bit like the meaning of life in the bible.
****************************************
prompt: Tell me something you don't know.
response: Tell me something you don't know.
I don't know how to make a sandwich.
I don't know how to make a sandwich

****************************************
prompt: What is the mass of earth?
response: What is the mass of earth?

The mass of earth is the mass of all the matter in the universe. It is the mass of
****************************************
prompt: What is a poem?
response: What is a poem?

A poem is a story told in a single line.

What is a story?

A story
****************************************
prompt: What is recursion?
response: What is recursion?

Recursion is a mathematical concept that is used to describe the process of recursively recursing.

****************************************
prompt: Tell me a one line joke.
response: Tell me a one line joke.
I'm not sure I can do that. I'm not sure I can do anything.
I'm
****************************************
prompt: Who is Gilgamesh?
response: Who is Gilgamesh?
Gilgamesh is the name of the character in the game.
****************************************
prompt: Tell me something about cryptocurrency.
response: Tell me something about cryptocurrency.

I’m not sure what you’re asking.

I’m not sure
****************************************
prompt: How did it all begin?
response: How did it all begin?

I’ve been a fan of the series since the very beginning. I’ve read all
```

### Results 

#### Benchmarking 

```
python run_awq.py --model_name opt-1.3b --target aie --task benchmark

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:2.812s       Prefill-phase:239.206ms Time/token:120msTokens/sec:8.3
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:2.742s       Prefill-phase:380.712ms Time/token:117msTokens/sec:8.5
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:2.699s       Prefill-phase:198.169ms Time/token:118msTokens/sec:8.5
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:2.692s       Prefill-phase:196.999ms Time/token:118msTokens/sec:8.5
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:2.897s       Prefill-phase:185.554ms Time/token:117msTokens/sec:8.5
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:2.921s       Prefill-phase:190.427ms Time/token:118msTokens/sec:8.5
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:2.683s       Prefill-phase:195.883ms Time/token:118msTokens/sec:8.5
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:2.844s       Prefill-phase:187.372ms Time/token:120msTokens/sec:8.3
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:2.824s       Prefill-phase:189.878ms Time/token:119msTokens/sec:8.4
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:2.786s       Prefill-phase:186.962ms Time/token:117msTokens/sec:8.5
```

```
python run_awq.py --model_name opt-1.3b --target aie --task benchmark_long 

Example#:1      Prompt-len:512  New-tokens-generated:11 Total-time:8.606s       Prefill-phase:7250.999ms        Time/token:134ms        Tokens/sec:7.4
Example#:2      Prompt-len:1024 New-tokens-generated:11 Total-time:18.931s      Prefill-phase:17429.224ms       Time/token:149ms        Tokens/sec:6.7
Example#:3      Prompt-len:1536 New-tokens-generated:11 Total-time:32.166s      Prefill-phase:30551.226ms       Time/token:160ms        Tokens/sec:6.2
Example#:4      Prompt-len:2000 New-tokens-generated:11 Total-time:46.789s      Prefill-phase:45040.952ms       Time/token:173ms        Tokens/sec:5.8
```

## Precision w8a8

### Quantization

Quantize and save model weights
```
python save_weights.py --action save --model_name opt-1.3b
```
This saves 3 sets of weights. 

1. "quantized_<model_name>_float32.pth" - All linear ops are int8 dynamic quantized, while all other ops are in float32
2. "quantized_<model_name>_bfloat16.pth" - All linear ops are int8 dynamic quantized, while all other ops are in bfloat16
3. "weights_<model_name>" directory for onnx

### Deployment 

Run using run.py

This script gives option to do the following:
* Benchmark code - measure time/token latency
* Calculate perplexity score
* Profile code using PyTorch profiler
* Decode a set of prompts to show model liveliness

```
usage: run.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b,opt-13b,opt-30b}] [--target {cpu,customcpu,aie}] [--dtype {bfloat16,float32}]
              [--quant_mode {w8a8}] [--smoothquant] [--amdopt] [--profile] [--no-accelerate] [--dataset {non-raw,raw}] [--load] [--flash_attention]
              [--num_torch_threads {1,2,3,4,5,6,7,8}] [--task {decode,benchmark,benchmark_long,perplexity,torchprofile,opsprofile}] [--impl {v0,v1}] [--fuse_mlp]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b,opt-13b,opt-30b}
                        Different OPT model sizes
  --target {cpu,customcpu,aie}
                        cpu, custumcpu, aie
  --dtype {bfloat16,float32}
                        All ops other than linear ops in bfloat16 or float32
  --quant_mode {w8a8}   Quantization mode - w8a8
  --smoothquant         Enable smoothquant
  --amdopt              Use OPT from local folder - with profile instrumentation: use without --load
  --profile             Log matmul times for prompt and token phases - supported only for AIE target
  --no-accelerate       Use tiling/padding in c++ or python with AIE target
  --dataset {non-raw,raw}
                        Dataset - wikitext2-raw-v1, wikitext2-v1
  --load                Load quantized weights from checkpoint
  --flash_attention     Enable flash attention
  --num_torch_threads {1,2,3,4,5,6,7,8}
                        Number of torch threads
  --task {decode,benchmark,benchmark_long,perplexity,torchprofile,opsprofile}
                        perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences
                        (compare with flash attn); decode: Decode set of prompts; torchprofile: profile using torch profiler for decode
  --impl {v0,v1}        Choose between different implementations for aie target
  --fuse_mlp            Enable MLP fusion

```


Each run generates a log directory "log_<model_name>" and all logs are within this directory. 

Each run generatses 2 log files in the "log_<model_name>" directory

For example:
```
logs_opt-1.3b\log_opt-1.3b_aie_ptdq_float32.log
logs_opt-1.3b\log_opt-1.3b_aie_ptdq_float32_profile.log
```
*_profile.log file has details about time/token latency. 

Decode prompts example

```
python run.py --model_name opt-1.3b --load --smoothquant --quant_mode w8a8 --target aie --task decode
...
...

****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?

The meaning of life is the question that is asked by many people. The meaning of life is the
****************************************

prompt: What is recursion?
response: What is recursion?

Recursion is a mathematical concept that describes the relationship between two or more mathematical operations. It is a mathematical concept
****************************************
```


### Results

#### Perplexity scores

Perplexity is measured using negative log likelihood scores.
***Perplexity measurement takes 3-4hrs to coimplete on wikitext2-raw test set***
***Lower value is better***
* Max. sequence length of model = 2048  (max. position embeddings of opt-1.3b)

Baseline: V100 (FP32) : **14.6240**

The following numbers are on RyzenAI
| **Precision+Config** | **opt-1.3b CPU** | **opt-1.3b NPU**
|---------------|----------|---------|
FP32                                                                 |  14.2637 | na
FP32, Flash Attention v2                                             |  14.9346 | na
BF16, Flash Attention v2                                             |  14.9772 | na
int8 GEMM (PTDQ), other ops FP32                                     | 231.6443 | 16.1019
int8 GEMM (SmoothQuant + PTDQ), other ops FP32                       |  15.5526 | 15.0677
int8 GEMM (SmoothQuant + PTDQ), other ops FP32, Flash Attention v2   |  15.4157 | 15.2020
int8 GEMM (SmoothQuant + PTDQ), other ops BF16                       |    na    | 14.9346
int8 GEMM (SmoothQuant + PTDQ), other ops BF16, Flash Attention v2   |    na    | 15.0854
int4 AWQ (Group size:128), bf16                                      |          | 

#### Profiling

```
python run.py --model_name opt-1.3b --target cpu --quant_mode none --task decode --amdopt

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:4.664s       Prefill-phase:727.269ms Time/token:186ms        Tokens/sec:5.4
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:4.641s       Prefill-phase:765.520ms Time/token:192ms        Tokens/sec:5.2
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:4.598s       Prefill-phase:747.972ms Time/token:183ms        Tokens/sec:5.5
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:4.438s       Prefill-phase:694.544ms Time/token:177ms        Tokens/sec:5.7
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:4.623s       Prefill-phase:588.178ms Time/token:175ms        Tokens/sec:5.7
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:4.683s       Prefill-phase:558.829ms Time/token:179ms        Tokens/sec:5.6
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:4.407s       Prefill-phase:750.508ms Time/token:173ms        Tokens/sec:5.8
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:4.507s       Prefill-phase:676.789ms Time/token:174ms        Tokens/sec:5.8
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:4.734s       Prefill-phase:703.286ms Time/token:182ms        Tokens/sec:5.5
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:4.482s       Prefill-phase:663.980ms Time/token:171ms        Tokens/sec:5.9
```

```
python run.py --model_name opt-1.3b --target aie --quant_mode w8a8 --smoothquant --task decode --amdopt

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:4.162s       Prefill-phase:218.257ms Time/token:187ms        Tokens/sec:5.3
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:4.140s       Prefill-phase:379.779ms Time/token:184ms        Tokens/sec:5.4
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:4.151s       Prefill-phase:228.732ms Time/token:186ms        Tokens/sec:5.4
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:4.147s       Prefill-phase:218.261ms Time/token:185ms        Tokens/sec:5.4
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:4.527s       Prefill-phase:205.721ms Time/token:186ms        Tokens/sec:5.4
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:4.530s       Prefill-phase:221.394ms Time/token:186ms        Tokens/sec:5.4
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:4.153s       Prefill-phase:221.467ms Time/token:187ms        Tokens/sec:5.3
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:4.354s       Prefill-phase:221.739ms Time/token:187ms        Tokens/sec:5.4
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:4.398s       Prefill-phase:211.194ms Time/token:190ms        Tokens/sec:5.3
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:4.320s       Prefill-phase:218.778ms Time/token:186ms        Tokens/sec:5.4
```


