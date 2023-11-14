<table align="center" class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Generative AI on Ryzen AI Software</h1>
    </td>
 </tr>
</table>


<h1 align="center">OPT1.3B - PyTorch Flow</h1>

## Overview

This example shows how to deploy OPT models on IPU using custom IPU+CPU backend in **eager mode**. The flow consists of the sequence of steps enumerated below.
1. Load FP32 or BFloat16 OPT model
2. Perform state-of-the-art [SmoothQuant](https://arxiv.org/pdf/2211.10438.pdf) to condition weights before quantization.
3. Quantize the model using PyTorch's dynamic quantization
4. Identify quantized linear nodes in the model
5. Replace each quantized linear node with a custom Linear node (QLinear)
6. Initialize weights through Torch's PackedParams
7. Pad, tile and cache weights as initialization step
8. Run inference - benchmark, decode or measure perplexity

In this flow, compute primitives are choses to optimize prompt/prefill phase and token phase, both independently.
<br/><br/>

<div align="center">
<img src="EagerMode.png" >
<p>OPT PyTorch Eager Mode Flow</p>
</div>

<br/><br/>

**Model Structure**

This image shows model structure after quantizing and replacing Linear node with IPU custom linear node.
![Model Structure](opt1.3b.png)

**Model Variants**

OPT has several variants: **opt-125m, opt-350m, opt-1.3b, opt-2.7b, opt-6.7b, opt-13b, etc.**. The user can select any model variant for this example but the upper limit is dependent on the system memory capacity. ***For Ryzen-AI, limit to opt1.3b***. 

Max. sequence length of model is 2048  (max. position embeddings), i.e. maximum number of tokens in input prompt. 

## Step 1: Prerequisites

### Prereq 1 - Create and activate a conda environment using the env.yaml file provided

```
conda env create --file=env.yaml
conda activate ryzenai-transformers
```

### Prereq 2 - Set environment variables 

```
.\setup_pytorch.bat
```

### Prereq 3 - Build dependencies

```
pip install ..\ops\cpp --force-reinstall
```

### Prereq 4 [Optional] - Run Python Unit test cases 

Run the Python unit tests: 
```
cd opt-1.3b\tests\python
pytest --num_dlls 1 --num_workers 1 test_qlinear.py
pytest --num_dlls 1 --num_workers 2 test_qlinear.py
pytest --num_dlls 2 --num_workers 1 test_qlinear.py
pytest --num_dlls 2 --num_workers 2 test_qlinear.py
pytest test_opt_flash_attention.py
```

## Quantize and save model weights
### Step 1- Quantize and save model weights
```
cd opt-1.3b\opt-pytorch
python save_weights.py --action save --model_name opt-1.3b
```
This saves "quantized_<model_name>_float32.pth" with all linear ops dynamically quantized to int8, while all other ops are in float32. We use this model for text generation below.

## Text generation

The inference script run.py gives options to do the following:
* Benchmark code - measure time/token latency
* Calculate perplexity score
* Profile code using PyTorch profiler
* Decode a set of prompts to show model liveliness

Note: The script implementation deploys operators on IPU using --target aie

```
python run.py --help
usage: run.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b,opt-13b,opt-30b}] [--target {cpu,aie}] [--dtype {bfloat16,float32}] [--quant_mode {none,ptdq}] [--smoothquant] [--amdopt]
              [--dataset {non-raw,raw}] [--load] [--flash_attention] [--num_torch_threads {1,2,3,4,5,6,7,8}] [--task {decode,benchmark,benchmark_long,perplexity,torchprofile}]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b,opt-13b,opt-30b}
                        Different OPT model sizes
  --target {cpu,aie}    cpu, aie
  --dtype {bfloat16,float32}
                        All ops other than linear ops in bfloat16 or float32
  --quant_mode {none,ptdq}
                        Quantization mode - none, dynamic or static quant
  --smoothquant         Enable smoothquant
  --amdopt              Use OPT from local folder - with profile instrumentation
  --dataset {non-raw,raw}
                        Dataset - wikitext2-raw-v1, wikitext2-v1
  --load                Load quantized weights from checkpoint
  --flash_attention     Enable flash attention
  --num_torch_threads {1,2,3,4,5,6,7,8}
                        Number of torch threads
  --task {decode,benchmark,benchmark_long,perplexity,torchprofile}
                        perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode: Decode
                        set of prompts; torchprofile: profile using torch profiler for decode
```

### Logifles
Each run generates a log directory "log_<model_name>" and all logs are within this directory. 

Each run generatses 2 log files in the "log_<model_name>" directory

For example:
```
logs_opt-1.3b\log_opt-1.3b_aie_ptdq_float32.log
logs_opt-1.3b\log_opt-1.3b_aie_ptdq_float32_profile.log
```
*_profile.log file has details about time/token latency. 

### Decode prompts example
```
python run.py --model_name opt-1.3b --load --smoothquant --quant_mode ptdq --target aie --task decode
...
...

****************************************
prompt: What is the meaning of life?
response: What is the meaning of life?

The meaning of life is the question that is asked by many people. The meaning of life is the
****************************************
prompt: Who is Gilgamesh?
response: Who is Gilgamesh?

Gilgamesh is a character from the epic poem Gilgamesh. He is the son of the god
****************************************
prompt: What is recursion?
response: What is recursion?

Recursion is a mathematical concept that describes the relationship between two or more mathematical operations. It is a mathematical concept
****************************************
```

### Perplexity scores
[Perplexity](https://huggingface.co/docs/transformers/perplexity) is a metric commonly used to evaluate language models. It represents the model's ability to generate uniformly among the set of specified tokens in the corpus. Specifically, it quantifies how surprised or uncertain the model is when trying to predict the next word in a sequence based on the context of the previous words. A lower perplexity indicates that the model is better at predicting the next word, while a higher perplexity suggests that the model struggles with the prediction.

* Max. sequence length of model = 2048  (max. position embeddings of opt-1.3b)

Baseline: V100 (FP32) : **14.6240**

The following numbers are on RyzenAI
| **Precision+Config** | **opt-1.3b CPU** | **opt-1.3b IPU**
|---------------|----------|---------|
FP32                                                                 |  14.2637 | na
FP32, Flash Attention v2                                             |  14.9346 | na
BF16, Flash Attention v2                                             |  14.9772 | na
int8 GEMM (PTDQ), other ops FP32                                     | 231.6443 | 16.1019
int8 GEMM (SmoothQuant + PTDQ), other ops FP32                       |  15.5526 | 15.0677
int8 GEMM (SmoothQuant + PTDQ), other ops FP32, Flash Attention v2   |  15.4157 | 15.2020
int8 GEMM (SmoothQuant + PTDQ), other ops BF16                       |    na    | 14.9346
int8 GEMM (SmoothQuant + PTDQ), other ops BF16, Flash Attention v2   |    na    | 15.0854

### Profiling
To get these numbers with timer instrumented in the model use "--amdopt" option.

The token time is calculation is explained as follows:

![Token time](./opt-1.3b_token_time_rel.png)


### Example decode with short prompt 

First example in this test was used to explain the profiling distribution in the above figure.

```
python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --task decode --smoothquant --amdopt

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:4.763s       Prefill-phase:283.918ms Time/token:212ms        Tokens/sec:4.7
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:4.685s       Prefill-phase:438.123ms Time/token:212ms        Tokens/sec:4.7
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:4.698s       Prefill-phase:291.086ms Time/token:209ms        Tokens/sec:4.8
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:4.711s       Prefill-phase:275.517ms Time/token:211ms        Tokens/sec:4.7
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:5.218s       Prefill-phase:271.073ms Time/token:214ms        Tokens/sec:4.7
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:5.159s       Prefill-phase:257.795ms Time/token:213ms        Tokens/sec:4.7
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:4.752s       Prefill-phase:269.300ms Time/token:213ms        Tokens/sec:4.7
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:4.914s       Prefill-phase:267.731ms Time/token:211ms        Tokens/sec:4.7
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:4.925s       Prefill-phase:272.001ms Time/token:211ms        Tokens/sec:4.7
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:4.935s       Prefill-phase:273.164ms Time/token:211ms        Tokens/sec:4.7
```

### Profiling with larger prompt lengths - (8-256)
```
python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --task benchmark --smoothquant --amdopt

Example#:1      Prompt-len:8    New-tokens-generated:11 Total-time:2.427s       Prefill-phase:296.159ms         Time/token:212ms        Tokens/sec:4.7
Example#:2      Prompt-len:16   New-tokens-generated:11 Total-time:2.612s       Prefill-phase:514.400ms         Time/token:209ms        Tokens/sec:4.8
Example#:3      Prompt-len:32   New-tokens-generated:11 Total-time:3.127s       Prefill-phase:983.083ms         Time/token:214ms        Tokens/sec:4.7
Example#:4      Prompt-len:64   New-tokens-generated:11 Total-time:4.210s       Prefill-phase:1982.016ms        Time/token:221ms        Tokens/sec:4.5
Example#:5      Prompt-len:128  New-tokens-generated:11 Total-time:6.356s       Prefill-phase:4121.728ms        Time/token:223ms        Tokens/sec:4.5
Example#:6      Prompt-len:256  New-tokens-generated:11 Total-time:10.797s      Prefill-phase:8438.292ms        Time/token:232ms        Tokens/sec:4.3
```

### Profiling with larger prompt lengths (512-2000)
```
python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --task benchmark_long --smoothquant --amdopt

Example#:1      Prompt-len:512  New-tokens-generated:11 Total-time:21.291s      Prefill-phase:18675.431ms       Time/token:257ms        Tokens/sec:3.9
Example#:2      Prompt-len:1024 New-tokens-generated:11 Total-time:47.637s      Prefill-phase:44667.306ms       Time/token:287ms        Tokens/sec:3.5
Example#:3      Prompt-len:1536 New-tokens-generated:11 Total-time:83.366s      Prefill-phase:79917.232ms       Time/token:328ms        Tokens/sec:3.0
Example#:4      Prompt-len:2000 New-tokens-generated:11 Total-time:123.110s     Prefill-phase:119394.423ms      Time/token:352ms        Tokens/sec:2.8
```

### Profiling with larger prompt lengths (512-2000) - with flash attention v2

With Flash Attention, prompt phase improves by ~30%.

```
python run.py --model_name opt-1.3b --target aie --quant_mode ptdq --task benchmark_long --smoothquant --amdopt --flash_attention

Example#:1      Prompt-len:512  New-tokens-generated:11 Total-time:18.046s      Prefill-phase:15486.544ms       Time/token:255ms        Tokens/sec:3.9
Example#:2      Prompt-len:1024 New-tokens-generated:11 Total-time:36.085s      Prefill-phase:33059.042ms       Time/token:293ms        Tokens/sec:3.4
Example#:3      Prompt-len:1536 New-tokens-generated:11 Total-time:57.453s      Prefill-phase:54065.473ms       Time/token:321ms        Tokens/sec:3.1
Example#:4      Prompt-len:2000 New-tokens-generated:11 Total-time:79.283s      Prefill-phase:75502.877ms       Time/token:357ms        Tokens/sec:2.8
```



## Text completion using the OPT model
The opt_demo.py script gives the user options to run the model on any set or prompts with 3 search strategies
```
python opt_demo.py --help
usage: opt_demo.py [-h] [--model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}] [--target {cpu,aie}] [--quant_mode {none,ptdq}] [--load]

optional arguments:
  -h, --help            show this help message and exit
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b,opt-6.7b}
                        Different OPT model sizes
  --target {cpu,aie}    cpu, aie
  --quant_mode {none,ptdq}
                        Quantization mode - none, or smoothquant+pytorch dynamic-quant
  --load                Load quantized weights from checkpoint
```

The demo gives user flexibility to provide any prompts, with different search options and output token lengths.
Three search options are provided: ***greedy, stochastic and contrastive***. These search options provide different level of quality to text generation process. User can modify the strengths of parameters in this file. 

***[Optional] Enable printing tokens as they are generated. This feature is available without --load option only:***

This is optional, to see individual tokens as they print to the screen in greedy search mode, open the 

```installationfolder\anaconda3\envs\ryzenai-transformers\lib\site-packages\transformers\generation\utils.py```

In ```def greedy_search(...)``` function,  

after ```next_tokens = torch.argmax(next_tokens_scores, dim=-1)```, 

add this new line: ```print(self.tokenizer.decode(next_tokens)) ```

This prints each new token to the screen as the text-generation process unfolds. 

```
python opt_demo.py  --quant_mode ptdq --target aie --load
python opt_demo.py  --quant_mode ptdq --target cpu --load
python opt_demo.py  --quant_mode none --target cpu
```

Here are examples for 3 search options for the same prompt and token length on IPU with SmoothQuant + Pytorch Dynamic Quantization:

```
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 0
Setting search to:  Greedy search
San Francisco is a city of contrasts. It’s a city of the arts, of the food, of the people, of the history
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 1
Setting search to:  Stochastic search
San Francisco is a city of incredible contrasts. It has the highest concentration of Jews of any city in the world, and it is known as a
********************
Enter prompt or 'exit': San Francisco is a city of
Enter response length (1-1000): 30
Enter 0(greedy search) 1(stochastic search) or 2(contrastive search): 2
Setting search to:  Contrastive search
San Francisco is a city of many cultures.

The city has a long history of immigration and is home to the largest number of immigrants in
********************
```
