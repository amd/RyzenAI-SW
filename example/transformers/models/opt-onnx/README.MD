# OPT - ONNX

**_NOTE:_**  Please ensure that you followed the environment setup instructions from the [Transformer folder readme](../../README.md) before following the steps here.

The OPT-ONNX flow uses an eager mode execution flow which consists of the sequence of steps as enumerated below.
1. Generate ort dynamic quantized model
2. Load model and run inference


## Quantization

- Go to the opt-onnx folder
```
cd models/opt-onnx
```

- Generate ort dynamic quantized model
```
.\prepare_model.bat opt-1.3b
```
This script creates the directory `<model_name>_ortquantized`.

## Deployment

Load model and run inference

This script gives option to do the following:
* Benchmark code - measure time/token latency
* Calculate perplexity score
* Profile code
* Decode a set of prompts to show model liveliness

Run the following batch script to set the required environment variables:

  ```
  .\set_opt_onnx_env.bat opt-1.3b
  ```

Then, you can use `run.py` to run the model.
```
python run.py --help
usage: run.py [-h] [--local_path LOCAL_PATH] [--target {cpu,aie}] [--perplexity]
              [--model_name {opt-1.3b}] [--dataset {non-raw,raw}]

options:
  -h, --help            show this help message and exit
  --local_path LOCAL_PATH
                        Local directory path to ONNX model
  --target {cpu,aie}    cpu, aie
  --perplexity          Calculate perplexity on wikitext2 instead of decoding prompts
  --model_name {opt-125m,opt-350m,opt-1.3b,opt-2.7b}
                        Different OPT model sizes
  --quant_mode {w8a8,none}
                        Quantization mode - w8a8
  --dataset {non-raw,raw}
                        wikitext2-raw-v1, wikitext2-v1
  --amdort              Use ORT from local folder - with profile instrumentation
  --profile             Log matmul times for prompt and token phases - supported only for AIE target
  --impl {v0,v1}        Choose between different implementations for aie target
  --task {decode,benchmark,benchmark_long,perplexity}
                        perplexity: Measure perplexity on wikitext2 dataset; benchmark: Benchmark latency w.r.t prompt length; benchmark_long: Benchmark long sequences (compare with flash attn); decode:
                        Decode set of prompts
  --num_intra_op_threads {0,1,2,3,4,5,6,7,8}
                        Number of intra op num threads
  --num_inter_op_threads {0,1,2,3,4,5,6,7,8}
                        Number of inter op num threads
  --intra_op_spinning {0,1}
                        Disable intra op spinning
  --inter_op_spinning {0,1}
                        Disable inter op spinning
```
Each run generates a log directory `log_<model_name>` and all logs are within this directory.

- Decode prompts
```
python run.py --model_name opt-1.3b --target cpu --local_path ./opt-1.3b_ortquantized 
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized 
```
:pushpin: Note: `XLNX_ENABLE_CACHE` env variable is used for caching and is set to 1 by default.
If it is 0, the cached executable will be ignored and the model will be recompiled.

One example of an answer is shown below.
```
prompt: What is recursion?
response: What is recursion?

Recursion is a mathematical concept that allows you to solve a problem by recursively solving it.
```

## Result 

### Profiling
To enable token time calculation, use `--amdort` flag. 

```
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized --amdort --impl v0 --task decode --profile

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:3.609s       Prefill-phase:179.272ms Time/token:162ms        Tokens/sec:6.2
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:3.593s       Prefill-phase:330.536ms Time/token:162ms        Tokens/sec:6.2
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:3.607s       Prefill-phase:176.843ms Time/token:162ms        Tokens/sec:6.2
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:3.581s       Prefill-phase:186.321ms Time/token:161ms        Tokens/sec:6.2
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:3.937s       Prefill-phase:180.061ms Time/token:162ms        Tokens/sec:6.2
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:3.937s       Prefill-phase:177.206ms Time/token:162ms        Tokens/sec:6.2
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:3.594s       Prefill-phase:178.153ms Time/token:162ms        Tokens/sec:6.2
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:3.765s       Prefill-phase:186.603ms Time/token:162ms        Tokens/sec:6.2
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:3.752s       Prefill-phase:180.150ms Time/token:161ms        Tokens/sec:6.2
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:3.765s       Prefill-phase:176.219ms Time/token:162ms        Tokens/sec:6.2
```

```
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized --amdort --impl v1 --task decode --profile 

Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:2.166s       Prefill-phase:107.030ms Time/token:97ms Tokens/sec:10.3
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:2.087s       Prefill-phase:109.505ms Time/token:98ms Tokens/sec:10.2
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:2.182s       Prefill-phase:107.662ms Time/token:98ms Tokens/sec:10.2
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:2.228s       Prefill-phase:107.579ms Time/token:100ms        Tokens/sec:10.0
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:2.369s       Prefill-phase:109.470ms Time/token:97ms Tokens/sec:10.3
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:2.384s       Prefill-phase:107.331ms Time/token:98ms Tokens/sec:10.2
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:2.180s       Prefill-phase:107.233ms Time/token:98ms Tokens/sec:10.2
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:2.290s       Prefill-phase:107.330ms Time/token:98ms Tokens/sec:10.2
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:2.353s       Prefill-phase:106.399ms Time/token:101ms        Tokens/sec:9.9
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:2.290s       Prefill-phase:106.604ms Time/token:98ms Tokens/sec:10.2

```

### Perplexity on wikitext2-raw dataset

```
python run.py --model_name opt-1.3b --target cpu --local_path ./opt-1.3b_ortquantized --task perplexity
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized --task perplexity
```
Perplexity with ort quantized model
```
Perplexity: 15.468582153320312
Time taken to measure ppl on RyzenAI: 6967.616829633713s
```
