<table align="center" class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Generative AI on Ryzen AI Software Platform</h1>
    </td>
 </tr>
</table>


<h1 align="center">OPT1.3B - ONNX Runtime Flow</h1>



## Overview

This flow demonstrates the out-of-the-box solution provided by Vitis AI Execution Provider (EP) to run transformer models on different target devices (IPU/CPU). We provide sample scripts to download a pretrained Huggingface transformer model, quantize it using ORT dynamic quantizer, and execute it on the target using the EP. 

The flow consists of the sequence of steps enumerated below.

<br/><br/>

<div align="center">
<img src="opt_eager.png" >
<p>OPT Eager Mode Flow</p>
</div>

<br/><br/>

**Quantization Process**

  1. Load pretrained Hugging Face OPT large language model and perform SmoothQuant to condition the weights
  2. Export saved PyTorch model to ONNX using Optimum
  3. Quantize the MATMUL ops in the model using ONNX Runtime (ORT) quantizer

**Text Generation**

  4. Load the model and run inference using Vitis AI Execution Provider

<br/><br/>
**Here are the steps that you can follow to run the OPT-1.3b model on the Ryzen AI software platform**
<br/><br/>

## Step 1 - Install Ryzen AI package:

```
conda activate ryzenai-transformers
```

Please refer to the [manual installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html#) to properlly install the Ryzen AI software.

```
## On Anaconda Command Prompt, go to transformers root directory
cd ..\..

## Run setup script
.\setup.bat

## Navigate back to models/vision_transformer_onnx
cd models\vision_transformer_onnx

## Run setup for examples
.\setup.bat
```



## Step 2 - Generate ort dynamic quantized model
```
.\prepare_model.bat opt-1.3b
```
This script creates the directory `<model_name>_ortquantized`.

## Step 3 - Load model and run inference
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
  --model_name {opt-1.3b}
                        Different OPT model sizes
  --dataset {non-raw,raw}
                        wikitext2-raw-v1, wikitext2-v1
  --amdort              Use ORT from local folder - with profile instrumentation
```
Each run generates a log directory `log_<model_name>` and all logs are within this directory.

### Decode prompts
```
python run.py --model_name opt-1.3b --target cpu --local_path ./opt-1.3b_ortquantized 
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized 
```
:pushpin: Note: `XLNX_ENABLE_CACHE` env variable is used for caching and is set to 1 by default.
If it is 0, the cached executable will be ignored and the model will be recompiled.

One example of an answer is shown below.
```
Prompt: What does Xilinx do?
Response:

Xilinx is a global leader in the design and manufacture of integrated circuits, software, and services
```

### Profiling
To enable token time calculation, use `--amdort` flag. To understand more about how it is measured, please look at the detailed description [here](../opt/README.MD#profiling).

One example is given below:
```
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized --amdort

Sample Output from Profiler:
Example#:1      Prompt-len:8    New-tokens-generated:22 Total-time:3.791s       Prefill-phase:187.682ms Time/token:171ms        Tokens/sec:5.9
Example#:2      Prompt-len:9    New-tokens-generated:21 Total-time:3.772s       Prefill-phase:343.520ms Time/token:171ms        Tokens/sec:5.9
Example#:3      Prompt-len:8    New-tokens-generated:22 Total-time:3.781s       Prefill-phase:187.055ms Time/token:171ms        Tokens/sec:5.9
Example#:4      Prompt-len:8    New-tokens-generated:22 Total-time:3.784s       Prefill-phase:187.196ms Time/token:171ms        Tokens/sec:5.9
Example#:5      Prompt-len:6    New-tokens-generated:24 Total-time:4.134s       Prefill-phase:186.372ms Time/token:171ms        Tokens/sec:5.9
Example#:6      Prompt-len:6    New-tokens-generated:24 Total-time:4.125s       Prefill-phase:186.526ms Time/token:170ms        Tokens/sec:5.9
Example#:7      Prompt-len:8    New-tokens-generated:22 Total-time:3.791s       Prefill-phase:188.370ms Time/token:171ms        Tokens/sec:5.9
Example#:8      Prompt-len:7    New-tokens-generated:23 Total-time:3.960s       Prefill-phase:187.688ms Time/token:171ms        Tokens/sec:5.9
Example#:9      Prompt-len:7    New-tokens-generated:23 Total-time:3.971s       Prefill-phase:188.224ms Time/token:171ms        Tokens/sec:5.9
Example#:10     Prompt-len:7    New-tokens-generated:23 Total-time:3.954s       Prefill-phase:187.036ms Time/token:171ms        Tokens/sec:5.9
```

### Perplexity on wikitext2-raw dataset
```
python run.py --model_name opt-1.3b --target cpu --local_path ./opt-1.3b_ortquantized --perplexity
python run.py --model_name opt-1.3b --target aie --local_path ./opt-1.3b_ortquantized --perplexity
```
Perplexity with ort quantized model
```
Perplexity: 15.468582153320312
Time taken to measure ppl on RyzenAI: 6967.616829633713s
```

