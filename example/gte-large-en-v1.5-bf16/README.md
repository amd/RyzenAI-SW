<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Text Embedding </h1>
    </td>
 </tr>
</table>

## Introduction

GTE is a Long context-multilingual Text Embedding model which have been developed and published by Institute for Intelligent Computing, Alibaba Group. The model has Transformer based architecture and showcases state-of-the-art scores on various benchmarks.

For more details, refer to the Hugging Face Model Card: https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5

## Overview

The following steps outline how to deploy the model on an NPU:

- Download the model from Hugging Face and convert it to ONNX (Opset 17).
- Compile and run the model on an NPU using ONNX Runtime with the Vitis AI Execution Provider.

## Setup Instructions

Activate the conda environment created by the RyzenAI installer

```bash
conda activate <env_name>
cd <RyzenAI-SW>\example\gte-large-en-v1.5-bf16
```

## Download the GTE model

Download the model from Huggingface and convert to ONNX format

```bash
python download_model.py --model_name "Alibaba-NLP/gte-large-en-v1.5" --output_dir "models"
```

The script ``download_model.py`` downloads the model from the Hugging Face checkpoint using the model name ``Alibaba-NLP/gte-large-en-v1.5`` and converts it to ONNX format.

ONNX models will be saved as: ``models/gte-large-en-v1.5.onnx``


## Run the model on NPU

Compile and run the FP32 model on NPU

```bash
python run.py --model_path "models/gte-large-en-v1.5.onnx"
```

The ONNX Runtime Vitis AI Execution Provider compiles and runs the model on an NPU. The first-time compilation may take some time, but the compiled model is cached and used for subsequent runs.

- Configuration file is passed by the ``config_file`` provider option.
- Model cache is saved in directory specified by the ``cache_dir`` and ``cache_key`` provider options.

```python
   npu_session = ort.InferenceSession(
        model_path,
        providers=["VitisAIExecutionProvider"],
        provider_options=[{"config_file": "vitisai_config.json",
                           "cache_dir": str(cache_dir),
                           "cacheKey": "modelcachekey"}],
    )
```

After a successful run, the model outputs the number of operations offloaded to CPU/NPU. The results from the text encoder are compared with CPU results.

### Sample Output

```bash
Result: Out of all operators, some will get offloaded to CPU and rest on NPU.
[Vitis AI EP] No. of Operators :   CPU     7  VAIML  1178
[Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU  1

Model compiled Successfully
Creating NPU session
NPU Embeddings
[[ 3.4179688e-01 -7.6953125e-01 -9.4921875e-01 ...  2.8710938e-01
  -8.9843750e-01 -4.4335938e-01]
 [ 3.1250000e-01 -1.1520386e-03 -1.2343750e+00 ...  4.2724609e-02
  -1.1484375e+00  2.3046875e-01]
 [ 2.5585938e-01 -1.0107422e-01 -2.4609375e-01 ...  6.9531250e-01
  -8.7109375e-01 -2.6367188e-01]
 [-3.7597656e-02 -6.3476562e-02 -1.0390625e+00 ...  3.4960938e-01
  -5.1562500e-01  6.2890625e-01]]
Creating CPU session
CPU Embeddings
[[ 0.33007812 -0.77734375 -0.9921875  ...  0.29882812 -0.90625
  -0.4609375 ]
 [ 0.265625   -0.02111816 -1.2265625  ...  0.03173828 -1.1640625
   0.24511719]
 [ 0.21484375 -0.09570312 -0.28515625 ...  0.7109375  -0.86328125
  -0.26171875]
 [-0.07324219 -0.0456543  -1.0625     ...  0.33203125 -0.55078125
   0.64453125]]
Mean Absolute Error between CPU and NPU Embeddings:  0.016811986
```
