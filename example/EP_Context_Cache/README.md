<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI EP Context Cache Example </h1>
    </td>
 </tr>
</table>

# Getting started with Ryzen AI EP Context Cache

This is an example showing how to compile and run the ResNet18 model from https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx
 on AMD's Ryzen AI NPU by new convenient EP Context Cache with ease of usage (start from Ryzen AI 1.5). 



# Activate Ryzen AI conda environment


```bash
#Install Ryzen AI msi with relative NPU driver
conda activate ryzen-ai-1.x
```

# Generate EP Context Cache directly, no need to quantize firstly

```bash
# https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx

python compile.py resnet18-v2-7.onnx

WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250729 08:31:30.094357 12396 vitisai_compile_model.cpp:1157] Vitis AI EP Load ONNX Model Success
I20250729 08:31:30.094357 12396 vitisai_compile_model.cpp:1158] Graph Input Node Name/Shape (1)
I20250729 08:31:30.094357 12396 vitisai_compile_model.cpp:1162]          data : [-1x3x224x224]
I20250729 08:31:30.095352 12396 vitisai_compile_model.cpp:1168] Graph Output Node Name/Shape (1)
I20250729 08:31:30.095352 12396 vitisai_compile_model.cpp:1172]          resnetv22_dense0_fwd : [-1x1000]
Adding RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\... to installation search path
 
subpartition path = ....\resnet18-ep-context\resnet18-v2-7\vaiml_par_0\0
[Vitis AI EP] No. of Operators : VAIML    60
[Vitis AI EP] No. of Subgraphs : VAIML     1

# EP context cache model is saved as resnet18-v2-7_ctx.onnx

```

# Run Inference with EP Context Cache
```bash

python run.py resnet18-v2-7_ctx.onnx
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250729 08:39:23.587743  2180 vitisai_compile_model.cpp:1157] Vitis AI EP Load ONNX Model Success
I20250729 08:39:23.587743  2180 vitisai_compile_model.cpp:1158] Graph Input Node Name/Shape (1)
I20250729 08:39:23.587743  2180 vitisai_compile_model.cpp:1162]          data : [-1x3x224x224]
I20250729 08:39:23.587743  2180 vitisai_compile_model.cpp:1168] Graph Output Node Name/Shape (1)
I20250729 08:39:23.587743  2180 vitisai_compile_model.cpp:1172]          resnetv22_dense0_fwd : [-1x1000]
[Vitis AI EP] No. of Subgraphs supported by Vitis AI EP: VAIML     1
Top 3 Probabilities
[208 207 176]
------------------------------------|------------
Classification                      |Percentage
------------------------------------|------------
Labrador retriever                  |   67.43
------------------------------------|------------
golden retriever                    |    9.13
------------------------------------|------------
Saluki, gazelle hound               |    8.05
------------------------------------|------------
INFO: Test passed

```




