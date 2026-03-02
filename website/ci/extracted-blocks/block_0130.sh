# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\getting_started_resnet\bf16\docs\README_C++.mdx:206

usage: app.exe <onnx model> <json_config> [mode]
  mode: 'classification' (default) or 'benchmark'
-------------------------------------------------------
Performing compatibility check for VitisAI EP 1.5.0
-------------------------------------------------------
 - NPU Device ID     : 0x17f0
 - NPU Device Name   : NPU Compute Accelerator Device
 - NPU Driver Version: 32.0.203.280
Environment compatible for VitisAI EP
STX/KRK NPU device detected.

-------------------------------------------------------
Running model on CPU
-------------------------------------------------------
Creating ORT env
Initializing session options
Creating ONNX Session
ONNX model : ../models/resnet_quantized_bf16.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running 100 inferences of the model
Operation took 0.290223 seconds
Done
-------------------------------------------------------

-------------------------------------------------------
Running model on NPU
-------------------------------------------------------
Creating ORT env
Initializing session options
Configuring VAI EP
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250929 16:02:40.645891 23128 register_ssmlp.cpp:124] Registering Custom Operator: com.amd:SSMLP
I20250929 16:02:40.645891 23128 register_matmulnbits.cpp:110] Registering Custom Operator: com.amd:MatMulNBits
Creating ONNX Session
I20250929 16:02:40.806568 23128 vitisai_compile_model.cpp:1266] Vitis AI EP Load ONNX Model Success
I20250929 16:02:40.814989 23128 vitisai_compile_model.cpp:1267] Graph Input Node Name/Shape (1)
I20250929 16:02:40.814989 23128 vitisai_compile_model.cpp:1271]          input : [-1x3x32x32]
I20250929 16:02:40.814989 23128 vitisai_compile_model.cpp:1277] Graph Output Node Name/Shape (1)
I20250929 16:02:40.814989 23128 vitisai_compile_model.cpp:1281]          output : [-1x10]
[Vitis AI EP] No. of Operators :   CPU     2  VAIML   272
[Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU      1
ONNX model : ../models/resnet_quantized_bf16.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running 100 inferences of the model
Operation took 0.298719 seconds
Done
-------------------------------------------------------

Test Done.
