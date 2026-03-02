# Extracted from C:\Users\bconsolvo\code\RyzenAI-SW\docs\models-tutorials\vision\getting_started_resnet\bf16\docs\README_C++.mdx:69
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
Running classification on sample images...

--- Testing image: airplane.bin ---
Predicted class: airplane
Top 3 predictions:
  1. airplane (probability: 3.9844)
  2. ship (probability: 3.7344)
  3. automobile (probability: 1.8203)

--- Testing image: automobile.bin ---
Predicted class: truck
Top 3 predictions:
  1. truck (probability: 6.7812)
  2. automobile (probability: 5.4688)
  3. ship (probability: 0.2930)

--- Testing image: cat.bin ---
Predicted class: cat
Top 3 predictions:
  1. cat (probability: 7.8750)
  2. frog (probability: 2.8750)
  3. dog (probability: 2.1094)
--- Testing image: ship.bin ---
Predicted class: ship
Top 3 predictions:
  1. ship (probability: 9.0000)
  2. automobile (probability: 2.4688)
  3. airplane (probability: 1.7891)

--- Testing image: dog.bin ---
Predicted class: dog
Top 3 predictions:
  1. dog (probability: 6.4688)
  2. cat (probability: 5.0000)
  3. deer (probability: 2.0312)
  
Done
-------------------------------------------------------

-------------------------------------------------------
Running model on NPU
-------------------------------------------------------
Creating ORT env
Initializing session options
Configuring VAI EP
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250929 15:57:49.471036  1528 register_ssmlp.cpp:124] Registering Custom Operator: com.amd:SSMLP
I20250929 15:57:49.471036  1528 register_matmulnbits.cpp:110] Registering Custom Operator: com.amd:MatMulNBits
Creating ONNX Session
I20250929 15:57:49.685086  1528 vitisai_compile_model.cpp:1266] Vitis AI EP Load ONNX Model Success
I20250929 15:57:49.685086  1528 vitisai_compile_model.cpp:1267] Graph Input Node Name/Shape (1)
I20250929 15:57:49.685086  1528 vitisai_compile_model.cpp:1271]          input : [-1x3x32x32]
I20250929 15:57:49.685086  1528 vitisai_compile_model.cpp:1277] Graph Output Node Name/Shape (1)
I20250929 15:57:49.685086  1528 vitisai_compile_model.cpp:1281]          output : [-1x10]
[Vitis AI EP] No. of Operators :   CPU     2  VAIML   272
[Vitis AI EP] No. of Subgraphs :   NPU     1 Actually running on NPU      1
ONNX model : ../models/resnet_quantized_bf16.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running classification on sample images...

--- Testing image: airplane.bin ---
Predicted class: airplane
Top 3 predictions:
  1. airplane (probability: 4.0312)
  2. ship (probability: 3.8594)
  3. automobile (probability: 1.8594)

--- Testing image: automobile.bin ---
Predicted class: truck
Top 3 predictions:
  1. truck (probability: 7.0938)
  2. automobile (probability: 5.6875)
  3. ship (probability: 0.2910)

--- Testing image: cat.bin ---
Predicted class: cat
Top 3 predictions:
  1. cat (probability: 8.1875)
  2. frog (probability: 2.8906)
  3. dog (probability: 2.1250)
--- Testing image: ship.bin ---
Predicted class: ship
Top 3 predictions:
  1. ship (probability: 9.3125)
  2. automobile (probability: 2.6094)
  3. airplane (probability: 1.8281)

--- Testing image: dog.bin ---
Predicted class: dog
Top 3 predictions:
  1. dog (probability: 6.6250)
  2. cat (probability: 5.1250)
  3. deer (probability: 1.9297)
Done
-------------------------------------------------------

Test Done.
-------------------------------------------------------
