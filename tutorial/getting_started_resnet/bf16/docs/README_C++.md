# ResNet18 CIFAR-10 Inference Example with C++

This example demonstrates how to run inference with a ResNet18 model trained on CIFAR-10 dataset using ONNX Runtime in C++.

**Note:** Ensure that you are following the instructions for model and dataset setup and compilation from [BF16 setup](../README.md)

## Requirements

- Visual Studio 2022 (for building the C++ application)
- CMake
- ResNet18 CIFAR-10 model in ONNX format

### Step 1: Preparing Test Images

Before running the application in classification mode, you'll need to prepare test images:


```python
cd app
python prepare_test_images.py
```

This script will:

- Download the CIFAR-10 dataset
- Extract sample images from each class
- Save the images in the required binary format in the `test_images` directory

###  Step 2:

```bash
compile.bat
```

This batch file will execute below two commands one after another.

```bash
cmake -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -B build -S . -G "Visual Studio 17 2022"
cmake --build .\build --config Release --target ALL_BUILD
```

The build step will create a build / folder, and all the system files (such as object files and intermediate builds files) will be generated inside it using the .cpp source file.
The build process will then create the final executable file from the compiled object files.

The CMakeLists.txt ensures that all required runtime DLLs are copied in the build\Release folder at build time. The compile.bat script takes care of copying the ONNX model, the VAIML compiled model folder and the JSON config file in the build\Release folder. The executable looks for these files in this specific directory.

## Step 3:

The application supports two modes:

1. **Classification Mode (default)**: Runs inference on sample CIFAR-10 images and prints the predicted class labels.
2. **Benchmark Mode**: Runs multiple iterations of inference to measure performance.

## Usage

```bash
build\Release\app.exe [model_path] [config_path] [mode]
```

- `model_path`: Path to the ONNX model (default: `models/resnet_trained_for_cifar10.onnx`)
- `config_path`: Path to the Vitis AI configuration file (default: `vitisai_config.json`)
- `mode`: Either `classification` (default) or `benchmark`

### Examples

Run in classification mode (default):

```bash
build\Release\app.exe ../models/resnet_trained_for_cifar10.onnx ../vitisai_config.json classification
```

In classification mode, the application will:

1. Load test images from the `test_images` directory
2. Run inference using the model
3. Print the predicted class and top-3 predictions with probabilities

The output from the run command will look like below.

```bash
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
ONNX model : ..\models\resnet_trained_for_cifar10.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running classification on sample images...

--- Testing image: airplane.bin ---
Predicted class: airplane
Top 3 predictions:
  1. airplane (probability: 4.0282)
  2. ship (probability: 3.7459)
  3. automobile (probability: 1.8095)

--- Testing image: automobile.bin ---
Predicted class: truck
Top 3 predictions:
  1. truck (probability: 6.8240)
  2. automobile (probability: 5.5014)
  3. ship (probability: 0.2835)

--- Testing image: cat.bin ---
Predicted class: cat
Top 3 predictions:
  1. cat (probability: 7.8910)
  2. frog (probability: 2.8882)
  3. dog (probability: 2.1396)

--- Testing image: ship.bin ---
Predicted class: ship
Top 3 predictions:
  1. ship (probability: 9.0202)
  2. automobile (probability: 2.4488)
  3. airplane (probability: 1.8038)

--- Testing image: dog.bin ---
Predicted class: dog
Top 3 predictions:
  1. dog (probability: 6.5393)
  2. cat (probability: 5.0560)
  3. deer (probability: 2.0006)
Done
-------------------------------------------------------

-------------------------------------------------------
Running model on NPU
-------------------------------------------------------
Creating ORT env
Initializing session options
Configuring VAI EP
Creating ONNX Session
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250626 17:14:45.115626 46400 vitisai_compile_model.cpp:1157] Vitis AI EP Load ONNX Model Success
I20250626 17:14:45.115626 46400 vitisai_compile_model.cpp:1158] Graph Input Node Name/Shape (1)
I20250626 17:14:45.115626 46400 vitisai_compile_model.cpp:1162]          input : [-1x3x32x32]
I20250626 17:14:45.115626 46400 vitisai_compile_model.cpp:1168] Graph Output Node Name/Shape (1)
I20250626 17:14:45.115626 46400 vitisai_compile_model.cpp:1172]          output : [-1x10]
[Vitis AI EP] No. of Operators : VAIML   124
[Vitis AI EP] No. of Subgraphs : VAIML     1
ONNX model : ..\models\resnet_trained_for_cifar10.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running classification on sample images...

--- Testing image: airplane.bin ---
Predicted class: airplane
Top 3 predictions:
  1. airplane (probability: 3.9062)
  2. ship (probability: 3.7812)
  3. automobile (probability: 1.8359)

--- Testing image: automobile.bin ---
Predicted class: truck
Top 3 predictions:
  1. truck (probability: 6.9375)
  2. automobile (probability: 5.6875)
  3. ship (probability: 0.2637)

--- Testing image: cat.bin ---
Predicted class: cat
Top 3 predictions:
  1. cat (probability: 8.1250)
  2. frog (probability: 3.0000)
  3. dog (probability: 2.1562)

--- Testing image: ship.bin ---
Predicted class: ship
Top 3 predictions:
  1. ship (probability: 9.0625)
  2. automobile (probability: 2.4219)
  3. airplane (probability: 1.7969)

--- Testing image: dog.bin ---
Predicted class: dog
Top 3 predictions:
  1. dog (probability: 6.4375)
  2. cat (probability: 5.0625)
  3. deer (probability: 1.8594)
Done
-------------------------------------------------------

Test Done.
```

Run in benchmark mode:

```bash
build\Release\app.exe ../models/resnet_trained_for_cifar10.onnx ../vitisai_config.json benchmark
```

In benchmark mode, the application will:

1. Run 100 inferences with random input data
2. Measure and report the total execution time

The output from the run command will look like below.

```bash
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
ONNX model : ..\models\resnet_trained_for_cifar10.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running 100 inferences of the model
Operation took 0.198749 seconds
Done
-------------------------------------------------------

-------------------------------------------------------
Running model on NPU
-------------------------------------------------------
Creating ORT env
Initializing session options
Configuring VAI EP
Creating ONNX Session
WARNING: Logging before InitGoogleLogging() is written to STDERR
I20250626 17:13:40.685429 78924 vitisai_compile_model.cpp:1157] Vitis AI EP Load ONNX Model Success
I20250626 17:13:40.685429 78924 vitisai_compile_model.cpp:1158] Graph Input Node Name/Shape (1)
I20250626 17:13:40.685429 78924 vitisai_compile_model.cpp:1162]          input : [-1x3x32x32]
I20250626 17:13:40.685429 78924 vitisai_compile_model.cpp:1168] Graph Output Node Name/Shape (1)
I20250626 17:13:40.685429 78924 vitisai_compile_model.cpp:1172]          output : [-1x10]
[Vitis AI EP] No. of Operators : VAIML   124
[Vitis AI EP] No. of Subgraphs : VAIML     1
ONNX model : ..\models\resnet_trained_for_cifar10.onnx
  input -1x3x32x32
  output -1x10
Dynamic batch size detected. Setting batch size to 1.
Running 100 inferences of the model
Operation took 0.750481 seconds
Done
-------------------------------------------------------

Test Done.
```