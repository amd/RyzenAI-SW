<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI iGPU Example </h1>
    </td>
 </tr>
</table>

# Getting started with iGPU 

This is an example showing how to run the ResNet50 model from PyTorch on AMD's integrated GPU. We will use Olive to convert the model to ONNX format, convert it to FP16 precision, and optimize it. We will then use DirectML execution provider to run the model on the iGPU. 

## Activate Ryzen AI conda environment

Create a clone of the Ryzen AI installation conda environment to add required python packages

```python
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-1.3.0
conda create --name igpu-example --clone %RYZEN_AI_CONDA_ENV_NAME%
conda activate igpu-example
```

## Install Olive 

```bash
python -m pip install -r requirements.txt
```

## Install additional dependencies for the example 

```bash
python -m olive.workflows.run --config resnet50_config.json --setup
```

## Optimize the model using Olive 

```bash
python -m olive.workflows.run --config resnet50_config.json
```

The optimized models will be available in `./torch_to_onnx-float16_conversion-perf_tuning/.`


## Run the generated model on the iGPU 

### Deployment in Python 

```bash
python predict.py
```
**_NOTE:_**  In predict.py, line 15, the iGPU device ID is enumerated as 0. For PCs with multiple GPUs, you may adjust the device_id to target a specific iGPU.

### Deployment in C++

#### Prerequisites

1. Visual Studio 2022 Community edition, ensure “Desktop Development with C++” is installed
2. cmake (version >= 3.26)
3. opencv (version=4.6.0) required for the resnet50 example

#### Install OpenCV from source 

It is recommended to build OpenCV from the source code and use static build. The following instruction installs OpenCV in the location "C:\\opencv" as an example, this can be changed by modifying `CMAKE_PREFIX_PATH` in the following cmake command. You may first change the directory to where you want to clone the OpenCV repository.

```bash
git clone https://github.com/opencv/opencv.git -b 4.6.0
cd opencv

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G "Visual Studio 17 2022" "-DCMAKE_INSTALL_PREFIX=C:\opencv" "-DCMAKE_PREFIX_PATH=C:\opencv" -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_WITH_STATIC_CRT=OFF -B build

cmake --build build --config Release
cmake --install build --config Release
```
The build files will be written to ``build\``.

#### Run Olive-optimized ResNet50 model on the iGPU

Build the given ResNet50 C++ example: 

```bash
cd cpp 
compile.bat "path/to/your/opencv/build"
```

Run inference: 

```bash
run.bat
```




