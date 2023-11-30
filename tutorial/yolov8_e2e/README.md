<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Ryzen™ AI Tutorials</h1>
    </td>
 </tr>
</table>

#  Yolov8 on Ryzen AI


- Version:      Ryzen AI Software v1.0 
- Support:      AMD Ryzen 7040U, 7040HS series mobile processors with Windows 11 OS.
- Last update:  30 Nov. 2023


## Table of Contents

[1 Introduction](#1-introduction)

[2 Prerequisites](#2-prerequisites)

[3 Installation](#3-installation)

[4 Quantization](#4-quantization)

[5 Implementation](#5-implementation)

[License](#license)


## 1 Introduction

[Ryzen™ AI](https://ryzenai.docs.amd.com/en/latest/index.html) is a dedicated AI accelerator integrated on-chip with the CPU cores. The AMD Ryzen™ AI SDK enables developers to take machine learning models trained in PyTorch or TensorFlow and run them on laptops powered by Ryzen AI which can intelligently optimizes tasks and workloads, freeing-up CPU and GPU resources, and ensuring optimal performance at lower power.

In this Deep Learning(DL) tutorial, you will see how to deploy the Yolov8 detection model with ONNX framework on Ryzen AI laptop.

## 2 Prerequisites

- Linux server (GPU is preferred)
- AMD Ryzen AI Laptop with Windows 11 OS
- Visual Studio 2019 (with Desktop dev c++ & MSVC v142-vs2019 x64/x86 Spectre-mitigated libs)
- Anaconda or Miniconda
- Git
- openCV (version = 4.6.0)
- glog
- gflags
- cmake (version >= 3.26)
- python (version >= 3.9) (Recommended for python 3.9.13 64bit)
- IPU driver & IPU xclbin = 1.0 release 
- voe package = 1.0 release

## 3 Installation

Please refer to the [installation instructions](https://ryzenai.docs.amd.com/en/latest/inst.html#) to properlly install the Ryzen AI software.

### Denpendencies of Yolov8

There are some more libraries you need to install for the Yolov8 inference.

#### Cmake

```Conda Prompt
# pip install cmake
```

Output:

```
Collecting cmake
  Obtaining dependency information for cmake from https://files.pythonhosted.org/packages/e0/67/3cc8ccb0cebac463033e1f8588328de32f8f85cfd9d3150c05b57b827893/cmake-3.27.4.1-py2.py3-none-win_amd64.whl.metadata
  Downloading cmake-3.27.4.1-py2.py3-none-win_amd64.whl.metadata (6.8 kB)
Downloading cmake-3.27.4.1-py2.py3-none-win_amd64.whl (34.6 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 34.6/34.6 MB 147.5 kB/s eta 0:00:00
Installing collected packages: cmake
Successfully installed cmake-3.27.4.1
```

#### OpenCV

It is recommended to build OpenCV form source code and use static build. [Git](https://git-scm.com/download/win) is required to clone the repository.

Start a `Git Bash`. In the Git Bash, clone the repository

```Git Bash
# git clone https://github.com/opencv/opencv.git -b 4.6.0
```

Switch back to the `Conda Prompt`, and compile the OpenCV source code with cmake.

```Conda Prompt
# mkdir mybuild
# cd mybuild
# cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G "Visual Studio 16 2019" '-DCMAKE_INSTALL_PREFIX=C:\Program Files\opencv' '-DCMAKE_PREFIX_PATH=.\opencv' -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_WITH_STATIC_CRT=OFF -B build -S ../
# cmake --build build --config Release
# cmake --install build --config Release
# cd ../..
```

#### gflags

In the Git Bash, clone the repository

```Git Bash
# git clone https://github.com/gflags/gflags.git
```

Switch back to the `Conda Prompt`, and compile the gflags source code with cmake.

```Conda Prompt
# cd gflags
# mkdir mybuild
# cd mybuild
# cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G "Visual Studio 16 2019" '-DCMAKE_INSTALL_PREFIX=C:\Program Files\gflag'  -B build -S ../
# cmake --build build --config Release
# cmake --install build --config Release
# cd ../..
```

#### glog

In the Git Bash, clone the repository

```Git Bash
# git clone https://github.com/google/glog.git
```

Switch back to the `Conda Prompt`, and compile the glog source code with cmake.

```Conda Prompt
# cd glog
# mkdir mybuild
# cd mybuild
# cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G "Visual Studio 16 2019" '-DCMAKE_INSTALL_PREFIX=C:\Program Files\glog'  -B build -S ../
# cmake --build build --config Release
# cmake --install build --config Release
# cd ../..
```

All the dependencies on the Ryzen AI laptop are installed completely. User could run a end to end Yolov8 deplomyment progress with the following ***Section 4***, which will start from the FP32 Yolov8 model. The whole progress will last for several hours or one day depending on the hardware computing ability. 

Alternatively, user who wants a quick benchmark could skip ***Section 4*** and start from ***Section 5*** with pre-quantized model.

## 4 Quantization

In this section, we will leverage the Ryzen AI docker container on Linux GPU server for a quantized awared training(QAT).

Please follow the instrucion [here](https://ryzenai.docs.amd.com/en/latest/alternate_quantization_setup.html#install-pt-tf)  to build your docker container or pull prebuild docker from docker hub.

This tutorial will take GPU docker as a reference.

### Build Vitis AI GPU Docker

```Bash
$ cd <ryzen-ai-gpudockerfiles>
$ ./docker_build.sh -t gpu -f pytorch
```

### Prepare Coco Dataset

Download the COCO dataset from https://cocodataset.org/#download following the instruction and make sure the dataset structure is restored as below.
Please also update variable "DATA_PATH" in "coco.yaml" to point to the correct location.

```markdown
+ datasets/
    + coco/
        + labels/
        + annotations/
        + images/
        + test-dev2017.txt 
        + train2017.txt
        + val2017.txt
```

### Quantization
Clone the RyzenAI-SW repository.
```Bash
$ git clone https://github.com/amd/RyzenAI-SW.git
```

Start a Docker container using the image.

```Bash
$ docker run -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 8G 	xilinx/vitis-ai-pytorch-gpu:<Your-Image-Tag>
```

You can also pass the ***-v*** argument to mount any data directories from the host onto the container.

Then, setup the environment with following commands.

```Bash
$ cd RyzenAI-SW/tutorial/yolov8_e2e
$ sudo bash env_setup.sh
$ cd code
$ python3 setup.py develop
```

User could use the ***run_test.sh*** script to validate the float point model first before the quantization.

```Bash
$ bash run_test.sh
```

Then Quantize the model with following script.

```Bash
$ bash run_ptq.sh
```

Then quantize the model with QAT technique.

```Bash
$ bash run_qat.sh
```

Copy the quantized model to Ryzen AI laptop for the following deployment.

## 5 Implementation

### Compilation

If the ***section 4*** is skiped, please start a `Git Bash`. In the Git Bash, clone the repository

```Git Bash
# git clone https://github.com/amd/RyzenAI-SW.git
```

Switch back to the `Conda Prompt`, and compile the Yolov8 source code.

```Conda Prompt
# cd RyzenAI-SW/tutorial/yolov8_e2e/implement
# build.bat
```

The output will be generated as below.

```
......
    -- Installing: C:/Users/ibane/Desktop/voe-win_amd64-with_xcompiler_on-c07e419-latest/bin/camera_yolov8.exe
    -- Installing: C:/Users/ibane/Desktop/voe-win_amd64-with_xcompiler_on-c07e419-latest/bin/camera_yolov8_nx1x4.exe
    -- Installing: C:/Users/ibane/Desktop/voe-win_amd64-with_xcompiler_on-c07e419-latest/bin/test_jpeg_yolov8.exe
```

### Run with Image

To validate your setup, the following command will do the inference with single image.

Please modify the ***conda env name*** in the batch file before execution.

```
# run_jpeg.bat DetectionModel_int.onnx sample_yolov8.jpg
```

The output will be generated as below.

```
result: 0       person  490.38498       85.79535        640.00488       475.18262       0.932453     
result: 0       person  65.96048        97.76373        320.66068       473.83783       0.924142   
result: 0       person  182.15485       306.91266       445.14795       475.26132       0.893309   
result: 27      tie     584.48022       221.15732       632.27008       244.21243       0.851953   
result: 27      tie     175.62622       224.15210       235.84900       248.83557       0.651355    
```

### Run with Live Camera

To run with live camera, user needs to change the display and camera settings manually as below.

Please modify the ***conda env name*** in the batch file before execution.

- Go to `Display settings`, change Scale to ***100%*** in the `Scale & layout` section.
- Go to `Bluetooth & devices` -> `Cameras` -> `USB2.0 FHD UVC WebCam`, turn off the Background effects in the `Windows Studio Effects` section.

```
camera.bat
```

<p align="left">
<img src="images/image.png">
</p>

Possible options to run the yolov8 demo.

```
# camera.bat -h

Options:
      -c [parallel runs]: Specifies the (max) number of runs to invoke simultaneously. Default:1.
      -s [input_stream] set input stream, E.g. set 0 to use default camera.
      -x [intra_op_num_threads]: Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.
      -y [inter_op_num_threads]: Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default. Must >=0.    
      -D [Disable thread spinning]: disable spinning entirely for thread owned by onnxruntime intra-op thread pool.
      -Z [Force thread to stop spinning between runs]: disallow thread from spinning during runs to reduce cpu usage.
      -T [Set intra op thread affinities]: Specify intra op thread affinity string.
         [Example]: -T 1,2;3,4;5,6 or -T 1-2;3-4;5-6
         Use semicolon to separate configuration between threads.
         E.g. 1,2;3,4;5,6 specifies affinities for three threads, the first thread will be attached to the first and second logical processor.
      -R [Set camera resolution]: Specify the camera resolution by string.
         [Example]: -R 1280x720
         Default:1920x1080.
      -r [Set Display resolution]: Specify the display resolution by string.
         [Example]: -r 1280x720
         Default:1920x1080.
      -L Print detection log when turning on.
      -h: help
```

## License

The MIT License (MIT)

Copyright (c) 2022 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
