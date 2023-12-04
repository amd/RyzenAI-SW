<table style="width:100%">
  <tr>

<th width="100%" colspan="6"><img src="https://github.com/Xilinx/Image-Collateral/blob/main/xilinx-logo.png?raw=true" width="30%"/><h1>Multi Model Example </h1>

</tr>

</table>

## Table of Contents

- [1 Introduction](#1-Introduction)
- [2 Design Files](#2-Design-Files)
- [3 How to Build](#3-How-To-Build)
    - [3.1 Requirement](#31-Requirement)
    - [3.2 Clean Cache](#32-Clean-Cache)
    - [3.3 Prepare Conda Env](#33-Prepare-Conda-Env)
    - [3.4 Install python package](#34-Install-python-package)
    - [3.5 Compile the Application](#35-Compile-the-Application)
- [4 Run The Demos](#3-Run-The-Demos)

## 1 Introduction
This example serves as a practical guide, illustrating the step-by-step procedure for installing the required dependencies and successfully building the demo from the source code。

## 2 Design Files

```bash
.
├── CMakeLists.txt
├── README.md
├── build.bat #easy build script
└── src
    ├── CMakeLists.txt
    ├── app  #ipu_multi_model app source code
    ├── models
    ├── onnx # Onnx dependencies
    ├── processing
    └── util
```
## 3 How To Build:

### 3.1 Requirement
1. Visual Studio 2019 or 2022 (with Desktop dev c++ )
2. cmake (version >= 3.26)
3. python (version >= 3.9) (Recommended for python 3.9.13 64bit)
4. IPU driver & IPU xclbin reledease >= 20230823



### 3.2 Clean Cache 
When you replace the IPU driver or 1x4.xclbin, you need to clear the cache of the old compiled model, they are located in C:\temp\rd\vaip\\.cache, and delete everything under this folder.

### 3.3 Prepare Conda Env
About how to create conda env please refer to [Demo Readme](../../demo/multi-model-exec/README.md)

clone Opencv and Glog
1. opencv (version=4.6.0)
```
git clone https://github.com/opencv/opencv.git -b 4.6.0
cd opencv
mkdir mybuild
cd mybuild
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G 'Visual Studio 16 2019' '-DCMAKE_INSTALL_PREFIX=C:\Program Files\opencv' '-DCMAKE_PREFIX_PATH=.\opencv' -DCMAKE_BUILD_TYPE=Release -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_WITH_STATIC_CRT=OFF -B build -S ../
cmake --build build --config Release
cmake --install build --config Release
``` 
1. build gflags & glog
```
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir mybuild
cd mybuild
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G 'Visual Studio 17 2022' '-DCMAKE_INSTALL_PREFIX=C:\Program Files\gflag'  -B build -S ../
cmake --build build --config Release
cmake --install build --config Release
cd ../..
git clone https://github.com/google/glog.git
cd glog
mkdir mybuild
cd mybuild
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_CONFIGURATION_TYPES=Release -A x64 -T host=x64 -G 'Visual Studio 17 2022' '-DCMAKE_INSTALL_PREFIX=C:\Program Files\glog'  -B build -S ../
cmake --build build --config Release
cmake --install build --config Release
```

### 3.4 Install python package
1. opencv
```
python3 -m pip install opencv-python
```
2. pillow
```
python3 -m pip install pillow
```

### 3.5 Compile the Application

 We provide an easy script for compiling and installing the application. After the compilation stage is complete, you will find the executable file generated at ***bin/ipu_multi_models.exe***.

  ```bash
  .\build.bat
  ```
Output:
``` ......
    -- Installing: ..../bin/ipu_multi_models.exe
```


## 4 Run The Demos(Optional)

If you want to run the demo, please follow the [instruction](../../demo/multi-model-exec/README.md) in the multi-model-exec directory. 
