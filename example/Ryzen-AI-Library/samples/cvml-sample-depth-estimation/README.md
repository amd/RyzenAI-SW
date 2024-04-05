# Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.


## Running Depth-Estimation in Conda env
> NOTE: this step should come after end-user has ran **install.bat** 
0. Ensure ```C:\temp\Administrator\vaip\.cache\``` is empty (Running the sample-app without might otherwise erase other compiled models)

1. Open the command prompt and set PATH to the required dependencies:
    ```
    set PATH=<path\to\conda>;<path\to\conda>\condabin;<path\to\Ryzen-AI-Library-Public-Release>\windows;C;\Windows\System32\AMD
    ```
2. Activate the conda environment created when running install.bat
    ```
    conda activate <your_env>
    ```
3. Set ``PYTHONPATH`` and ```PYTHONHOME``` as the path to your conda environment
    ```
    set PYTHONHOME=<path\to\conda>\envs\<your_env>
	set PYTHONPATH=<path\to\conda>\envs\<your_env>
    ```
4. Move ``vaip_config.json`` from ``path\to\ryzen-ai-sw-1.0\voe-4.0-win_amd64\`` to ``path\to\Ryzen-AI-Library-Public-Release\windows``
5. Navigate to the samples directory and run depth-estimation application:
    ```
    cvml-samples-depth-estimation.exe -i path\to\input -o path\to\output
    ```

## Running Depth-Estimation without venv
> **Prerequisite 1**: install **Python 3.9 (3.9.13)** on your machine and ensure ``path\to\Python39`` and ``path\to\Python39\Scripts`` are at the top of system PATH. OR run the following on CMD Prompt if you don't want to add Python to path permanently

    set PATH=path\to\Python39;path\to\Python39\Scripts;%PATH% 
    
> **Prerequisite 2** Ensure ``C:Windows\System32\AMD`` is in PATH

1. Navigate to ``path\to\ryzen-ai-sw-1.0\ryzen-ai-sw-1.0\voe-4.0-win_amd64``

2. Install the following Python dependencies:
    ```
    pip install numpy
    pip install voe-0.1.0-cp39-cp39-win_amd64.whl
    ```
3. Move ``vaip_config.json`` from ``path\to\ryzen-ai-sw-1.0\voe-4.0-win_amd64\`` to ``path\to\Ryzen-AI-Library-Public-Release\windows``

4. Run RyzenAI sample

## Steps to re-compile the RyzenAI Depth-Estimation sample application
The RyzenAI library already includes the pre-built depth estimation application along with its dependencies.
The following steps show how to rebuild the application:

1. Navigate to the samples directory and create a build directory
    ```
    cd path-to-RyzenAI-Public-Library\samples\cvml-sample-depth-estimation
    mkdir build && cd build
    ```

2. Install OpenCV 4.7.0 prebuilt package, and set the ```OPENCV_INSTALL_ROOT``` environment variable.

3. Run ```cmake``` to configure the project and generate a native build system
    ```
    cmake -S .. -B . -DOPENCV_INSTALL_ROOT=%OPENCV_INSTALL_ROOT%
    ```

4. Build the project; the depth-estimation executable can be found under the ```build\Release``` directory.
    ```
    cmake --build . --config Release
    ```

6. Add ```path\to\RyzenAI-Public-Library\windows``` and ```%OPENCV_INSTALL_ROOT%\x64\v16\bin``` to your PATH
