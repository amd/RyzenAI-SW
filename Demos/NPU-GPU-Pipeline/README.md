<table class="sphinxhide" width="100%">
 <tr width="100%">
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI NPU-GPU Pipeline </h1>
   </td>
 </tr>
</table>

# Model Pipelining on NPU and iGPU

We showcase an application that has models strategically distributed and off-loaded to the NPU and integrated GPU (iGPU), based on varying intensity of computation requirements and hardware support. The application consists of two Convolutional Neural Network (CNN) based models that run on the NPU, and one generative model that is off-loaded to the iGPU. ONNX Runtime provides support for Vitis AI EP and DirectML EP, used to demonstrate inference on the NPU and iGPU respectively. Some of these models operate concurrently, thus utilizing the accelerator resources to their full potential. Please refer to [NPU-GPU article](https://www.amd.com/en/developer/resources/technical-articles/model-pipelining-on-npu-and-gpu-using-ryzen-ai-software.html) for high level implementation

## Prerequisites and Environment Setup

Install Ryzen AI Software using the automatic installer [Link](https://ryzenai.docs.amd.com/en/latest/inst.html). This should create a Python virtual environment that can be used for this example. Check the installation path variables

Below instructions can be executed on command prompt or Miniforge Prompt

Set Up Python Virtual Environment for NPU-GPU Pipeline

Step 1: Activate the Ryzen AI installation environment
```bat
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Step 2: Export the installed Python packages
```bat
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```
This creates a requirements file containing the packages installed in the Ryzen AI environment.

Step 3: Create a virtual environment for NPU-GPU Pipeline
```bat
cd /d "<RyzenAI-SW>\demo\NPU-GPU-Pipeline"
python -m venv npu-gpu-pipeline
```

Step 4: Deactivate the Ryzen AI installation environment
```bat
deactivate
```

Step 5: Activate the NPU-GPU Pipeline environment
```bat
"npu-gpu-pipeline\Scripts\activate.bat"
```

Step 6: Install the Ryzen AI Python dependencies
```bat
python -m pip install -r "%TEMP%\ryzenai_1.9_requirements.txt"
```

Step 7: Install the example-specific dependencies for YOLOv8 and RCAN
```bat
cd <RyzenAI-SW>\demo\NPU-GPU-Pipeline
python -m pip install -r requirements.txt
```

The npu-gpu-pipeline virtual environment is now configured for running the NPU-GPU Pipeline demo with YOLOv8 and RCAN.

step 8: Download pre-quantized ONNX models from Huggingface for [Yolov8](https://huggingface.co/amd/yolov8m/tree/main) and [RCAN](https://huggingface.co/amd/rcan/tree/main) in the same directory.

step 9: Install dependencies for Stable Diffusion:

```bash
python -m pip install -r stable_diffusion\requirements-common.txt
```

step 10: Make sure XLNX_VART_FIRMWARE is set to point to the correct xclbin from the VOE package
```bash
echo %XLNX_VART_FIRMWARE%
```

step 11: Copy vaip_config.json from the installed VOE package to this directory
```bash
copy %RYZEN_AI_INSTALLATION_PATH%\voe-4.0-win_amd64\vaip_config.json .
```

## Running the application

1. Generate optimized stable diffusion model using Olive:

```bash
cd stable_diffusion
python stable_diffusion.py --provider dml --optimize
```
The optimized FP16 models should be generated in models\optimized-dml\.

2. Run the example (the following command off-loads Stable Diffusion to the iGPU and Yolov8+RCAN to the NPU):

```bash
cd ..
python pipeline.py -i test/test_img2img.mp4 --npu --provider_config vaip_config.json --igpu
```
