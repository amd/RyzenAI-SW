# Model Pipelining on NPU and iGPU 

We showcase an application that has models strategically distributed and off-loaded to the NPU and integrated GPU (iGPU), based on varying intensity of computation requirements and hardware support. The application consists of two Convolutional Neural Network (CNN) based models that run on the NPU, and one generative model that is off-loaded to the iGPU. ONNX Runtime provides support for Vitis AI EP and DirectML EP, used to demonstrate inference on the NPU and iGPU respectively. Some of these models operate concurrently, thus utilizing the accelerator resources to their full potential. 

## Prerequisites and Environment Setup

Assumes Anaconda prompt for these instructions.

1. Install Ryzen AI Software using the automatic installer. This should create a conda environment that can be used for this example.
   Check the installation path variables
```bash
# Default location of RyzenAI software installation
echo %RYZEN_AI_INSTALLATION_PATH%
```


2. Install dependencies for Yolov8 and RCAN:

```bash
python -m pip install -r requirements.txt 
```

3. Download pre-quantized ONNX models from Huggingface for [Yolov8](https://huggingface.co/amd/yolov8m/tree/main) and [RCAN](https://huggingface.co/amd/rcan/tree/main) in the same directory. 

4. Install dependencies for Stable Diffusion:

```bash
python -m pip install -r stable_diffusion\requirements-common.txt 
```
5. Make sure XLNX_VART_FIRMWARE is set to point to the correct xclbin from the VOE package
```bash
echo %XLNX_VART_FIRMWARE%
```
6. Copy vaip_config.json from the installed VOE package to this directory
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
