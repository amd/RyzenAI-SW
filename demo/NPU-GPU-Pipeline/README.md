<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI NPU-GPU Pipeline </h1>
    </td>
 </tr>
</table>

# Model Pipelining on NPU and iGPU

🚨 **Notice on NPU-GPU Pipeline Demo**

The NPU-GPU pipeline examples are currently being migrated to the RyzenAI 1.3 release. Until the migration is complete, please use the RyzenAI 1.2 branch for this demo.

We showcase an application that has models strategically distributed and off-loaded to the NPU and integrated GPU (iGPU), based on varying intensity of computation requirements and hardware support. The application consists of two Convolutional Neural Network (CNN) based models that run on the NPU, and one generative model that is off-loaded to the iGPU. ONNX Runtime provides support for Vitis AI EP and DirectML EP, used to demonstrate inference on the NPU and iGPU respectively. Some of these models operate concurrently, thus utilizing the accelerator resources to their full potential. 

## Prerequisites and Environment Setup

Assumes Anaconda prompt for these instructions.

1. Install Ryzen AI Software using the automatic installer [Link](https://ryzenai.docs.amd.com/en/latest/inst.html). This should create a conda environment that can be used for this example. Check the installation path variables

```bash
# Default location of RyzenAI software installation
echo %RYZEN_AI_INSTALLATION_PATH%
```

2. Create a clone of the Ryzen AI installation conda environment to add required python packages

```python
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-1.3.0
conda create --name npu-gpu-pipeline --clone %RYZEN_AI_CONDA_ENV_NAME%
conda activate npu-gpu-pipeline
```

3. Install dependencies for Yolov8 and RCAN:

```bash
python -m pip install -r requirements.txt 
```

4. Download pre-quantized ONNX models from Huggingface for [Yolov8](https://huggingface.co/amd/yolov8m/tree/main) and [RCAN](https://huggingface.co/amd/rcan/tree/main) in the same directory. 

5. Install dependencies for Stable Diffusion:

```bash
python -m pip install -r stable_diffusion\requirements-common.txt 
```
6. Make sure XLNX_VART_FIRMWARE is set to point to the correct xclbin from the VOE package
```bash
echo %XLNX_VART_FIRMWARE%
```
7. Copy vaip_config.json from the installed VOE package to this directory
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
