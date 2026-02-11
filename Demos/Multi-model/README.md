<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Multi model demo </h1>
    </td>
 </tr>
</table>

# Multi Model Running on NPU

We showcase an application that loads 2 different models in a single thread.

```text
NOTE: Models has to be initiated in proper order. The largest model needs to be initiated first.
```

## Prerequisites and Environment Setup

Install Ryzen AI Software using the automatic installer [Link](https://ryzenai.docs.amd.com/en/latest/inst.html). This should create a  conda environment that can be used for this example. Check the installation path variables

Below instructions can be executed on Conda command prompt or Miniforge Prompt

1. Create a clone of the Ryzen AI installation conda environment to add required python packages

```python
set RYZEN_AI_CONDA_ENV_NAME=ryzen-ai-<version>
conda create --name npu-gpu-pipeline --clone %RYZEN_AI_CONDA_ENV_NAME%
conda activate npu-gpu-pipeline
```
2. Set RyzenAI Environment variable

```bash
# Location of RyzenAI software installation path or default at "C:\Program Files\RyzenAI\<version>"
set RYZEN_AI_INSTALLATION_PATH=<Path to RyzenAI Installation>
```

3. Additional configuration needed.

```text
   Running of multi models in a single app needs additional setup.
   Models has to be initiated in proper order. The largest model needs to be initiated first.
   Also maxSpillBufferSize (in bytes) in provider_options should be bigger than the largest model.
   Please check compile_multi_model.py for reference.
```

## Prepare models
Download ResNet50 and Mobilenet_v2 models

```bash
cd models
python download_2_models.py
```

## Running the application

1. Run the example:

```bash
cd ..
python compile_multi_model.py
```
