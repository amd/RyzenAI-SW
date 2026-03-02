# Stable Diffusion Demo

> import CIStatus from '@site/src/components/CIStatus';

import CIStatus from '@site/src/components/CIStatus';

# Stable Diffusion Demo

<CIStatus validated={false} />

Ryzen AI 1.7 provides preview demos of Stable Diffusion image-generation pipelines. The demos cover Image-to-Image and Text-to-Image using SD 1.5, SD 2.1-base, SD 2.1, SDXL-base-1.0, Segmind-Vega, SD-Turbo, SDXL-Turbo, SD 3.0, SD3.5 and SD3.5-Turbo.

The models for SD 1.5, SD 2.1-base, SD 2.1, SDXL-base-1.0, Segmind-Vega, SD-Turbo, SDXL-Turbo are available for public download. The SD3.0 / SD3.5 / SD3.5-Turbo models are only available to confirmed Stability AI licensees.

## Installation Steps

1. Ensure the latest version of Ryzen AI and NPU drivers are installed. See [installation instructions](/getting-started/installation).

2. The GenAI-SD folder is located in the RyzenAI installation tree. Navigate to the folder and run the following command:

```bat
cd "C:\Program Files\RyzenAI\1.7.0\GenAI-SD"
```

3. Activate the Conda environment for the Stable Diffusion demo packages:

```bash
conda activate ryzen-ai-1.7.0
```

4. Download the Stable Diffusion models:

   - [GenAI-SD-models-v0109.zip](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=GenAI-SD-models_v0109.zip)
   - [GenAI-SDXL-models-v0109.zip](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=GenAI-SDXL-models_v0109.zip)
   - [GenAI-Segmind-Vega-models-v0109.zip](https://account.amd.com/en/forms/downloads/ryzenai-eula-public-xef.html?filename=GenAI-Segmind-Vega-models_v0109.zip)

5. Extract the downloaded zip files and copy the models in the `GenAI-SD\models` folder. After installing all the models, the `GenAI-SD\models` folder should contain the following subfolders:

   - sd15
   - sd15_controlnet
   - sd21_base
   - sd-2.1-v
   - sd_turbo
   - sd_turbo_bs1
   - sdxl_turbo
   - sdxl_turbo_bs1
   - sdxl-base-1.0
   - segmind-vega

## Running the Demos

Activate the conda environment:

```bash
conda activate ryzen-ai-1.7.0
```

Optionally, set the NPU to high performance mode to maximize performance:

```bat
xrt-smi configure --pmode performance
```

Refer to the documentation on [xrt-smi configure](/tools/npu-management#xrt-smi-configure) for additional information.

### Image-to-Image with ControlNet

The image-to-image demo generates images based on a prompt and a control image for a Canny ControlNet. This demo supports SD 1.5 (512x512).

To run the demo, navigate to the `GenAI-SD\test` directory and run the following command:

```bat
python .\run_sd15_controlnet.py --model_id "stable-diffusion-v1-5" --model_path ..\models\sd15_controlnet_bfp\ --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
```

The demo script uses a predefined prompt and `ref\control.png` as the control image. The output image and control image are saved in the `generated_images` folder.

The control image can be modified and custom prompts can be provided with the `--prompt` option. For instance:

```bat
python run_sd15_controlnet.py --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" --model_path ..\models\sd15_controlnet_bfp\ --prompt "A red bird on a grey sky" --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
```

### Text-to-Image

The text-to-image generates images based on text prompts. This demo supports SD 1.5 (512x512), SD 2.1-base (512x512), SD 2.1 (768x768), SDXL-base (1024x1024), SD-Turbo (512x512), SDXL-Turbo (512x512), Segmind-Vega (1024x1024).

To run the demo, navigate to the `GenAI-SD\test` directory and run the following commands to run with each of the supported models:

```bat
python run_sd.py    --model_id "stable-diffusion-v1-5/stable-diffusion-v1-5" --model_path ..\models\sd15_bfp\ --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd.py    --model_id "stabilityai/stable-diffusion-2-1-base" --model_path ..\models\sd21_base_bfp --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd.py    --model_id "stabilityai/stable-diffusion-2-1" --model_path ..\models\sd-2.1-v\ --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd.py    --model_id "stabilityai/sd-turbo" --model_path ..\models\sd_turbo_bfp --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd.py    --model_id "stabilityai/sd-turbo" --model_path ..\models\sd_turbo_bs1_bfp --num_images_per_prompt 1 --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd_xl.py --model_id "stabilityai/sdxl-turbo" --model_path ..\models\sdxl_turbo_bfp --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd_xl.py --model_id "stabilityai/sdxl-turbo" --model_path ..\models\sdxl_turbo_bs1_bfp --num_images_per_prompt 1 --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd_xl.py --model_id "stabilityai/stable-diffusion-xl-base-1.0"  --model_path ..\models\sdxl-base-1.0_bfp\ --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
python run_sd_xl.py --model_id "segmind/Segmind-Vega" --model_path ..\models\segmind-vega_bfp\ --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
```

To run the sd3/sd3.5/sd3.5-Turbo models, you need to set env:DD_PLUGINS_ROOT before running the demo. For instance:

```bat
set DD_PLUGINS_ROOT=C:\Program Files\RyzenAI\1.7.0\GenAI-SD\lib\transaction\stx\
```

The demo script uses a predefined prompt for each of the models. The output images are saved in the `generated_images` folder.

Custom prompts can be provided with the `--prompt` option. For instance:

```bat
python run_sd.py --model_id "stabilityai/stable-diffusion-2-1-base" --model_path ..\models\sd21_base_bfp  --prompt "A bouquet of roses, impressionist style" --custom_op_path "C:\Program Files\RyzenAI\1.7.0\deployment\onnx_custom_ops.dll"
```

---

Ryzen AI is licensed under [MIT License](https://github.com/amd/ryzen-ai-documentation/blob/main/License). Refer to the [LICENSE File](https://github.com/amd/ryzen-ai-documentation/blob/main/License) for the full license text and copyright notice.
