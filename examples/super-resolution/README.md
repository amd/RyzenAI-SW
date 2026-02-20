<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Super-Resolution </h1>
    </td>
 </tr>
</table>

# Running Super-Resolution Models on Ryzen AI

This Ryzen AI example lets you upscale images using state-of-the-art super-resolution models running locally on your AMD NPU. Two model families are provided:

- **Real-ESRGAN** — A perceptually optimized model for **4x** upscaling, available in four tile sizes ([Wang et al., 2021](https://arxiv.org/abs/2107.10833))
- **SESR-M7** — A super-efficient model for **2x** upscaling ([Bhardwaj et al., 2022](https://arxiv.org/abs/2103.09404))

The models have been quantized to INT8 and optimized for the AMD Ryzen AI NPU using [ONNX Runtime](https://onnxruntime.ai/) with the [Vitis AI Execution Provider](https://onnxruntime.ai/docs/execution-providers/Vitis-AI-ExecutionProvider.html). 

Here, you can learn how you can get inference up and running for these models, hosted on Hugging Face.

## Prerequisites

### Hardware

You need an AMD AI PC with a Ryzen AI NPU. Supported processors:

| Series | Codename | Abbreviation | Launch Year | Windows 11 |
|--------|----------|--------------|-------------|------------|
| Ryzen AI Max PRO 300 Series | Strix Halo | STX | 2025 | ✔️ |
| Ryzen AI PRO 300 Series | Strix Point / Krackan Point | STX/KRK | 2025 | ✔️ |
| Ryzen AI Max 300 Series | Strix Halo | STX | 2025 | ✔️ |
| Ryzen AI 300 Series | Strix Point | STX | 2025 | ✔️ |
| Ryzen Pro 200 Series | Hawk Point | HPT | 2025 | ✔️ |
| Ryzen 200 Series | Hawk Point | HPT | 2025 | ✔️ |
| Ryzen PRO 8000 Series | Hawk Point | HPT | 2024 | ✔️ |
| Ryzen 8000 Series | Hawk Point | HPT | 2024 | ✔️ |
| Ryzen Pro 7000 Series | Phoenix | PHX | 2023 | ✔️ |
| Ryzen 7000 Series | Phoenix | PHX | 2023 | ✔️ |

### Software

1. Install the latest NPU drivers and Ryzen AI Software by following the [Ryzen AI SW Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html). Allow approximately **30 minutes** for the full installation.

2. Activate the Ryzen AI conda environment and set the installation path. Substitute the correct version number for `v.v.v` (e.g., `1.7.0`):

```powershell
conda activate ryzen-ai-v.v.v
$Env:RYZEN_AI_INSTALLATION_PATH = 'C:/Program Files/RyzenAI/v.v.v/'
```
---

## Real-ESRGAN — 4x Super-Resolution

**Real-ESRGAN** (Real Enhanced Super-Resolution Generative Adversarial Networks) produces high-quality **4x** upscaled images with strong perceptual fidelity. This version has been re-trained with reduced feature channels and fewer stacked blocks for improved efficiency, then quantized to INT8 for the AMD NPU. Here is an example input and output image produced with the 256x256 tile variant of the model.


| Input image | Output image | 
| --- | --- |
| ![assets/input_tiger_320x480_108005.png](assets/input_tiger_320x480_108005.png) | ![assets/output_tiger_4x_1280x1920_108005.png](assets/output_tiger_4x_1280x1920_108005.png) | 

*Figure 1: Input 320x480 scaled up by 4x to 1280x1920 with Real-ESRGAN model running on AMD AI PC NPU. Source: EDSR Benchmark dataset (edsr_benchmark\B100\HR\108005.png).*

> For more details on the tiling method, please visit the AMD blog: [Super Resolution Acceleration on Ryzen AI NPU](https://www.amd.com/en/developer/resources/technical-articles/2026/super-res-acceleration-on-ryzen-ai-npu.html).

Four tile-size variants are available. All share the same code and inference pipeline — only the ONNX model file differs:

| Model | Tile Size | Hugging Face Link |
|-------|-----------|-------------------|
| Real-ESRGAN | 128×128 | [amd/realesrgan-128x128-tiles-amdnpu](https://huggingface.co/amd/realesrgan-128x128-tiles-amdnpu) |
| Real-ESRGAN | 256×256 | [amd/realesrgan-256x256-tiles-amdnpu](https://huggingface.co/amd/realesrgan-256x256-tiles-amdnpu) |
| Real-ESRGAN | 512×512 | [amd/realesrgan-512x512-tiles-amdnpu](https://huggingface.co/amd/realesrgan-512x512-tiles-amdnpu) |
| Real-ESRGAN | 1024×1024 | [amd/realesrgan-1024x1024-tiles-amdnpu](https://huggingface.co/amd/realesrgan-1024x1024-tiles-amdnpu) |

**Tile size tradeoffs:** A larger tile size reduces stitching overhead and may produce fewer boundary artifacts, but requires more memory per tile. A smaller tile size is more memory-friendly and can be better suited for lower-resolution inputs. All variants produce identical 4x upscaled output quality on the same image content.

### Quick Start

The example below uses the 256×256 variant. To use a different tile size, substitute the repository name and ONNX filename accordingly.

1. Clone the model repository from Hugging Face:

```powershell
git clone https://hf.co/amd/realesrgan-256x256-tiles-amdnpu
cd realesrgan-256x256-tiles-amdnpu
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run inference on an image:

```powershell
python onnx_inference.py --onnx onnx-models/realesrgan_nchw_256x256_u8s8.onnx --input <path-to-image> --out-dir outputs --device npu
```

For other tile sizes, swap the ONNX model path:
- `onnx-models/realesrgan_nchw_128x128_u8s8.onnx`
- `onnx-models/realesrgan_nchw_512x512_u8s8.onnx`
- `onnx-models/realesrgan_nchw_1024x1024_u8s8.onnx`

### Arguments

| Argument | Description |
|----------|-------------|
| `--onnx` | Path to the ONNX model file |
| `--input` | A single image file or a directory (recursively processes `.png`, `.jpg`, `.jpeg`) |
| `--out-dir` | Output directory for the upscaled images |
| `--device` | `npu` (default) or `cpu` |

For evaluation scripts, benchmark datasets, and accuracy metrics, see the full model card on Hugging Face: [amd/realesrgan-256x256-tiles-amdnpu](https://huggingface.co/amd/realesrgan-256x256-tiles-amdnpu). 

---

## SESR-M7 — 2x Super-Resolution

**SESR** (Super-Efficient Super Resolution) is a lightweight CNN-based model designed for fast, efficient image upscaling. The SESR-M7 is one of the smaller variants of SESR and uses a tiling technique to work across the image in 256x256 pixel tiles to produce a **2x** super-resolution output. Even though the tile size is 256×256, the inference pipeline handles images of almost any size. For example, here is an input of 339x510 and the 2x output of 678x1020. 

> For more details on the tiling method, please visit the AMD blog:  [Super Resolution Acceleration on Ryzen AI NPU](https://www.amd.com/en/developer/resources/technical-articles/2026/super-res-acceleration-on-ryzen-ai-npu.html).

| Input image | Output image | 
| --- | --- |
| ![assets/input_ice_climber_0844.png](assets/input_ice_climber_0844.png) | ![assets/output_ice_climber_0844_x2.png](assets/output_ice_climber_0844_x2.png) | 

*Figure 2: Ice climber image upscaled by 2x with SESR-M7 model running on AMD AI PC NPU. Source: DIV2K dataset (DIV2K_valid_LR_bicubic\X4\0844x4.png).*

📄 **Full model card:** [amd/sesr-m7-256x256-tiles-amdnpu](https://huggingface.co/amd/sesr-m7-256x256-tiles-amdnpu)

### Quick Start

1. Clone the model repository from Hugging Face:

```powershell
git clone https://hf.co/amd/sesr-m7-256x256-tiles-amdnpu
cd sesr-m7-256x256-tiles-amdnpu
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run inference on an image:

```powershell
python onnx_inference.py --onnx onnx-models/sesr_nchw_int8.onnx --input <path-to-image> --out-dir outputs --device npu
```

Replace `<path-to-image>` with the path to a `.png` or `.jpg` file, or a directory of images.

### Arguments

| Argument | Description |
|----------|-------------|
| `--onnx` | Path to the ONNX model file |
| `--input` | A single image file or a directory (recursively processes `.png`, `.jpg`, `.jpeg`) |
| `--out-dir` | Output directory for the upscaled images |
| `--device` | `npu` (default) or `cpu` |

For evaluation scripts, benchmark results, and training details, see the [full model card on Hugging Face](https://huggingface.co/amd/sesr-m7-256x256-tiles-amdnpu).


---

## Notes

- **First-run compilation:** If the model has not been previously compiled for your NPU, the first inference run will take several minutes to compile and cache the model. Subsequent runs will be fast. The compiled model cache is stored in the `modelcachekey_*` directory within the repository.
- **CPU fallback:** All models can also run on CPU by passing `--device cpu`. No NPU drivers or Ryzen AI SW are required for CPU execution, but performance will be significantly slower.

## Additional Resources

Join the [AMD AI Developer Program](https://www.amd.com/en/developer/ai-dev-program.html) to get access to tools, resources, and a community of developers building on AMD hardware.

- **Ryzen AI Documentation:** [ryzenai.docs.amd.com](https://ryzenai.docs.amd.com/en/latest/)
- **Technical Deep-Dive — Tiling Strategy on NPU:** [Super Resolution Acceleration on Ryzen AI NPU](https://www.amd.com/en/developer/resources/technical-articles/2026/super-res-acceleration-on-ryzen-ai-npu.html)
- **AMD Developer Community Discord:** [discord.gg/amd-dev](https://discord.gg/amd-dev)
- **Hugging Face Model Cards:**
  - [amd/sesr-m7-256x256-tiles-amdnpu](https://huggingface.co/amd/sesr-m7-256x256-tiles-amdnpu)
  - [amd/realesrgan-128x128-tiles-amdnpu](https://huggingface.co/amd/realesrgan-128x128-tiles-amdnpu)
  - [amd/realesrgan-256x256-tiles-amdnpu](https://huggingface.co/amd/realesrgan-256x256-tiles-amdnpu)
  - [amd/realesrgan-512x512-tiles-amdnpu](https://huggingface.co/amd/realesrgan-512x512-tiles-amdnpu)
  - [amd/realesrgan-1024x1024-tiles-amdnpu](https://huggingface.co/amd/realesrgan-1024x1024-tiles-amdnpu)
