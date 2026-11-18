# Model-specific quantization improvements for the Ryzen AI NPU

Quantization recipes for three CNN / transformer models, tuned for accuracy on the Ryzen AI NPU
(VitisAI EP). Each model folder contains everything needed to rebuild and evaluate the model:

- the FP32 ONNX export script,
- the Olive + AMD Quark quantization config and any custom Olive passes,
- a script that generates the calibration images from public datasets, with their source, pre-processing and order,
- the accuracy test, and the CPU EP / NPU results.

Each model folder also contains the pre-built quantized model (`output/ryzen_model/model.onnx`) that
the results below were measured on, so the models can be evaluated without re-quantizing.

| Model | Task | Recipe | Accuracy test | FP32, CPU EP | INT8, CPU EP | INT8, NPU |
|---|---|---|---|---|---|---|
| [ConvNeXt-Tiny](ConvNext/README.md) | Classification | Int8 Entropy, CLE | ImageNet-1K val, 5,000 images, Top-1 % | 78.00 | 77.18 | **76.20** |
| [DETR-ResNet50](DeTR/README.md) | Object detection | XINT8 trunk (MinMSE) + UInt16 tail (Percentile) | COCO 2017 val, 4,952 images, mAP | 0.380 (FP16) | 0.322 | **0.288** |
| [Real-ESRGAN x4](ESRGAN/README.md) | Super-resolution 250→1000 | UInt8/Int8 MinMSE, power-of-2, CLE + AdaRound | DIV2K val, 99 images, PSNR-Y dB | 25.73 | 25.87 | **25.86** |

For DETR, the reference is FP16 (an FP32 run on the same harness is not available). NPU latency is
1.88 ms (ConvNeXt), 17.4 ms (DETR) and 98.0 ms (ESRGAN) per image. Each model README has the full
results, including Top-5, mAP@.5 / @.75 and SSIM.

## Layout

```
Model_specific_Improvements/
├── README.md                     this file
├── requirements_quant.txt        FP32 export + Olive/Quark quantization environment
├── requirements_accuracy.txt     accuracy test environment (CPU EP + VitisAI EP through Windows ML)
├── ConvNext/  DeTR/  ESRGAN/     one folder per model, same structure:
│   ├── README.md                 recipe, calibration, accuracy method, results
│   ├── export_fp32.py            writes <model>_fp32.onnx next to the script
│   ├── olive_ryzen_config.json   Olive workflow (QuarkQuantization + custom passes)
│   ├── olive_ryzen_data_scripts.py   calibration data loader and pre-processing
│   ├── olive_ryzen/              custom Olive passes + pass registry (ConvNext, DeTR only)
│   ├── run_quant.bat / .sh       runs the Olive workflow from the model folder
│   ├── output/ryzen_model/       pre-built quantized model (ESRGAN: also its run summary)
│   ├── calibration_dataset/
│   │   ├── prepare_calibration_dataset.py   generates images/ from a public dataset
│   │   └── images/               calibration images (generated, not in the repository)
│   └── accuracy/
│       ├── <model>_accuracy_test.py
│       ├── ort_test_utils.py     session creation (CPU EP / VitisAI EP)
│       └── winml_helper.py       registers the VitisAI EP through Windows ML
```

Generated files (`*_fp32.onnx`, `calibration_dataset/images/`, `cache/`, `output/`, `output_prev/`,
`tmp_olive_custom/`, `torch_model/`, `quantized_info.csv`) are listed in `.gitignore`. The exception is
`<model>/output/ryzen_model/`, which holds the pre-built model and is committed.

## Workflow

For each model, from its folder:

1. Optional: `python calibration_dataset/prepare_calibration_dataset.py` generates the calibration images in `calibration_dataset/images/` (quantization env). For ESRGAN, add `--div2k <DIV2K_train_HR folder or zip>` or `--download`. If this step is skipped, `run_quant` generates a missing or incomplete dataset automatically (ESRGAN uses `--download`).
2. `python export_fp32.py`: export the FP32 ONNX model (quantization env). The README lists the expected SHA-256.
3. `run_quant.bat` (Windows) or `bash run_quant.sh` (Linux): quantize (quantization env). The model is written to `output/ryzen_model/model.onnx`. This step is optional: the pre-built model is already there, and `run_quant` only rebuilds it with `--force` (the old `output/` is moved to `output_prev/`).
4. `cd accuracy` and run the accuracy test on the CPU EP and the NPU (accuracy env).

Steps 1 to 3 are needed only to re-quantize; the accuracy test runs on the pre-built model directly.

The scripts change to their own folder, so they can be started from anywhere. The config paths
(`calibration_dataset/...`, `cache`, `output`) are relative to the model folder.

## Environments

Two separate Python environments are used, because the quantization stack (Olive 0.11 / Quark 0.10,
`onnxruntime` 1.22, `numpy` 1.26) and the Windows ML runtime (`onnxruntime-windowsml` 1.25, `numpy` 2.x)
can't share one environment.

**Quantization (Windows or Linux, Python 3.11 or 3.12):**

```bat
python -m venv quant_env
quant_env\Scripts\activate                 :: Linux: source quant_env/bin/activate
pip install -r requirements_quant.txt
```

The versions are pinned to the environment the recipes were built with. Other versions of
torch / transformers / onnx can change the exported graph, and other Olive / Quark versions can change
the quantization result.

The quantization runs need free space in the system temp folder (`%TEMP%` / `/tmp`). Quark's MinMSE
calibration caches activations there: about 260 GB for DETR and at least 190 GB free (200 GB recommended)
for ESRGAN. ConvNeXt (Entropy) needs little. Point `TEMP`/`TMP` (Windows) or `TMPDIR` (Linux) to a large
disk if needed.

**Accuracy (Windows 11, Python 3.12):**

```bat
conda create -y -n mdl-accu python=3.12
conda activate mdl-accu
pip install -r requirements_accuracy.txt
```

The accuracy tests create sessions through `ort.get_ep_devices()`. The VitisAI EP is registered through
Windows ML (`accuracy/winml_helper.py`), so the NPU driver and a Windows ML VitisAI EP package must be
installed on the machine. For VitisAI EP sessions, `ort_test_utils.create_session` sets a `cache_key`
that includes the model's MD5, so two quantizations of the same graph never share a compiled NPU binary.
The first NPU session of a model compiles it (up to about 5 minutes for ESRGAN); later sessions use
the cache.

The datasets for the accuracy tests are downloaded on first use: ImageNet-1K val and COCO 2017 val are
streamed from Hugging Face; DIV2K val is downloaded from the ETH Zürich server.

## Calibration data

| Model | Images | Source | Order |
|---|---|---|---|
| ConvNeXt | 200 | ImageNet-derived images (a subset of the Hugging Face `timm/mini-imagenet` train split) | Pinned by `calib_list.txt` |
| DETR | 200 | COCO 2017 val (HF `detection-datasets/coco`, pinned revision `cf0b2233…`, seed 42) | `os.listdir()` |
| ESRGAN | 10 | DIV2K train HR (seed 42), center crop 1000 → bicubic 250 | Pinned by `manifest.txt` |

The calibration images are not included in the repository. `calibration_dataset/prepare_calibration_dataset.py`
in each model folder generates them from the public dataset, with the file names and order the
recipes use. `run_quant.bat` and `run_quant.sh` check the expected image count and automatically run
the preparation script when the dataset is missing or incomplete; ESRGAN uses the script's
`--download` mode. Entropy calibration (ConvNeXt) depends on the image order: in two tests, a
different order of the same 200 images cost 5 to 6 points of Top-1. Each model README describes the
source of the images and the pre-processing used at calibration time.

## Test setup

| Item | Value |
|---|---|
| NPU | Ryzen AI NPU |
| OS | Windows 11 Pro 24H2 |
| VitisAI EP | 1.8.61.0 through Windows ML |
| Accuracy runtime | `onnxruntime-windowsml` 1.25.2, Python 3.12 |
| Quantization | Olive 0.11.0 + AMD Quark 0.10, Python 3.12 |

The NPU was idle (no other inference process) during the NPU accuracy runs.

## License and third-party notices

The code is released under the MIT License, see [LICENSE.txt](../../LICENSE.txt). Third-party code
and the models and datasets used by the recipes are listed below.

**Models**

| Model | Source | License |
|---|---|---|
| ConvNeXt-Tiny | torchvision `convnext_tiny`, weights `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` (downloaded by `export_fp32.py`) | torchvision code: BSD 3-Clause. The weights were trained on ImageNet-1K. |
| DETR-ResNet50 | Hugging Face `facebook/detr-resnet-50`, revision `no_timm` (downloaded by `export_fp32.py`) | Apache 2.0 |
| Real-ESRGAN x4plus | `RealESRGAN_x4plus.pth` from the Real-ESRGAN v0.1.0 release (downloaded by `export_fp32.py`) | BSD 3-Clause |
| ConvNeXt INT8 model (included) | `ConvNext/output/ryzen_model/model.onnx`, a quantized derivative of the torchvision `IMAGENET1K_V1` weights | torchvision code: BSD 3-Clause. The weights were trained on ImageNet-1K. |
| DETR INT8 model (included) | `DeTR/output/ryzen_model/model.onnx`, a quantized derivative of `facebook/detr-resnet-50` | Apache 2.0 |
| ESRGAN INT8 model (included) | `ESRGAN/output/ryzen_model/model.onnx`, a quantized derivative of `RealESRGAN_x4plus.pth` | BSD 3-Clause |

**Datasets**

| Dataset | Use | Terms |
|---|---|---|
| ImageNet-1K | ConvNeXt calibration images (200 images from `timm/mini-imagenet`, downloaded by the script); ConvNeXt accuracy test (`evanarlian/imagenet_1k_resized_256`, downloaded) | ImageNet terms of access: non-commercial research and educational use only. ImageNet does not own the images. Users must obtain access and accept the ImageNet terms themselves; the terms also apply to the Hugging Face copies used here. |
| COCO 2017 | DETR calibration images (200 val images, downloaded by the script); DETR accuracy test (`detection-datasets/coco`, downloaded) | Annotations: CC BY 4.0. The images are Flickr photos that remain under the licenses of their owners (Flickr Terms of Use). |
| DIV2K | ESRGAN calibration images (10 images derived from train HR by the script); ESRGAN accuracy test (valid HR, downloaded) | Academic research use only. |

Citations:

- ImageNet: J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li and L. Fei-Fei, "ImageNet: A Large-Scale Hierarchical Image Database", CVPR 2009.
- COCO: T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context", ECCV 2014.
- DIV2K: E. Agustsson and R. Timofte, "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study", CVPR Workshops 2017.

**Third-party code**

| File | Origin | License |
|---|---|---|
| `*/accuracy/winml_helper.py` | Adapted from the Windows ML Python sample in the Microsoft Learn documentation ([MicrosoftDocs/windows-ai-docs](https://github.com/MicrosoftDocs/windows-ai-docs)) | MIT, Copyright (c) Microsoft Corporation |
| `ConvNext/olive_ryzen/olive_ryzen_config.json`, `DeTR/olive_ryzen/olive_ryzen_config.json` | Copy of the pass registry `olive/olive_config.json` from [Microsoft Olive](https://github.com/microsoft/Olive), with the AMD custom passes appended | MIT, Copyright (c) Microsoft Corporation |

The Python packages in `requirements_quant.txt` and `requirements_accuracy.txt` are installed from PyPI
and GitHub under their own licenses.
