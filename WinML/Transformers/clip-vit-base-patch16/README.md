# WinML CLIP-ViT 
Run quantized CLIP Vision Transformer models on Windows NPU for zero-shot image classification using natural language prompts.


## Installation Instructions

Install the required python packages in the conda environment `winml_clip` and Windows Apps SDK using the [Windows ML installation instructions](../README.md#winml-installation-instructions) in the main README.:

```sh
cd <RyzenAI-SW>\WinML\Transformers\clip-vit-base-patch16
conda create -n winml_clip --clone winml_env
conda activate winml_clip
pip install --upgrade -r .\requirements.txt
```

## Model Download and Export

Download the CLIP model from HuggingFace and export to ONNX format:

```bash
python download_clip.py
```

This will:
- Download `openai/clip-vit-base-patch16` from HuggingFace
- Export to ONNX format with proper input/output names

## Model Conversion for NPU

Get the optimized CLIP model for NPU using AI Toolkit:

Model conversion steps:
1. Open the CLIP model in VS Code with AI Toolkit extension installed
2. Right-click the model file (`./model/model.onnx`) and select "Convert Model"
3. Choose the target platform (e.g., AMD NPU)
4. Select quantization settings (e.g., A16W8 for activation 16-bit, weight 8-bit)
5. The toolkit will generate an optimized model (typically saved to `./model_a16w8/model.onnx`)

## Run Inference

Run inference with the base ONNX model:

```bash
python run_clip.py --model model\model.onnx
```

Run inference with NPU-optimized quantized model:

```bash
python run_clip.py --model model_a16w8\model.onnx --ep_policy NPU
```

Classify a custom image from URL:

```bash
python run_clip.py --image_url "https://example.com/your-image.jpg"
```

### Command-Line Arguments

- `--ep_policy`: Execution provider policy (default: NPU)
  - `NPU`: Prefer NPU execution using VitisAIExecutionProvider
  - `CPU`: Prefer CPU execution using CPUExecutionProvider
  - `DEFAULT`: Use default provider selection
- `--model`: Path to ONNX model (default: `./model/model.onnx`)
- `--image_url`: URL of image to classify (default: COCO cat image)

### Expected Output

Expected output for NPU with cat image:

```bash
Registering execution providers ...
paths: {'VitisAIExecutionProvider': 'C:\\Program Files\\WindowsApps\\MicrosoftCorporationII.WinML.AMD.NPU.EP.1.8_1.8.50.0_x64__8wekyb3d8bbwe\\ExecutionProvider\\onnxruntime_providers_vitisai.dll'}
Registered execution provider: VitisAIExecutionProvider with library path: C:\Program Files\WindowsApps\MicrosoftCorporationII.WinML.AMD.NPU.EP.1.8_1.8.50.0_x64__8wekyb3d8bbwe\ExecutionProvider\onnxruntime_providers_vitisai.dll
Model path: C:\Users\afotovva\Documents\repos\RyzenAI-SW\WinML\Transformers\clip-vit-base-patch16\model_a16w8\model.onnx
Image URL: http://images.cocodataset.org/val2017/000000039769.jpg
Loading image ...
Preparing inputs ...
Creating session ...
Set provider selection policy to: NPU
2026-02-27 12:03:59.3205637 [W:onnxruntime:, cache_dir.cpp:70 cache_dir.cpp] skip update cache dir: in-mem mode
Active execution providers (priority order): ['VitisAIExecutionProvider', 'CPUExecutionProvider']
Primary provider (highest priority): VitisAIExecutionProvider
Running inference ...

============================================================
CLASSIFICATION RESULTS
============================================================
a photo of a cat               : 98.58%
a photo of a dog               :  0.00%
a photo of a bird              :  0.00%
a photo of a person            :  0.97%
a photo of a car               :  0.06%
a photo of a bicycle           :  0.01%
a photo of a chair             :  0.20%
a photo of a bottle            :  0.15%
a photo of an airplane         :  0.01%
a photo of a boat              :  0.00%
============================================================
Most likely: a photo of a cat (98.58%)
============================================================

Validating against PyTorch reference ...
Loading weights: 100%|██████| 398/398 [00:00<00:00, 6804.99it/s, Materializing param=visual_projection.weight]
CLIPModel LOAD REPORT from: openai/clip-vit-base-patch16
Key                                  | Status     |  |
-------------------------------------+------------+--+-
text_model.embeddings.position_ids   | UNEXPECTED |  |
vision_model.embeddings.position_ids | UNEXPECTED |  |

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Probability distribution similarity: 0.999977
Maximum element-wise difference: 0.008242
Validation: PASSED (similarity > 0.95)
```



