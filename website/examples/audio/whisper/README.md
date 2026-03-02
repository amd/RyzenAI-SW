# Automatic Speech Recognition with OpenAI Whisper

> Run fast, on-device speech recognition with Ryzen AI and OpenAI Whisper. This guide covers running pre-built NPU-optimized models (quick start) and exporting your own models from Hugging Face (advanced).

# Automatic Speech Recognition with OpenAI Whisper

Run fast, on-device speech recognition with Ryzen AI and OpenAI Whisper. This guide covers running pre-built NPU-optimized models (quick start) and exporting your own models from Hugging Face (advanced).

## Supported Models

| Model | Parameters | NPU Support | Auto-Download |
|-------|-----------|-------------|---------------|
| whisper-base | 74M | Yes | [amd/whisper-base-onnx-npu](https://huggingface.co/amd/whisper-base-onnx-npu) |
| whisper-small | 244M | Yes | [amd/whisper-small-onnx-npu](https://huggingface.co/amd/whisper-small-onnx-npu) |
| whisper-medium | 769M | Yes | [amd/whisper-medium-onnx-npu](https://huggingface.co/amd/whisper-medium-onnx-npu) |
| whisper-large-v3-turbo | 809M | Yes | [amd/whisper-large-v3-turbo-onnx-npu](https://huggingface.co/amd/whisper-large-v3-turbo-onnx-npu) |

## Prerequisites

1. **Install Ryzen AI Software** — follow the [Installation guide](/getting-started/installation).

2. **Activate environment**

```bash
conda activate ryzen-ai-1.4.0
```

3. **Install dependencies**

```bash
cd docs/models-tutorials/audio/whisper
pip install -r requirements.txt
```

## Quick Start: Transcribe an Audio File

Models are auto-downloaded from AMD's Hugging Face repos on first run.

```bash
python run_whisper.py \
  --model-type whisper-base \
  --device npu \
  --input audio_files/1089-134686-0000.wav
```

Replace `whisper-base` with any supported model (`whisper-small`, `whisper-medium`, `whisper-large-v3-turbo`).

## Live Microphone Transcription

```bash
python run_whisper.py \
  --model-type whisper-base \
  --device npu \
  --input mic \
  --duration 0
```

`--duration 0` records continuously until Ctrl+C or silence is detected.

## Dataset Evaluation (WER, CER, RTF)

Evaluate on LibriSpeech samples to measure Word Error Rate, Character Error Rate, Real-Time Factor, and Time to First Token:

```bash
python run_whisper.py \
  --model-type whisper-base \
  --device npu \
  --eval-dir eval_dataset/LibriSpeech-samples \
  --results-dir results
```

## NPU Configuration

### How NPU Acceleration Works

When running on the NPU, Whisper's encoder and decoder are accelerated through the Vitis AI Execution Provider. For whisper-base:

```text notest
# Encoder operations
[Vitis AI EP] No. of Operators : VAIML   225
[Vitis AI EP] No. of Subgraphs : VAIML     1

# Decoder operations
[Vitis AI EP] No. of Operators :   CPU    24  VAIML   341
[Vitis AI EP] No. of Subgraphs : VAIML     2
```

100% of encoder operators and 93.4% of decoder operators run on the NPU.

### Execution Provider Configuration

Edit `config/model_config.json` to configure execution providers per model. For NPU, set `cache_key`, `cache_dir`, and point to the appropriate VitisAI config:

```json
{
  "config_file": "config/vitisai_config_whisper_decoder.json",
  "cache_dir": "./cache",
  "cache_key": "whisper_medium_decoder"
}
```

### Whisper-Medium Special Configuration

Whisper-medium requires additional flags in `config/vitisai_config_whisper_encoder.json`:

```json
"vaiml_config": {
  "optimize_level": 3,
  "aiecompiler_args": "--system-stack-size=512"
}
```

- `optimize_level=3`: aggressive optimizations for larger models
- `--system-stack-size=512`: increases AI Engine stack size for whisper-medium's resource demands

## Advanced: Export Your Own Models

If you need to export a custom Whisper model (e.g., a fine-tuned variant) from Hugging Face to ONNX with static shapes for NPU:

### Step 1: Export to ONNX

```bash
optimum-cli export onnx \
  --model openai/whisper-base \
  --task automatic-speech-recognition \
  whisper-base-onnx/
```

### Step 2: Convert Dynamic to Static Shapes

The NPU requires static input shapes. Use the included conversion script:

```bash
python dynamic_to_static.py
```

This uses `params.json` to fix dynamic dimensions in the encoder and decoder ONNX models.

### Step 3: Run with Explicit Paths

```bash
python run_whisper.py \
  --encoder whisper-base-onnx/encoder_model.onnx \
  --decoder whisper-base-onnx/decoder_model.onnx \
  --device npu \
  --input audio_files/1089-134686-0000.wav
```

## Whisper.cpp (C++ Alternative)

Ryzen AI also provides NPU acceleration for [whisper.cpp](https://github.com/ggerganov/whisper.cpp) through an AMD-maintained fork. On Ryzen AI 300 Series, the encoder fully offloads to the NPU for significant speedup versus CPU-only runs. NPU acceleration is currently Windows-only with Linux support planned.

For setup steps and NPU-optimized model guidance, see the [AMD whisper.cpp fork](https://github.com/amd/whisper.cpp?tab=readme-ov-file#amd-ryzen-ai-support-for-npu).

## Notes

- First run on NPU takes ~15 minutes for model compilation. Subsequent runs use the cached compiled model.
- Supports both CPU and NPU devices via the `--device` flag.
- Use `--language` to force a specific language for transcription.
