# Parakeet ASR on Ryzen AI NPU

Speech-to-text transcription using NVIDIA's **Parakeet TDT 0.6B** model, optimized for the **AMD Ryzen AI NPU**. Achieves **35-43x real-time** transcription by running the Conformer encoder on the NPU, LSTM decoder on the integrated Radeon GPU, and mel features on the CPU—all three processors working in parallel.

This demo includes an OpenAI Whisper-compatible REST API, a CLI benchmark tool, and a real-time microphone transcription demo.

## Features

* 🚀 High-performance NPU-accelerated speech recognition (35-43x real-time)
* 🎯 Distributed compute across NPU, iGPU, and CPU for optimal performance
* 🎧 Real-time microphone transcription
* 📊 Benchmark tool with detailed performance metrics
* 🌐 OpenAI Whisper-compatible REST API

## Prerequisites

Install Ryzen AI Software using the automatic installer. See [RyzenAI documentation](https://ryzenai.docs.amd.com/en/latest/inst.html) for installation instructions.

**System Requirements:**

- AMD Ryzen AI processor (Strix)
- Windows 11
- Miniforge with `ryzen-ai-<version>` conda environment
- onnxruntime-vitisai, flexml-lite (included in Ryzen AI SDK)
- sounddevice (for live microphone mode)

## Performance

| Configuration | Speed | Hardware |
|---|---|---|
| CPU INT8 | 17-18x real-time | Zen 5 CPU only |
| **NPU BF16 (default power)** | **35x real-time** | NPU + iGPU + CPU |
| **NPU BF16 (performance mode)** | **43x real-time** | NPU + iGPU + CPU |

Tested on 16.5 minutes of audio (RTF=0.023-0.030). See [OPTIMIZATION.md](OPTIMIZATION.md) for the full optimization journey.

To set NPU performance mode: `C:\Windows\System32\AMD\xrt-smi.exe configure --pmode performance`

## Quick Start

### Step 1: Install Dependencies

Clone the repository and create a new conda environment based on your Ryzen AI installation:

```bash
git clone https://github.com/amd/RyzenAI-SW.git
cd RyzenAI-SW/Demos/ASR/Parakeet-TDT

conda create --name asr_parakeet_env --clone ryzen-ai-<version>
conda activate asr_parakeet_env

pip install -r requirements.txt
```

### Step 2: Download Models

Download the FP32 Parakeet TDT models from HuggingFace (~2.4GB):

```bash
python download_models.py --precision fp32
```

Downloads FP32 models (~2.4GB) from HuggingFace. For INT8 (CPU-only, smaller):

```bash
python download_models.py
```

### 2. Prepare Models for NPU

```bash
conda activate ryzen-ai-1.7.1

# Static shapes + NPU compiler fixes (Pad->Conv fuse, attention mask patch)
python preprocess_for_npu.py --precision fp32
```

### 3. Run

**Benchmark (NPU + iGPU):**
```bash
conda activate ryzen-ai-1.7.1
python test_transcribe.py audio.wav --device npu --decoder-device gpu --runs 3
```

**Live microphone transcription:**
```bash
pip install sounddevice
python live_transcribe.py --device npu
```

**API server:**
```bash
pip install -r requirements.txt
python server.py --device npu
```

**CPU-only (no Ryzen AI needed):**
```bash
pip install onnxruntime
python test_transcribe.py audio.wav --device cpu
```

> **Note:** The first NPU run triggers VAIML compilation which is cached at `C:\temp\<user>\vaip\.cache\`. Subsequent runs load from cache in ~4-6 seconds. The cache is keyed by model signature, so it is shared across directories using the same model.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Audio (WAV)                                            │
│    ↓                                                    │
│  Mel Filterbank (CPU, vectorized numpy)     ~25ms/chunk │
│    ↓                                                    │
│  Conformer Encoder (NPU, BF16)             ~300ms/chunk │
│    ↓                                                    │
│  TDT LSTM Decoder (iGPU, DirectML)    ~1.0ms/step ×188 │
│    ↓                                                    │
│  Text output                                            │
└─────────────────────────────────────────────────────────┘

For multi-chunk audio, encoder and decoder run in parallel:
  NPU encodes chunk N+1 while iGPU decodes chunk N
```

## Files

```
server.py                    FastAPI server (Whisper-compatible API)
test_transcribe.py           Benchmark with per-stage timing breakdown
live_transcribe.py           Real-time microphone transcription
benchmark_npu.py             Multi-config VAIML parameter sweep

inference/
  __init__.py
  transcriber.py             ONNX Runtime pipeline (NPU encoder + iGPU decoder)
  mel.py                     Vectorized 128-bin mel filterbank
  audio.py                   WAV parsing

preprocess_for_npu.py        Static shapes + NPU Pad/mask fixes (FP32 encoder -> .static.npu.onnx)
fuse_pads_direct.py          Optional: fuse Pad->Conv on a legacy .static.onnx only
optimize_model.py            Experimental ORT fold + fusion (needs unfused static.onnx)
fuse_attn_pads.py            Analyze attention Pad ops

models/
  vai_ep_config.json         VitisAI EP config (optimize_level=3)
  static_config.json         Static shape config (15s chunks)
  config.json                Model parameters
  vocab.txt                  SentencePiece vocabulary (8193 tokens)
  encoder-model.fp32.static.npu.onnx   Static encoder (Pad-fused, for NPU)
  decoder_joint-model.fp32.static.onnx Static decoder
```

## API Reference

### Transcribe Audio

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| file | file | Yes | Audio file (WAV, max 25MB) |
| model | string | No | Model name (accepted but ignored) |
| language | string | No | ISO-639-1 code (default: en) |
| response_format | string | No | json, text, srt, vtt, verbose_json |

### Other Endpoints

- `GET /v1/models` -- List models
- `GET /v1/info` -- Execution provider info
- `GET /health` -- Health check

## CLI Options

```
python test_transcribe.py audio.wav [options]
  --device {cpu,npu,gpu}           Encoder device (default: cpu)
  --decoder-device {auto,cpu,gpu}  Decoder device (default: auto)
  --models-dir DIR                 Models directory (default: ./models)
  --runs N                         Benchmark runs (default: 1)
  --debug                          Verbose logging

python live_transcribe.py [options]
  --device {cpu,npu,gpu}           Execution device
  --test-mic                       Test microphone levels
  --list-devices                   Show audio devices

python server.py [options]
  --device {cpu,npu}               Execution device
  --port PORT                      Server port (default: 5092)
  --host HOST                      Server host (default: 0.0.0.0)
```

## Requirements

**CPU mode:**
- Python 3.10+
- onnxruntime
- numpy, fastapi, uvicorn

**NPU mode (Ryzen AI):**
- AMD Ryzen AI processor (Strix/XDNA2)
- Windows 11
- Miniforge with `ryzen-ai-1.7.0` or `ryzen-ai-1.7.1` conda environment
- onnxruntime-vitisai, flexml-lite (included in Ryzen AI SDK)
- sounddevice (for live microphone mode)

## Troubleshooting

**VitisAI EP not available:**
```bash
conda activate ryzen-ai-<version>
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# Should show: ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
```

**NPU startup takes ~4-6 seconds:** This is normal -- VAIML loads the compiled encoder from its cache at `C:\temp\<user>\vaip\.cache\`. If the cache is missing (first run or new model), compilation will take longer.

**All ops falling back to CPU:** If you see `unknown type 9` errors or `CPU 1434` in the log, the VAIML compiler failed to partition the model. Re-run `python preprocess_for_npu.py --precision fp32` so the encoder includes Pad->Conv fusion and the VAIML 1.7.x attention-mask rewrite. This is currently tested on Strix NPUs; Strix Halo may have compatibility issues with the VAIML frontend.

**vaiml.dll not found:** Ensure flexml-lite is installed and conda env is activated. The transcriber auto-discovers it via `sys.prefix`.

**Audio format not supported:** Convert with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## Credits

- **NVIDIA** -- Parakeet TDT 0.6B model
- **Ivan Stupakov** (@istupakov) -- ONNX conversion
- **achetronic** -- Original Go implementation
- **AMD** -- Ryzen AI NPU, VitisAI EP, DirectML EP
