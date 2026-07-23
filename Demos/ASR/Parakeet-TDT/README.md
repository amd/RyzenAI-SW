<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Automatic Speech Recognition - Parakeet </h1>
    </td>
 </tr>
</table>

# Automatic Speech Recognition using Parakeet TDT on Ryzen AI NPU

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

**To set NPU performance mode:**

```bash
C:\Windows\System32\AMD\xrt-smi.exe configure --pmode performance
```

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

**Expected Output:**
```bash
Parakeet TDT 0.6B ONNX Model Downloader
========================================
Precision : fp32
Output    : <RyzenAI-SW>\Demos\ASR\Parakeet-TDT\models
Files     : 6

  [DOWN] Model configuration: config.json
         97.0 B / 97.0 B (100.0%) - 62.2 KB/s - ETA 0s
         Done in 0.0s
  [DOWN] SentencePiece vocabulary: vocab.txt
         91.7 KB / 91.7 KB (100.0%) - 1.1 MB/s - ETA 0s
         Done in 0.1s
  [DOWN] Mel filterbank ONNX model: nemo128.onnx
         136.5 KB / 136.5 KB (100.0%) - 953.7 KB/s - ETA 0s
         Done in 0.1s
  [DOWN] Full precision encoder (~2.4GB): encoder-model.onnx
         39.8 MB / 39.8 MB (100.0%) - 12.2 MB/s - ETA 0s
         Done in 3.3s
  [DOWN] Encoder external data: encoder-model.onnx.data
         2.3 GB / 2.3 GB (100.0%) - 13.6 MB/s - ETA 0s99ss
         Done in 171.1s
  [DOWN] Full precision TDT decoder (~72MB): decoder_joint-model.onnx
         69.2 MB / 69.2 MB (100.0%) - 23.5 MB/s - ETA 0s
         Done in 3.0s

Download complete: 6 succeeded, 0 failed
Models ready in: <RyzenAI-SW>\Demos\ASR\Parakeet-TDT\models
```

For INT8 models (CPU-only, smaller size):

```bash
python download_models.py
```

**Expected Output:**
```bash
Parakeet TDT 0.6B ONNX Model Downloader
========================================
Precision : int8
Output    : <RyzenAI-SW>\Demos\ASR\Parakeet-TDT\models
Files     : 5

  [SKIP] config.json already exists (97.0 B)
  [SKIP] vocab.txt already exists (91.7 KB)
  [SKIP] nemo128.onnx already exists (136.5 KB)
  [DOWN] Quantized encoder (~652MB): encoder-model.int8.onnx
         622.0 MB / 622.0 MB (100.0%) - 20.1 MB/s - ETA 0s
         Done in 30.9s
  [DOWN] Quantized TDT decoder (~18MB): decoder_joint-model.int8.onnx
         17.4 MB / 17.4 MB (100.0%) - 19.3 MB/s - ETA 0s
         Done in 0.9s

Download complete: 5 succeeded, 0 failed
Models ready in: <RyzenAI-SW>\Demos\ASR\Parakeet-TDT\models
```


### Step 3: Prepare Models for NPU

Convert the dynamic ONNX models to static shapes required for NPU execution:

```bash
# Convert to static shapes (required for NPU)
python convert_static.py --precision fp32
```

**Expected Output:**
```bash
ONNX version: 1.18.0
Parakeet Static Shape Converter
================================
Chunk size   : 15s
Fixed frames : 1498 mel frames
Encoded len  : ~188 frames (after 8x subsampling)

[1/2] Converting encoder...
  Loading encoder: models\encoder-model.onnx
  External data file found: models\encoder-model.onnx.data (2.3GB)
  Loading model + external data into memory (may take a minute)...
  Fixing encoder input: audio_signal -> [1, 128, 1498]
  Fixing encoder output: outputs -> [1, 1024, 188]
  Skipping shape inference (model >2GB, would strip weights)
  Saving static encoder: models\encoder-model.fp32.static.onnx
  Saving with external data: encoder-model.fp32.static.onnx.data
  Done (ONNX: 39.8 MB + data: 4645.2 MB)

[2/2] Converting decoder...
  Loading decoder: models\decoder_joint-model.onnx
  Fixing decoder inputs: encoder_outputs->[1,1024,1], targets->[1,1], states->[2,1,640]
  Fixing decoder outputs: logits->[1,1,1,8198], states->[2,1,640]
  Running shape inference...
  Saving static decoder: models\decoder_joint-model.fp32.static.onnx
  Done (69.2 MB)

Static models saved:
  Encoder: models\encoder-model.fp32.static.onnx
  Decoder: models\decoder_joint-model.fp32.static.onnx
  Config:  models\static_config.json

To use with NPU:
  python test_transcribe.py audio.wav --device npu
```

Fuse Pad operations with Conv layers for VAIML compiler compatibility:

```bash
# Fuse Pad->Conv ops and update config (required for VAIML compiler)
python fuse_pads_direct.py
```

**Expected Output:**
```bash
Loading models\encoder-model.fp32.static.onnx...
  External data: models\encoder-model.fp32.static.onnx.data (4.9GB)
Fused 24 Pad->Conv pairs
Remaining Pad ops: 24
Saving models\encoder-model.fp32.static.npu.onnx...
  With external data: encoder-model.fp32.static.npu.onnx.data
  ONNX: 41.8 MB
  Data: 4870.8 MB
Verifying model loads...
  Nodes: 4467
  Inputs: ['audio_signal', 'length']
  Outputs: ['outputs', 'encoded_lengths']
Updated models\static_config.json: encoder_model -> encoder-model.fp32.static.npu.onnx

Done! Test with:
  python test_transcribe.py audio.wav --device npu
```

### Step 4: Run Inference

#### Benchmark Mode (NPU + iGPU)

Run transcription benchmarks with detailed performance breakdown:

```bash
python test_transcribe.py test_audio.wav --device npu --decoder-device gpu --runs 3
```

**Expected Output:**
```bash
Initializing transcriber (device=npu)...
18:36:32 [INFO] inference.transcriber: Added flexml lib to PATH: C:\Users\<user_name>\AppData\Local\miniforge3\envs\asr_parakeet_env\Lib\site-packages\flexml\flexml_extras\lib
18:36:32 [INFO] inference.transcriber: Static model config: 15s chunks, 1498 fixed mel frames, encoded_len=188
18:36:32 [INFO] inference.transcriber: Using static model from config: encoder-model.fp32.static.npu.onnx
18:36:32 [INFO] inference.transcriber: Using static model from config: decoder_joint-model.fp32.static.onnx
18:36:32 [INFO] inference.transcriber: Loading encoder: models\encoder-model.fp32.static.npu.onnx
18:36:32 [INFO] inference.transcriber: Using VitisAI EP BF16 path for parakeet_encoder (config=models\vai_ep_config.json)
WARNING: Logging before InitGoogleLogging() is written to STDERR
F20260413 18:36:35.318872 35800 graph.cpp:1058] unknown type 9 name=/Slice_1_output_0
18:36:37 [INFO] inference.transcriber: Encoder loaded in 5.03s
18:36:37 [INFO] inference.transcriber: Loading decoder: models\decoder_joint-model.fp32.static.onnx (iGPU (DirectML))
18:36:38 [INFO] inference.transcriber: Decoder loaded in 0.69s (iGPU (DirectML))
18:36:38 [INFO] inference.transcriber: Transcriber initialized: device=npu, vocab_size=8193, encoder_dim=1024, static=True
Initialization: 6.26s
ONNX Runtime: 1.23.3.dev20260320
Available providers: ['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
Encoder providers: ['VitisAIExecutionProvider', 'CPUExecutionProvider']
Decoder providers: ['DmlExecutionProvider', 'CPUExecutionProvider']

18:36:39 [INFO] inference.transcriber: Transcribed 4.57s audio in 0.729s (RTF=0.159, device=npu, chunks=1): 'Today sound is designed to suit you entertainment.'
  Run 1/3: 0.729s (6.3x real-time)
18:36:39 [INFO] inference.transcriber: Transcribed 4.57s audio in 0.645s (RTF=0.141, device=npu, chunks=1): 'Today sound is designed to suit you entertainment.'
  Run 2/3: 0.645s (7.1x real-time)
18:36:40 [INFO] inference.transcriber: Transcribed 4.57s audio in 0.659s (RTF=0.144, device=npu, chunks=1): 'Today sound is designed to suit you entertainment.'
  Run 3/3: 0.659s (6.9x real-time)

============================================================
  RESULTS (NPU)
============================================================
  Audio duration : 4.6s (0.1 min)
  Device         : NPU
  Encoder        : VitisAIExecutionProvider
  Decoder        : DmlExecutionProvider
  Runs           : 3
  Average time   : 0.678s  (RTF=0.1483, 6.7x real-time)
  Best time      : 0.645s  (RTF=0.1411, 7.1x real-time)
  Worst time     : 0.729s

  --- Stage Breakdown (last run) ---
  Mel features   :     3.4ms  ( 0.5%)
  Encoder (NPU)  :   604.0ms  (91.6%)
  Decoder (GPU)  :    49.1ms  ( 7.4%)
  Other overhead :     2.8ms  ( 0.4%)
  Decoder steps  : 57  (17 tokens emitted)
  Avg per step   : 0.86ms/step

  --- Stage Breakdown (average over runs) ---
  Mel features   :     6.1ms  ( 0.9%)
  Encoder        :   612.5ms  (90.3%)
  Decoder        :    55.6ms  ( 8.2%)
============================================================

  TRANSCRIPT
------------------------------------------------------------
  Today sound is designed to suit you entertainment.
------------------------------------------------------------

18:36:40 [INFO] inference.transcriber: Transcriber closed
```

#### Live Microphone Transcription

Transcribe speech in real-time from your microphone:

```bash
# Install sounddevice for microphone support
pip install sounddevice

# Run live transcription
python live_transcribe.py --device npu
```

**Expected Output:**

```bash
Microphone: Microphone Array (AMD Audio Dev (2ch)
Loading Parakeet TDT 0.6B (NPU)...
WARNING: Logging before InitGoogleLogging() is written to STDERR
F20260413 18:38:23.849195 45820 graph.cpp:1058] unknown type 9 name=/Slice_1_output_0
Model loaded in 4.2s
Encoder: VitisAIExecutionProvider
Decoder: CPUExecutionProvider

  Calibrating ambient noise (2.0s) -- stay quiet...
  Ambient RMS : 0.00001
  Threshold   : 0.01001
  Headroom    : 934.1x ambient

============================================================
  LIVE TRANSCRIPTION -- speak into your microphone!
  Level meter: [:::T....] T=threshold, :|=level
  Press Ctrl+C to stop.
============================================================

  [.T......................................] 0.0000        [  1] (2.7s -> 0.69s = 4x) Hello Rising V
  [  1] (2.5s -> 0.83s = 3x) Hello, how are you?
  [.T......................................] 0.0000

============================================================
  SESSION TRANSCRIPT
============================================================
    1. Hello, how are you?
============================================================
```

#### API Server Mode

Start a Whisper-compatible REST API server:

```bash
# Start the FastAPI server on NPU
python server.py --device npu
```

The server starts on `http://localhost:5092` and provides an OpenAI Whisper-compatible API endpoint.

**Expected Output:**
```
Loading Parakeet TDT 0.6B on NPU...
Model loaded successfully
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5092
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

convert_static.py            Convert dynamic ONNX to static shapes for NPU
fuse_pads_direct.py          Fuse Pad->Conv for VAIML compiler compatibility
optimize_model.py            ORT constant folding + fusion
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

- `GET /v1/models` — List available models
- `GET /v1/info` — Execution provider information
- `GET /health` — Health check

## CLI Options

### test_transcribe.py

```
python test_transcribe.py audio.wav [options]
  --device {cpu,npu,gpu}           Encoder device (default: cpu)
  --decoder-device {auto,cpu,gpu}  Decoder device (default: auto)
  --models-dir DIR                 Models directory (default: ./models)
  --runs N                         Benchmark runs (default: 1)
  --debug                          Verbose logging
```

### live_transcribe.py

```
python live_transcribe.py [options]
  --device {cpu,npu,gpu}           Execution device
  --test-mic                       Test microphone levels
  --list-devices                   Show audio devices
```

### server.py

```
python server.py [options]
  --device {cpu,npu}               Execution device
  --port PORT                      Server port (default: 5092)
  --host HOST                      Server host (default: 0.0.0.0)
```

## Requirements

**NPU mode:**
- Python 3.10+
- onnxruntime-vitisai (included in Ryzen AI SDK)
- flexml-lite (included in Ryzen AI SDK)
- numpy, fastapi, uvicorn
- sounddevice (for microphone input)

**CPU mode:**
- Python 3.10+
- onnxruntime
- numpy, fastapi, uvicorn

## Troubleshooting

### VitisAI EP not available

Verify that the VitisAI Execution Provider is available in your environment:

```bash
conda activate ryzen-ai-<version>
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

**Expected Output:**
```
['VitisAIExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
```

### NPU startup takes ~4-6 seconds

This is normal behavior. VAIML loads the compiled encoder from its cache at `C:\temp\<user>\vaip\.cache\`. If the cache is missing (first run or new model), compilation will take longer (~15-20 minutes on first run).

### All ops falling back to CPU

If you see `unknown type 9` errors or `CPU 1434` in the log, the VAIML compiler failed to partition the model. Ensure you ran `python fuse_pads_direct.py` after `convert_static.py`. This is currently tested on Strix NPUs; Strix Halo may have compatibility issues with the VAIML frontend.

### vaiml.dll not found

Ensure flexml-lite is installed and conda environment is activated. The transcriber auto-discovers it via `sys.prefix`.

### Audio format not supported

Convert your audio file to WAV format using ffmpeg:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

This converts the audio to 16kHz sample rate, mono channel WAV format required by Parakeet.
