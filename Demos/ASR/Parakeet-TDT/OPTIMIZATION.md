# Parakeet TDT 0.6B on Ryzen AI NPU -- Optimization Journey

## Summary

We took the Parakeet TDT 0.6B speech recognition model from **6.1x real-time to 26.0x real-time** on the AMD Ryzen AI NPU -- a **4.3x improvement** through systematic optimization. The final system uses all three processors on the Ryzen AI chip simultaneously: the NPU runs the Conformer encoder, the integrated Radeon GPU runs the LSTM decoder, and the CPU handles mel feature extraction and orchestration.

**Hardware:** AMD Ryzen AI (Strix/XDNA2), Windows 11  
**Software:** ONNX Runtime 1.23.2 (VitisAI EP), flexml-lite 1.7.0, Python 3.12  
**Model:** nvidia/parakeet-tdt-0.6b (600M params, Conformer encoder + TDT LSTM decoder)

## Performance Results

| Configuration | Speed | RTF | Notes |
|---|---|---|---|
| CPU INT8 (baseline) | 17-18x | 0.056 | Zen 5, ONNX Runtime CPU EP |
| **NPU BF16 (optimized)** | **26.0x** | **0.038** | All optimizations below |

Tested on 16.5 minutes of audio (987.7 seconds). NPU beats CPU by 44%.

## Optimization Steps (in order)

### 1. Static Shape Models + NPU Graph Fixes

The VitisAI BF16 compiler (VAIML) requires static tensor shapes to compile for the NPU's AI Engine. We converted the dynamic-shape ONNX models to fixed shapes:

- **Encoder:** Fixed at 1498 mel frames (15-second audio chunks)
- **Decoder:** Fixed at batch=1, single timestep

The VAIML compiler also errored on standalone Pad ops in depthwise convs: *"Zero padding is not supported for access patterns for kernel ports."* We fused 24 `Pad -> Conv` pairs by folding padding into Conv's `pads` attribute. For VAIML 1.7.x we additionally rewrite the shared attention mask path to avoid BOOL `Slice` / `Where` patterns (`unknown type 9`).

Script: `preprocess_for_npu.py` (single step for FP32: writes `encoder-model.fp32.static.npu.onnx`)

After this, BF16 compilation succeeded and the NPU produced correct transcriptions.

**Result: NPU working at 6.1x real-time** (first unoptimized baseline)

### 2. Vectorized Mel Feature Extraction (6.1x -> 7.2x)

The mel filterbank extraction used a Python for-loop over ~1500 frames, taking 255ms per chunk (15% of total time). We replaced it with fully vectorized numpy operations:

- Stride-based frame extraction (all frames at once)
- Batch FFT with `np.fft.rfft(axis=1)`
- Matrix multiply through filterbank `power @ filterbank.T`

**Impact: 255ms -> 17ms per chunk (15x faster)**

File: `inference/mel.py`

### 3. Decoder on CPU Only (code cleanup)

We forced the decoder to always use CPUExecutionProvider instead of going through VitisAI EP. The decoder is a tiny 27-node LSTM called ~130 times per chunk -- running it through VitisAI EP added unnecessary overhead. (Profiling later showed VitisAI was already falling back to CPU for the decoder, so this was mainly a code clarity improvement.)

File: `inference/transcriber.py`

### 4. VitisAI optimize_level 2 (7.2x -> 20.7x)

Bumped the VAIML compiler optimization level from 1 to 2 in `vai_ep_config.json`. This was the single largest improvement, likely enabling better operator fusion, memory layout optimization, and instruction scheduling in the compiled NPU kernel.

Requires clearing the VAIML cache and recompiling (~30 minutes).

File: `models/vai_ep_config.json`

### 5. Pre-allocated Decoder Buffers (20.7x -> 23.1x)

The TDT decoder loop runs ~14,000 iterations for a 16-minute audio file. Each iteration was allocating new numpy arrays for inputs (targets, target_length, encoder slice) and copying state arrays. We:

- **Pre-allocated all input buffers** and mutate them in-place
- **Cached session.run reference** and input names as local variables
- **Used `np.copyto` instead of `.copy()`** for state updates (avoids allocation)
- **Pre-squeezed encoder output** to avoid per-step slicing overhead
- **Built the feed dict once** and reused it (the dict values are mutable arrays)

File: `inference/transcriber.py` (`_init_decoder_buffers`, `_tdt_decode`)

### 6. Encoder/Decoder Pipelining (concurrent with step 5)

For multi-chunk audio, we pipeline the encoder and decoder across chunks using a background thread:

```
Chunk 0: [mel+encoder on NPU]──────────[decoder on CPU]
Chunk 1:            [mel+encoder on NPU]──────────[decoder on CPU]
                    ^── decoder for chunk 0 runs here, overlapped
```

This works because ONNX Runtime releases the GIL during `session.run()`, so the NPU encoder and CPU decoder genuinely run in parallel. For 66 chunks of audio, this hides ~8 seconds of decoder time behind encoder time.

File: `inference/transcriber.py` (`_process_chunks_pipelined`)

### 7. VitisAI optimize_level 3 (23.1x -> 24.2x)

Bumped VAIML compiler to the highest optimization level. Shaved ~3 seconds off the encoder (39.9s -> 36.8s for 16.5 min of audio).

File: `models/vai_ep_config.json`

### 8. Triple-Processor Pipeline: iGPU Decoder (24.2x -> 26.0x)

Moved the decoder from CPU to the integrated AMD Radeon GPU via DirectML Execution Provider. ORT profiling confirmed all 12 decoder ops (2 LSTM + 10 fused DML nodes) run on the iGPU.

The decoder is slower per-step on iGPU (1.06ms vs 0.66ms on CPU due to GPU dispatch overhead), but the real win is **freeing the CPU** for mel feature extraction. Mel dropped from 3.9s to 1.7s because it no longer competes with the decoder for CPU cores.

Final architecture -- all three processors working simultaneously:
- **NPU** (AI Engine): Conformer encoder (36.5s, 95% of compute)
- **iGPU** (Radeon, DirectML): LSTM decoder (hidden behind encoder via pipelining)
- **CPU** (Zen 5): Mel features, audio parsing, orchestration

File: `inference/transcriber.py` (`--decoder-device gpu`)

## What Didn't Help

| Experiment | Result | Why |
|---|---|---|
| **DML IO Binding** (keep states on GPU) | Slower | Per-call bind/OrtValue creation overhead exceeded state transfer savings for tiny LSTM tensors |
| **CPU IO Binding** | Slower | Same reason -- bind overhead > dict path for small tensors |
| **preferred_data_storage: vectorized** | Same as auto | VAIML auto-selection was already optimal for the Conformer |
| **Decoder on NPU** | Not attempted | NPU can only run one model at a time -- would lose pipelining overlap |

## Profiling Insights

ORT profiling of the encoder reveals:

| Provider | Time | % | Ops |
|---|---|---|---|
| VitisAIExecutionProvider | 496.1ms | 99.8% | Single fused NPU kernel |
| CPUExecutionProvider | 0.8ms | 0.2% | Transpose, Add, Cast (trivial) |

The entire encoder compiles into **a single monolithic NPU kernel** (`vitis_ai_ep_1`). The 24 remaining Pad ops in self-attention (relative positional encoding) are handled within the NPU partition -- no partition splits, no data transfer overhead.

At ~500ms per 15-second chunk, the encoder is at the **NPU hardware limit** for this BF16 Conformer model.

## Stage Breakdown (16.5-minute audio, final configuration)

```
Mel features   :  1.7s  ( 4.4%)  -- CPU (vectorized numpy)
Encoder (NPU)  : 36.5s  (95.0%)  -- NPU (single BF16 kernel)
Decoder (iGPU) : 15.1s  (39.2%)  -- iGPU (hidden behind encoder)
Pipeline overlap: -14.8s (-38.5%) -- decoder runs during next encoder
─────────────────────────────────
Total wall time : 38.0s  (26.0x real-time)
```

## Files

| File | Purpose |
|---|---|
| `inference/transcriber.py` | Main pipeline: encoder/decoder sessions, pipelining, pre-allocated buffers |
| `inference/mel.py` | Vectorized mel filterbank extraction |
| `inference/audio.py` | WAV parsing |
| `models/vai_ep_config.json` | VitisAI EP config (optimize_level=3, data_storage=auto) |
| `models/static_config.json` | Static shape config (15s chunks, 1498 frames) |
| `preprocess_for_npu.py` | Static shapes + Pad->Conv fuse + VAIML 1.7.x mask rewrite (FP32) |
| `fuse_pads_direct.py` | Optional legacy: fuse Pad->Conv on an unfused `.static.onnx` |
| `optimize_model.py` | Experimental ORT constant folding + fusion (unfused static.onnx only) |
| `fuse_attn_pads.py` | Analyze remaining attention Pad ops |
| `test_transcribe.py` | Benchmark script with per-stage timing |
| `benchmark_npu.py` | Multi-config benchmark sweep |
| `live_transcribe.py` | Real-time microphone transcription demo |
| `server.py` | OpenAI Whisper-compatible FastAPI server |

## How to Reproduce

```powershell
# Environment
conda activate ryzen-ai-1.7.0

# Download models
python download_models.py --precision fp32

# Static shapes + NPU encoder fixes
python preprocess_for_npu.py --precision fp32

# Benchmark (first run = ~30min compile, subsequent = ~4s init)
python test_transcribe.py audio.wav --device npu --decoder-device gpu --runs 3

# Live microphone demo
python live_transcribe.py --device npu

# API server
python server.py --device npu
```

## Future Opportunities

- **INT8 quantization for NPU**: If VAIML supports INT8 on XDNA2, the encoder could be 2x faster
- **Larger batch sizes**: Process multiple audio streams in parallel on the NPU
- **VAIML compiler updates**: Future compiler versions may improve BF16 performance
- **Longer static chunks**: 30s or 60s chunks would reduce chunking overhead for long audio
