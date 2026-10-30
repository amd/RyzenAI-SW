# VLM Benchmark

Benchmarking and inference tools for Vision Language Models (VLMs) using ONNX Runtime GenAI.

**Scripts provided:**
- **`vlm_run.py`** - Quick inference to test a model and see output
- **`vlm_benchmark.py`** - Run benchmark for a single image/size  
- **`run_all_benchmarks.py`** - Run benchmarks across multiple image sizes and save results


## Metrics Measured

| Metric | Description |
|--------|-------------|
| **Preprocessing Time** | Time spent in multimodal processor (image preprocessing) |
| **TTFT** | Time To First Token - includes prefill + first token generation |
| **TPS (Decode)** | Tokens per second during decode phase |
| **Prefill Throughput** | Tokens per second during prefill (prompt processing) |
| **Image Tokens** | Number of tokens generated from the image |

## Requirements

Step 1: Activate the Ryzen AI installation environment
```bash
"C:\Program Files\RyzenAI\ryzen-ai-1.9.0\Scripts\activate.bat"
set "RYZEN_AI_INSTALLATION_PATH=C:\Program Files\RyzenAI\ryzen-ai-1.9.0"
```

Step 2: Export the installed Python packages
```bash
python -m pip freeze > "%TEMP%\ryzenai_1.9_requirements.txt"
```

Step 3: Create a virtual environment for vlm-bench
```bash
cd /d <RyzenAI-SW>\LLM-examples\VLM
python -m venv vlm-bench
```

Step 4: Deactivate the Ryzen AI installation environment
```bash
deactivate
```

Step 5: Activate the example environment
```bash
"vlm-bench\Scripts\activate.bat"
```

Step 6: Install environment dependencies
```bash
python -m pip install onnxruntime-genai pillow numpy psutil
```

> **Note:** The above setup is for CPU models. To run AMD NPU models, activate the Ryzen AI python virtual environment instead.

## Quick Start

### 1. Generate a test image

```bash
python generate_test_image.py
```

This creates `test_image.jpg` (4032x4032) with varied visual patterns for benchmarking.


### 2. Quick inference (test model output)

```bash
python vlm_run.py -m /path/to/model -i image.jpg
```

### 3. Run benchmark for a single size

```bash
python vlm_benchmark.py \
  -m /path/to/model \
  -i test_image.jpg \
  --image_size 1024 1024 \
  --max_tokens 50 \
  -n 3 \
  -w 1 \
  -o results.json \
  -v
```

### 4. Run benchmarks across multiple image sizes

```bash
python run_all_benchmarks.py -m /path/to/model
```

Or run specific sizes with verbose output:

```bash
python run_all_benchmarks.py -m /path/to/model --sizes 1024x1024 2048x2048 -v
```

## Scripts

### `vlm_run.py`

Simple inference script to quickly test a model and see output. No benchmarking, just runs the model once.

```
usage: vlm_run.py [-h] -m MODEL_PATH -i IMAGE_PATH
                  [-p PROMPT] [-e {cpu,dml,follow_config}]
                  [--image_size WIDTH HEIGHT]
                  [--max_tokens MAX_TOKENS] [--max_length MAX_LENGTH]
                  [--show_prompt]

Arguments:
  -m, --model_path      Path to the ONNX model folder
  -i, --image_path      Path to input image
  -p, --prompt          Prompt text (default: "Describe this image in detail.")
  -e, --execution_provider
                        Execution provider: cpu, dml, follow_config
  --image_size          Resize image to this size (width height)
  --max_tokens          Max tokens to generate (default: 256)
  --max_length          Max context length (default: 16384)
  --show_prompt         Print the full prompt before generation
```

**Examples:**
```bash
# Basic inference
python vlm_run.py -m ./model -i image.jpg

# Custom prompt
python vlm_run.py -m ./model -i image.jpg -p "What's in this image?"

# Resize image before running
python vlm_run.py -m ./model -i image.jpg --image_size 1024 1024

# Generate more tokens
python vlm_run.py -m ./model -i image.jpg --max_tokens 512

# Debug prompt formatting
python vlm_run.py -m ./model -i image.jpg --show_prompt
```

### `vlm_benchmark.py`

Benchmark engine for a single image/size combination. Use this for quick single-size tests.

```
usage: vlm_benchmark.py [-h] -m MODEL_PATH -i IMAGE_PATH
                        [-e {cpu,dml,follow_config}]
                        [-p PROMPT] [--image_size WIDTH HEIGHT]
                        [--max_tokens MAX_TOKENS] [--max_length MAX_LENGTH]
                        [-n NUM_ITERATIONS] [-w WARMUP_ITERATIONS]
                        [-o OUTPUT_JSON] [-v]

Arguments:
  -m, --model_path      Path to the ONNX model folder
  -i, --image_path      Path to input image
  -e, --execution_provider
                        Execution provider: cpu, dml, follow_config
  -p, --prompt          Prompt text for the VLM
  --image_size          Target image size (width height)
  --max_tokens          Max tokens to generate (default: 100)
  --max_length          Max context length (default: 16384)
  -n, --num_iterations  Benchmark iterations (default: 5)
  -w, --warmup_iterations
                        Warmup iterations (default: 2)
  -o, --output_json     Export results to JSON
  -v, --verbose         Enable verbose output (shows generated text)
```

**Examples:**
```bash
# Quick test
python vlm_benchmark.py -m ./model -i image.jpg --image_size 1024 1024

# Full benchmark with JSON export
python vlm_benchmark.py -m ./model -i image.jpg --image_size 1024 1024 \
  -n 5 -w 2 --max_tokens 100 -o results.json -v
```

### `run_all_benchmarks.py`

Batch runner for benchmarking across multiple image sizes. Outputs:
- `benchmark_<size>.json` - per-size results
- `all_benchmarks_combined.json` - combined JSON
- `summary.md` - markdown table

```
usage: run_all_benchmarks.py [-h] -m MODEL_PATH [-i IMAGE] [-o OUTPUT_DIR]
                             [-p PROMPT] [--max_tokens MAX_TOKENS]
                             [--max_length MAX_LENGTH]
                             [-n NUM_ITERATIONS] [-w WARMUP_ITERATIONS]
                             [--sizes SIZES [SIZES ...]] [-v] [--compact]

Arguments:
  -m, --model_path      Path to the ONNX model folder (required)
  -i, --image           Source image path (default: ./test_image.jpg)
  -o, --output_dir      Output directory for results (default: ./benchmark_results)
  -p, --prompt          Prompt text
  --max_tokens          Max tokens to generate (default: 30)
  --max_length          Max context length (default: 16384)
  -n, --num_iterations  Benchmark iterations (default: 2)
  -w, --warmup_iterations Warmup iterations (default: 1)
  --sizes               Specific sizes to benchmark (e.g., 1024x1024 2048x2048)
  -v, --verbose         Enable verbose output
  --compact             Print compact summary (Total Tokens, Image Tokens, TTFT, Decode TPS)
```

**Examples:**
```bash
# Run all sizes (448x448 to 4032x4032)
python run_all_benchmarks.py -m ./phi-4-mm-onnx

# Custom image and output directory
python run_all_benchmarks.py -m ./model -i my_image.jpg -o ./my_results

# Run specific sizes only
python run_all_benchmarks.py -m ./model --sizes 1024x1024 2048x2048

# Verbose mode (shows generated text)
python run_all_benchmarks.py -m ./model -v --sizes 1024x1024

# Compact output (simplified table)
python run_all_benchmarks.py -m ./model --compact
```

### `generate_test_image.py`

Generates synthetic test images with visual complexity for benchmarking.

The generated image includes:
- Charts (bar, line, pie)
- Text at multiple sizes
- Geometric shapes and patterns 
- Varying complexity regions

```bash
# Default: 4032x4032
python generate_test_image.py

# Custom size and output
python generate_test_image.py -s 2048 -o my_test_image.jpg
```

## Output Example

### Single Benchmark Output

```
============================================================
VLM BENCHMARK RESULTS
============================================================
Model Type: phi4mm
Image Size: 1024x1024
Number of runs: 3

Token Breakdown:
  Total Prompt Tokens: 2635
  Text Tokens:         16
  Image Tokens:        2619

Preprocessing Time (image processing):
  Average:       74.41 ms

Time To First Token (TTFT):
  Average:    46948.55 ms

Prefill Throughput:
  Average:       56.97 tokens/sec

Token Generation (excluding first token):
  Average:      277.03 ms/token
  TPS:            3.61 tokens/sec

Peak Memory Usage:
       8192.45 MB (8.00 GB)
============================================================
```

### Batch Benchmark Summary Table

```
==============================================================================================================
BENCHMARK SUMMARY
==============================================================================================================
Model:  ./phi-4-mm-fine-tuned-onnx
Image:  ./test_image.jpg
Prompt: "Describe this image in detail."
Text tokens (constant across sizes): 16

Size         Img Tok    Total Tok   Preproc (ms)   TTFT (s)     Prefill        Decode         Decode Lat
                                                                (tokens/s)     (tokens/s)     (ms/token)
--------------------------------------------------------------------------------------------------------------
448x448      539        555         9.76           8.94         62.10          4.05           246.94
1024x1024    2619       2635        74.41          46.95        56.97          3.61           277.03
2048x2048    5765       5781        198.19         137.52       42.05          2.29           436.58
==============================================================================================================
```

### Compact Output (`--compact`)

```
============================================================
BENCHMARK SUMMARY
============================================================
Model:  ./phi-4-mm-fine-tuned-onnx
Image:  ./test_image.jpg
Prompt: "Describe this image in detail."
Text tokens (constant across sizes): 16

Size         Total Tok   Img Tok    TTFT (s)     TPS
                                                 (tok/s)
------------------------------------------------------------
448x448      555         539        8.95         4.05
1024x1024    2635        2619       47.02        3.61
2048x2048    5781        5765       137.72       2.29
============================================================
```

## JSON Output Format

Results exported with `-o` include:

```json
{
  "config": {
    "model_path": "...",
    "model_type": "phi4mm",
    "image_size": [1024, 1024],
    "prompt": "...",
    "max_tokens": 50,
    "max_length": 16384
  },
  "summary": {
    "token_breakdown": {
      "prompt_tokens": 2635,
      "text_tokens": 16,
      "image_tokens": 2619
    },
    "preprocessing_ms": { "avg": 74.41, "p50": 74.95, ... },
    "ttft_ms": { "avg": 46948.55, "p50": 47005.69, ... },
    "prefill_throughput_tps": { "avg": 56.97, "p50": 57.06 },
    "token_generation_ms": { "avg": 277.03, "tps_avg": 3.61, ... },
    "peak_memory_bytes": 8589545472,
    "peak_memory_mb": 8192.45
  },
  "raw_measurements": {
    "preprocessing_times_ms": [...],
    "ttft_times_ms": [...],
    "token_gen_times_ms": [...]
  }
}
```