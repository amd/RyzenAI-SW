<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI Automatic Speech Recognition </h1>
    </td>
 </tr>
</table>

# Automatic Speech Recognition using OpenAI Whisper

Unlock fast, on-device speech recognition with RyzenAI and OpenAI’s Whisper. This demo walks you through preparing and running OpenAI's Whisper (base, small, medium) for fast, local ASR on AMD NPU.

## Features

* 🚀 Export Whisper models from Hugging Face to ONNX
* ⚙️ Optimize for static shape inference
* ⚡ Run ASR locally on CPU or NPU
* 📊 Evaluate ASR on LibriSpeech samples and report WER/CER
* 🎧 Supports transcription of audio files and microphone input
* ⏱️ Reports Performance using RTF and TTFT

## 🔗 Quick Links
- [Prerequisites](#prerequisites)
- [Export Whisper Model to ONNX](#export-whisper-model-to-onnx)
- [Accelerate Whisper on AMD NPU](#accelerate-whisper-on-amd-npu)
  - [Why run on NPU?](#why-run-on-npu)
  - [Set up VitisEP Configuration for NPU](#set-up-vitisep-configuration-for-npu)
- [ Usage](#usage)
  - [Transcribe Audio File](#transcribe-audio-file)
  - [Transcribe from Microphone](#transcribe-from-microphone)
  - [Evaluate on Dataset](#evaluate-on-dataset)
- [ Notes](#notes)

## 📦 Prerequisites

1. **Install Ryzen AI SDK**
   Follow [RyzenAI documentation](https://ryzenai.docs.amd.com/en/latest/inst.html#) to install SDK and drivers.

2. **Activate environment**

   ```bash
   conda activate ryzen-ai-1.5.0
   ```

3. **Clone repository**

   ```bash
   git clone https://github.com/amd/RyzenAI-SW.git
   cd RyzenAI-SW/example/ASR/Whisper-AI
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```


## ⚙️ Export Whisper Model to ONNX

1. **Export using Hugging Face Optimum**

   ```bash
   optimum-cli export onnx --model openai/whisper-base.en --opset 17 exported_model_directory
   ```

   * Output: `encoder_model.onnx`, `decoder_model.onnx`
   * Supports multilingual whisper-base, whisper-small, whisper-medium

2. **Convert dynamic ONNX to static**

   ```bash
   python dynamic_to_static.py --input_model exported_model_directory/encoder_model.onnx
   python dynamic_to_static.py --input_model exported_model_directory/decoder_model.onnx
   ```

   * Uses `onnxruntime.tools.make_dynamic_shape_fixed`
   * Final models overwrite originals in `exported_model_directory`

## ⚡Accelerate Whisper on AMD NPU

### Why run on NPU?

* Offloads compute from CPU onto NPU, freeing up CPU for other tasks.
* Delivers higher throughput and lower power consumption when running AI workloads
* Optimized execution of Whisper’s encoder and decoder models.
* Runs models with BFP16 precision for near-FP32 accuracy and INT8-like performance.

#### NPU Run for Whisper-Base
When running inference on the NPU, 100% of the encoder operators and 93.4% of the decoder operators are executed on the NPU.
```bash
   #encoder operations
   [Vitis AI EP] No. of Operators : VAIML   225
   [Vitis AI EP] No. of Subgraphs : VAIML     1

   #decoder operations
   [Vitis AI EP] No. of Operators :   CPU    24  VAIML   341
   [Vitis AI EP] No. of Subgraphs : VAIML     2
```
#### Set up VitisEP Configuration for NPU

* Edit `config/model_config.json` to specify Execution Providers.
* For NPU:

  * Set `cache_key` and `cache_dir`
  * Use corresponding `vitisai_config` from `config/`

Example:

```json
{
  "config_file": "config/whisper_vitisai.json",
  "cache_dir": "./cache",
  "cache_key": "whisper_base"
}
```
#### ⚠️ Special Instructions for Whisper-Medium
When running whisper-medium on NPU, it is recommended to add the following flags to `configs\vitisai_config_whisper_encoder.json` incase of compilation issues.
```json
"vaiml_config": {
  "optimize_level": 2,
  "aiecompiler_args": "--system-stack-size=512"
}
```
These settings:

- optimize_level=2: Enables aggressive optimizations for larger models.
- --system-stack-size=512: Increases the AI Engine system stack size to handle Whisper-Medium’s higher resource demand.

## 🚀 Usage

### Transcribe Audio File
Use this to transcribe a pre-recorded `.wav` file into text using the Whisper mode
```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --input path/to/audio.wav
```
- Replace <whisper-type> with whisper-base, whisper-small, or whisper-medium.

- Replace path/to/audio.wav with your audio file.

### Transcribe from Microphone
Run real-time speech-to-text by capturing audio from your microphone. This allows you to speak and see live transcription:

```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --input mic \
  --duration 0
```
- --duration 0 means continuous recording until stopped (Ctrl+C) or detects silence for a set duration

- Ideal for demos and testing live ASR performance.

### Evaluate on Dataset
Run batch evaluation on a dataset (e.g., LibriSpeech samples) to measure model performance with metrics like WER, CER, and RTF:
```bash
python run_whisper.py \
  --encoder exported_model_directory/encoder_model.onnx \
  --decoder exported_model_directory/decoder_model.onnx \
  --model-type <whisper-type> \
  --config-file config/model_config.json \
  --device npu \
  --eval-dir eval_dataset/LibriSpeech-samples \
  --results-dir results
```
- --eval-dir specifies the dataset directory.

- --results-dir is where evaluation reports (WER, CER, TTFT, RTF) will be saved.

- Useful for benchmarking and validating models.

## Notes

* First run on NPU may take \~15 min for model compilation.
* Ensure paths for encoder, decoder, and config files are correct.
* Supports CPU and NPU devices.

