<table class="sphinxhide" width="100%">
 <tr width="100%">
    <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1> Ryzen™ AI ASR </h1>
    </td>
 </tr>
</table>

# Running Whisper on Ryzen AI

This Ryzen AI example lets you bring in OpenAI’s Whisper model and run fast, local automatic speech recognition (ASR) on your AMD NPU. Whisper is a versatile speech model trained on 680,000+ hours of diverse audio, capable of speech-to-text, translation, and language detection.  
This example uses the [Whisper-base](https://huggingface.co/openai/whisper-base) variant and provides a simple demonstration of how to run it on the NPU. For real-time factor (RTF) evaluation of the model on the NPU, please refer to the [whisper-demo](https://github.com/amd/RyzenAI-SW/tree/main/demo/ASR/Whisper).

Learn how you can:  
- **Export Whisper models** from Hugging Face to ONNX format  
- **Optimize** them for static shape inference  
- **Run ASR** fully on-device using CPU or AMD NPU
- **Evaluate ASR** performance on sample data from public datasets like LibriSpeech.  

This example supports:    
- **Audio file transcription** – load your own `.wav` files for instant speech-to-text

## Prerequisites
**Step 1:**  Install the latest Conda environment using [RyzenAI Documentation](https://ryzenai.docs.amd.com/en/latest/inst.html#). 
Ensure the SDK and driver are installed. 

**Step 2:** Export Hugging face Whisper model to onnx and set static shape as mentioned below:
1. Activate conda environment:
```bash
   conda activate ryzen-ai-<version>
```
2. Clone the repository and navigate to the Whisper-AI directory:
```bash
   git clone https://github.com/amd/RyzenAI-SW.git
   cd \path\to\RyzenAI-SW\example\ASR\Whisper-AI
```
3. Install the necessary libraries:
```bash
   pip install -r requirements.txt
   ```
4. Export Whisper AI model to ONNX using [Hugging Face Optimum library](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model):
```bash
   optimum-cli export onnx --model openai/whisper-base.en --opset 17 exported_model_directory
   ```
**Note:** 
The above command creates a new directory `exported_model_directory` in the current path. In `exported_model_directory`, you should see `encoder_model.onnx` and `decoder_model.onnx` models available.

5. Convert the dynamic ONNX model to static using the `dynamic_to_static.py` script.
```bash
   #Convert the encoder
   python dynamic_to_static.py --input_model ".\exported_model_directory\encoder_model.onnx"
   
   #Convert the decoder
   python dynamic_to_static.py --input_model ".\exported_model_directory\decoder_model.onnx"
   ```
The `dynamic_to_static.py` script utilizes `onnxruntime.tools.make_dynamic_shape_fixed` to convert dynamic shapes in an ONNX model to static shapes. It takes as input a `params.json` file, which specifies the dynamic dimensions to be fixed and their target static values. After the conversion, the script verifies the correctness of the modified ONNX model using the ONNX Checker and performs a dummy inference to ensure the model runs as expected.

The `params.json` file defines the static shapes used to convert a dynamic Whisper-base ONNX model to a fixed-shape version suitable for optimized inference on NPUs.

```bash
   {
    "batch_size": "1",
    "encoder_sequence_length / 2": "1500",
    "decoder_sequence_length": "180"
}
```
- `"batch_size": "1"` - Fixes the model to process one audio sample at a time.  
- `"encoder_sequence_length / 2": "1500"` - Whisper converts audio to a log-Mel spectrogram with 3000 frames for 30s of audio. After 2× downsampling, the encoder input length becomes 1500. This is fixed in params.json for optimized static-shape inference.  
- `"decoder_sequence_length": "180"` - Fixed to 180 to match 30s of audio input (3000 tokens). At ~5 tokens/sec, average output is 150 tokens; 30-token buffer ensures completeness and handles variation

**Note:** The final static ONNX models are stored in `.\exported_model_directory\encoder_model.onnx` and `.\exported_model_directory\decoder_model.onnx`.

## Whisper ONNX Inference and Evaluation

The `run_whisper.py` script performs speech-to-text transcription using a Whisper-base model exported to ONNX format. It supports transcribing audio from WAV files or a live microphone stream and can evaluate model accuracy on a labeled dataset using WER and CER metrics. The script runs the encoder and decoder models via ONNX Runtime, with support for both CPU and NPU backends, and includes chunk-based processing for long audio inputs.

The `load_provider_options` function returns ONNX Runtime execution providers and configuration options based on the selected device (cpu or npu).

```bash
   provider = "VitisAIExecutionProvider"
        
   encoder_options = {
            "config_file": "vaiep_config.json",
            "cache_dir": "./cache/",
            "cache_key": "whisper_encoder"
        }
        
   decoder_options = {
            "config_file": "vaiep_config.json",
            "cache_dir": "./cache/",
            "cache_key": "whisper_decoder"
        }
```
When running on the NPU, the provider options for the encoder and decoder are identical, except for the cache directory used. Both utilize the official RAI `vaiep_config.json` [configuration file](https://ryzenai.docs.amd.com/en/latest/modelrun.html#config-file-options).

When running inference on the NPU, 100% of the encoder operators and 93.4% of the decoder operators are executed on the NPU.

```bash
   #encoder operations
   [Vitis AI EP] No. of Operators : VAIML   225
   [Vitis AI EP] No. of Subgraphs : VAIML     1
   
   #decoder operations
   [Vitis AI EP] No. of Operators :   CPU    24  VAIML   341
   [Vitis AI EP] No. of Subgraphs : VAIML     2
```

Command to run transcription using `.wav` file or microphone:
```bash
   python run_whisper.py \
    --encoder exported_model_directory\encoder_model.onnx \
    --decoder exported_model_directory\decoder_model.onnx \
    --device <cpu|npu> \
    --input <audio_files\.wav|"mic">
```

### Expected Output

Run the above command with sample audio file and observe the expected Model output below

--input audio_files\61-52s.wav

```bash
Transcription: Also, there was a stripling page who turned into a maze with so sweet a lady, sir. 
And in some manner, I do think she died. But then the picture was gone as quickly as it came. 
Sister Nell, do you hear these models? Take your place and let us see what the crystal can show to you, like is not young, Master. 
Though I am an old man. With all rant the opening of the tent to see what might be a miss. 
But Master Will, who peeped out first, needed no more than one glance. 
Mistress Fitzsooth to the rear of the Ted cries of "A knotting ham! A knotting ham!" before them fled the stroller and his three sons, capless and tear away.
"What is that tumult and rioting?" cried out the squire, authoritatively, and he blew twice on the silver whistle which hung at his belt.

```

### Model Evaluation

To evaluate model performance, we provide an eval_dataset directory containing sample audio from the LibriSpeech dataset. You can run the following command to generate a detailed report including WER, and CER metrics:
```bash
python run_whisper.py \
    --encoder exported_model_directory\encoder_model.onnx \
    --decoder exported_model_directory\decoder_model.onnx \
    --device <cpu|npu> \
    --eval-dir eval_dataset\LibriSpeech-samples \
    --results-dir <output_results_dir>
```

### Notes

- If the model has not been precompiled before, the first run will take approximately 15 minutes to compile.
- Ensure that the paths to the encoder, decoder, and configuration file are correctly set based on your environment.
