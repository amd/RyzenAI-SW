# Running Whisper-tiny on Ryzen AI

This is an example of running [whisper-tiny](https://huggingface.co/openai/whisper-tiny) model on Ryzen-AI Software platform. 


# whisper-onnx

### 1. Prepare conda environment
- Create a new conda environment based on py39, install the packages, and activate it:
```powershell
cd whisper-tiny
conda env create --name your_conda_env_name -f environment.yml
conda activate your_conda_env_name
```

### 2. Install onnxruntime and Vitis AI Execution Provider:
Download the setup package ``ryzen-ai-sw-0.9.tar.gz`` and extract it. 

https://account.amd.com/en/forms/downloads/ryzen-ai-software-platform-xef.html?filename=ryzen-ai-sw-0.9.zip

Copy the extracted setup package to current directory whisper-tiny.

The onnxruntime and Vitis AI Execution Provider should now be at - ``example\Whipser-tiny\ryzen-ai-sw-0.9\ryzen-ai-sw-0.9``

### 3. Build whisper-onnx
- Use the above conda environment then execute following command:

```powershell
cd whisper-tiny
```
- Use powershell to execute this command
```powershell
./build.ps1
```

- ``build.ps1`` - This script downloads a preprocessed dataset into ``temp_download``, the model - both floating point and quantized version of the encoder and the decoder into ``models``, and other necessary packages. 

- When whisper app is built successfully, the console output would be as shown below:
```powershell
...
build whisper-onnx demo success.
Build whisper-onnx success, you could run it now.
```


## 4. Transcribe audios
In this release version, it is suggestted that you transcribe audios with duration less than 10.24 seconds. One more limitation is that it recognizes only English speech in this release. 

### 4.1 Supported CLI options
- Execute this command to fetch all supported modes:
```powershell
whisper -h
```
- Supported options are shown as below:
```powershell
whisper -h
usage: whisper [-h] [--audio [AUDIO ...]] [--output_dir OUTPUT_DIR] [--target {aie-cpu,cpu-aie}] [--librispeech LIBRISPEECH] [--test_num TEST_NUM]

optional arguments:
  -h, --help            show this help message and exit
  --audio [AUDIO ...]   Specify the path to at least one or more audio files (wav, mp4, mp3, etc.). e.g. --audio aaa.mp4 bbb.mp3 ccc.mp4 (default: None)
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        directory to save the outputs (default: .)
  --target {aie-cpu,cpu-aie}
                        which target to run encoder and decoder models (default: cpu-aie)
  --librispeech LIBRISPEECH
                        test WER of LibriSpeech dataset if you set path of LibriSpeech dataset. (default: None)
  --test_num TEST_NUM   dataset samples count to calculate WER, 0 means whole dataset. (default: 0)
```

- More details about `target` argument:

| options | note |
|---------|------|
|`cpu-aie`|Float encoder model would be run with ORT CPU EP, quantized decoder model runs on IPU.|
|`aie-cpu`|Quantized encoder model runs on IPU, float decoder model runs with ORT CPU EP.|

**Note**: ``aie`` refers to the IPU in the ``--target`` argument. 

### 4.2 Decode your own audio
- Assign the audio path to `--audio` to decode a specific audio. An audio sample `test_audio.m4a` is made available in the `Whisper-tiny` directory. To decode the audio sample follow the below instruction:
```powershell
# this example means that the wav file would be run with quantized encoder onnx model on AIE and float decoder onnx model on ORT CPU EP
whisper --audio .\test_audio.m4a --target aie-cpu
```
- The console output would be as shown below, the decoding results would be saved into a text file in the current path with the same name as audio file. You could specify the folder path through `--output_dir` and the results would be saved into this specified folder path.
```powershell
---------------------result----------------------
Prediction Result:   Hi, welcome to experience AMD IPU.
Encoder running time: 0.04s, Decoder running time: 0.36s, Other process time: 0.32s
Real time factor: 0.081, Audio duration: 8.84s, Decoding time: 0.72s
-------------------------------------------------
[Info] Save the transcribe result into .\test_audio.m4a.txt successfully.
```
- The saved text file would be looked like this:
```text
Audio file name: test_audio.m4a
Prediction:  Hi, welcome to experience AMD IPU.
Encoder running time: 0.04s, Decoder running time: 0.36s, Other process time: 0.32s
Real time factor: 0.081, Audio duration: 8.84s, Decoding time: 0.72s
```


## 5. Test on LibriSpeech
LibriSpeech is a speech recognition dataset with diverse reading materials, various accents, and speakers, commonly used for speech processing research. If audio duration is longer than 10 seconds, it would be skipped when testing on this dataset in this release version.

- Assign the JSON file path which includes audios' metainfos and the number of test cases to `--librispeech` and `--test_num` respectively, if `--test_num` is not set or set to 0 would it test the whole dataset.
```powershell
# use our preprocessed dataset
whisper --librispeech librispeech-test-clean-wav.json --test_num 10 --target cpu-aie

```

- The console output would be as shown below, while the decoding results would be saved into a JSON in current path which is named `test_librispeech_results.json`. You also could assign `--output_dir` to specify the saved folder path.
```powershell
# Take `test_num=3` as an example
Warm up some steps...
warm up: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:09<00:00,  3.16s/audio]
Warm up done.
Dataset:   0%|                                                                                                                                | 0/3 [00:00<?, ?audio/s]---------------------result----------------------
transcript:  congratulations were poured in upon the princess everywhere during her journey
prediction:  congratulations we report in the p on the princess everywhere during her journey
Encoder running time: 0.53s, Decoder running time: 0.59s, Other process time: 0.3s
real time factor: 0.284, Audio duration: 5.01s, Decoding time: 1.42s
-------------------------------------------------
Dataset:  33%|████████████████████████████████████████                                                                                | 1/3 [00:01<00:02,  1.42s/audio]---------------------result----------------------
transcript:  this has indeed been a harassing day continued the young man his eyes fixed upon his friend
prediction:  this has indeed been a harassing day continued the young man his eyes fixed upon his friend
Encoder running time: 0.54s, Decoder running time: 0.73s, Other process time: 0.36s
real time factor: 0.278, Audio duration: 5.84s, Decoding time: 1.63s
-------------------------------------------------
Dataset:  67%|████████████████████████████████████████████████████████████████████████████████                                        | 2/3 [00:03<00:01,  1.54s/audio]---------------------result----------------------
transcript:  you will be frank with me i always am
prediction:  you will be frank with me i always am
Encoder running time: 0.56s, Decoder running time: 0.39s, Other process time: 0.3s
real time factor: 0.38, Audio duration: 3.3s, Decoding time: 1.25s
-------------------------------------------------
Dataset: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:04<00:00,  1.44s/audio]
WER>>>>>:0.1351, error scores: 5, total words in test set: 37, total samples: 3
RTF Average: 0.314, RTF 50%: 0.284, RTF 90%: 0.361, RTF 99%: 0.378
total encoder running time: 1.63s
total decoder running time: 1.71s
total other process time: 0.96s
Total decoding time: 4.3s
[Info] Test results are saved into .\test_librispeech_results.json successfully.
```

- The saved result JSON file looks like as shown below:
```json
[{"Final Results": {"word error rate": 0.1351, "total test samples": 3, "error scores": 5, "total test words": 37, "RTF Average": 0.314, "RTF 50%": 0.284, "RTF 90%": 0.361, "RTF 99%": 0.378, "total decoding time": "4.3s", "total encoder running time": "1.63s", "total decoder running time": "1.71s", "total other process time": "0.96s"}}, {"fname": "test-clean-wav/6930/75918/6930-75918-0002.wav", "transcript": "congratulations were poured in upon the princess everywhere during her journey", "prediction": "congratulations we report in the p on the princess everywhere during her journey", "real time factor": 0.284, "audio duration": "5.01s", "decoding time": "1.42s", "encoder running time": "0.53s", "decoder running time": "0.59s", "other process time": "0.3s"}, {"fname": "test-clean-wav/6930/75918/6930-75918-0006.wav", "transcript": "this has indeed been a harassing day continued the young man his eyes fixed upon his friend", "prediction": "this has indeed been a harassing day continued the young man his eyes fixed upon his friend", "real time factor": 0.278, "audio duration": "5.84s", "decoding time": "1.63s", "encoder running time": "0.54s", "decoder running time": "0.73s", "other process time": "0.36s"}, {"fname": "test-clean-wav/6930/75918/6930-75918-0007.wav", "transcript": "you will be frank with me i always am", "prediction": "you will be frank with me i always am", "real time factor": 0.38, "audio duration": "3.3s", "decoding time": "1.25s", "encoder running time": "0.56s", "decoder running time": "0.39s", "other process time": "0.3s"}]
```


